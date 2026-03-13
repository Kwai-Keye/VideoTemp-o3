"""
Shared utility functions for all benchmark evaluation scripts.
"""
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from typing import Tuple

import cv2
import openai
from decord import VideoReader


# ==============================================================================
# Constants
# ==============================================================================
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 64
FRAME_FACTOR = 2
FPS = 2
MAX_ITERATIONS = 3

# ==============================================================================
# Prompts
# ==============================================================================
PREFIX_PROMPT = """You are a helpful assistant.

Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

If analyzing a specific video segment is necessary to answer the question, you may use the following tool to extract a clip from `[start_time]` to `[end_time]`:

<tool_call>{\"name\":\"get_video_clip_frame\",\"arguments\":[{\"start_time\":[start_time],\"end_time\":[end_time]}]}</tool_call>

Use the insights from the clip to inform your reasoning and construct the final answer."""

CROP_SUCCESS_PROMPT = """Tool execution successful. Analyze the visual information from the provided video clip to answer the user's question."""
CROP_FAIL_PROMPT = """Tool execution failed. Please continue your analysis based on your existing knowledge and the information from the conversation so far."""


# ==============================================================================
# Video Utility Functions
# ==============================================================================
def _get_video_info(video_path: str) -> Tuple[float, int, int, int, float]:
    """
    Get basic video information.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (fps, width, height, total_frames, total_duration).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        frame_shape = vr[0].shape
        height, width = frame_shape[:2]
        total_duration = total_frames / fps if fps > 0 else 0

        if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0 or total_duration <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}")

        return fps, width, height, total_frames, total_duration

    except Exception as e:
        raise RuntimeError(f"Error reading video file {video_path}: {e}")


def smart_nframes(total_frames: int, video_fps: int | float) -> int:
    """
    Calculate the number of frames for video used for model inputs.

    Args:
        total_frames: The original total number of frames of the video.
        video_fps: The original fps of the video.

    Returns:
        The number of frames for video used for model inputs.
    """
    def ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * FPS
    if nframes > total_frames:
        print(f"Warning: smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _crop_video(input_path: str, output_dir: str, start_time: float, end_time: float) -> str:
    """
    Crop a video segment with strict FPS consistency checks.

    Args:
        input_path: Path to the input video file.
        output_dir: Directory to save the cropped video.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Path to the cropped video file, or an error message string on failure.
    """
    try:
        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid timestamp: start={start_time}, end={end_time}")

        orig_fps, orig_width, orig_height, total_frames, orig_duration = _get_video_info(input_path)

        start_time = min(max(0, start_time), orig_duration)
        end_time = min(end_time, orig_duration)
        clip_duration = end_time - start_time

        custom_temp_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(custom_temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=custom_temp_dir)
        output_path = temp_file.name
        temp_file.close()

        max_frames = int(round(clip_duration * orig_fps))
        nframes = smart_nframes(max_frames, orig_fps)
        crop_video_fps = nframes / clip_duration
        frame_interval = max_frames // nframes

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, crop_video_fps, (orig_width, orig_height))

        cap = cv2.VideoCapture(input_path)
        pos_set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * orig_fps))

        if not pos_set_success:
            print("Warning: Seeking to start frame failed, reading frame-by-frame...")
            current_pos = 0
            target_pos = int(start_time * orig_fps)
            while current_pos < target_pos and cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    raise RuntimeError("Unable to reach the starting frame (original video too short).")
                current_pos += 1

        current_frame_in_clip = 0
        while current_frame_in_clip < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Reached end of video early. Expected {max_frames} frames, got {current_frame_in_clip}")
                break
            if current_frame_in_clip % frame_interval == 0:
                out.write(frame)
            current_frame_in_clip += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video processing completed. Output: {output_path}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not generated: {output_path}")

        file_size = os.path.getsize(output_path)
        if file_size < 1024:
            raise RuntimeError(f"Output file too small ({file_size} bytes), no valid frame data.")

        return output_path

    except Exception as e:
        return f"Video processing error: {str(e)}"


# ==============================================================================
# Agent Core
# ==============================================================================
def run_agent_with_sandbox(
    client: openai.Client,
    model_name: str,
    user_prompt: str,
    user_video_path: str,
    user_image_path: str = None,
    output_base_dir: str = "eval/agent_runs",
) -> Tuple[list, str]:
    """
    Run an agentic loop with video clipping capability.

    The agent can iteratively call the `get_video_clip_frame` tool to crop
    video segments and use the cropped clips to answer the question.

    Args:
        client: OpenAI client instance.
        model_name: Name of the model to use.
        user_prompt: The question provided by the user.
            If ``user_image_path`` is provided, the prompt must contain the
            placeholder ``<image 1>`` to indicate where the image is inserted.
        user_video_path: Path to the user's video.
        user_image_path: Optional path to a supplementary image (used by
            Video-MMMU Adaptation tasks).
        output_base_dir: Base directory for saving intermediate cropped videos.

    Returns:
        Tuple of (conv_history, final_answer) where ``conv_history`` is a list
        of all assistant response texts and ``final_answer`` is the extracted
        answer string.
    """
    run_timestamp = int(time.time())
    output_dir = os.path.join(output_base_dir, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Intermediate files saved in: {os.path.abspath(output_dir)}")

    conversation_history = []
    conversation_history.append({"role": "system", "content": PREFIX_PROMPT})

    # Build initial user message
    if user_image_path:
        user_prompt_1, user_prompt_2 = user_prompt.split("<image 1>")
        initial_content = [
            {"type": "video", "video": user_video_path},
            {"type": "text", "text": user_prompt_1},
            {"type": "image", "image": user_image_path},
            {"type": "text", "text": user_prompt_2},
        ]
    else:
        initial_content = [
            {"type": "video", "video": user_video_path},
            {"type": "text", "text": user_prompt},
        ]
    conversation_history.append({"role": "user", "content": initial_content})

    print("\n" + "=" * 20 + " Agent is running " + "=" * 20)
    print(f"🤔 Question: {user_prompt}")
    print(f"🎬 Video: {user_video_path}")

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i + 1}/{MAX_ITERATIONS} ---")
        print("🧠 Calling model for reasoning...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature=0.1,
                max_tokens=4096,
                stop=["</answer>", "<|im_end|>"],
            )
            generated_text = response.choices[0].message.content
            print(f"🤖 Model Response:\n{generated_text}")
        except Exception as e:
            print(f"❌ API call failed: {e}")
            break

        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}],
        })

        if "</answer>" in generated_text:
            print("\n✅ Found final answer, task completed.")
            break

        timestamp_match = re.search(r"<tool_call>(.*?)</tool_call>", generated_text, re.DOTALL)
        if timestamp_match:
            try:
                tool_call = json.loads(timestamp_match.group(1).strip())
                if tool_call["name"] == "get_video_clip_frame":
                    clip_timestamps = [
                        [float(ts["start_time"]), float(ts["end_time"])]
                        for ts in tool_call["arguments"]
                    ]
                else:
                    raise NotImplementedError(f"Unsupported tool call: {tool_call}")
            except Exception as e:
                print(f"⚠️ Timestamp conversion error: {e}")
                break

            print(f"\n🐍 Found video clipping timestamps:\n---\n{clip_timestamps}\n---")

            video_save_dir = os.path.join(output_dir, f"iteration_{i + 1}_videos")
            os.makedirs(video_save_dir, exist_ok=True)

            clipped_videos = []
            error_info = []
            for start_time, end_time in clip_timestamps:
                processed_path = _crop_video(user_video_path, video_save_dir, start_time, end_time)
                if os.path.exists(processed_path):
                    clipped_videos.append(processed_path)
                else:
                    error_info.append(processed_path)

            feedback = []
            if error_info:
                feedback.extend([{"type": "text", "text": error} for error in error_info])
                feedback.append({"type": "text", "text": CROP_FAIL_PROMPT})
            else:
                print("✅ Video clipping executed successfully.")
                feedback.extend([{"type": "video", "video": path} for path in clipped_videos])
                feedback.append({"type": "text", "text": CROP_SUCCESS_PROMPT})

            conversation_history.append({"role": "user", "content": feedback})
        else:
            print("⚠️ Model did not provide answer or clipping timestamps. Terminating loop.")
            break

    print("\n" + "=" * 20 + " Agent run ended " + "=" * 20)
    final_response_text = conversation_history[-1]["content"][0]["text"]
    answer_match = re.search(r"<answer>(.*?)</answer>", final_response_text, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else final_response_text

    conv_history = [
        msg["content"][0]["text"]
        for msg in conversation_history
        if msg["role"] == "assistant"
    ]

    return conv_history, final_answer


# ==============================================================================
# VLLM Client Helper
# ==============================================================================
def connect_to_vllm(base_url: str = "http://0.0.0.0:8000/v1") -> Tuple[openai.Client, str]:
    """
    Connect to a VLLM server and return the client and model name.

    Args:
        base_url: Base URL of the VLLM server.

    Returns:
        Tuple of (client, model_name).

    Raises:
        SystemExit: If the connection fails.
    """
    import sys

    print(f"🚀 Connecting to VLLM server: {base_url}")
    try:
        client = openai.Client(api_key="EMPTY", base_url=base_url)
        models = client.models.list()
        if not models.data:
            raise ValueError("Server did not return any models.")
        model_name = models.data[0].id
        print(f"✅ Connection successful! Using model: {model_name}")
        return client, model_name
    except Exception as e:
        print("\n❌ Unable to connect to VLLM server.")
        print("Please confirm:")
        print("   1. VLLM server is running at the specified address and port.")
        print("   2. Network connection is stable.")
        print(f"   Error details: {e}\n")
        sys.exit(1)
