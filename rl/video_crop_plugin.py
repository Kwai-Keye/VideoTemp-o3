import re
import cv2
import os
import math
import json
import tempfile
import numpy as np
from datetime import datetime
from decord import VideoReader
from typing import Tuple, Optional, Dict, List, Union
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin import ORM, orms
from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.utils import get_logger
logger = get_logger()


PREFIX_PROMPT= """Think step-by-step before providing your final answer.

Enclose your entire reasoning process within <think> and </think> tags. Enclose your final answer within <answer> and </answer> tags.

If analyzing a specific video segment is necessary to answer the question, you may use the following tool to extract a clip from `[start_time]` to `[end_time]`:

<tool_call>{\"name\":\"get_video_clip_frame\",\"arguments\":[{\"start_time\":[start_time],\"end_time\":[end_time]}]}</tool_call>

Use the insights from the clip to inform your reasoning and construct the final answer.

The question is:
"""

CROP_SUCCESS_PROMPT = """Tool execution successful. Analyze the visual information from the provided video clip to answer the user's question."""
CROP_FAIL_PROMPT = """Tool execution failed. Please continue your analysis based on your existing knowledge and the information from the conversation so far."""



class VideoProcessingScheduler(MultiTurnScheduler):
    """
    Scheduler for multi-turn video processing
    The first turn processes the entire video, and subsequent turns clip the video based on timestamps output by the model
    """
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        # Core parameters for video processing
        self.current_video_path = None  # Track the current video being processed
        # self.max_frames = 64
        self.FPS_MIN_FRAMES = 4
        self.FPS_MAX_FRAMES = 64    # Maximum 64 frames for clipping
        self.FRAME_FACTOR = 2
        self.FPS = 2

        # Timestamp parsing pattern (referencing ReAct format)
        self.clip_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


    def _get_video_info(self, video_path: str) -> Tuple[float, int, int, int, float]:
        """
        Retrieve basic video information (internal utility method)

        Uses Decord to read video metadata.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Tuple[float, int, int, int, float]: A tuple containing the video's 
            frame rate (fps), width, height, total number of frames, and total duration.
        """
        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Use Decord VideoReader to get video information
        try:
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()  # Average frame rate
            total_frames = len(vr)  # Total number of frames
            frame_shape = vr[0].shape  # Get the shape of the first frame
            height, width = frame_shape[:2]  # Get height and width
            total_duration = total_frames / fps if fps > 0 else 0

            # Validate video information
            if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0 or total_duration <= 0:
                raise ValueError(f"Invalid video metadata for {video_path}")

            return fps, width, height, total_frames, total_duration

        except Exception as e:
            raise RuntimeError(f"Error reading video file {video_path}: {e}")


    def smart_nframes(
        self,
        total_frames: int,
        video_fps: int | float,
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        fps = self.FPS
        min_frames = ceil_by_factor(self.FPS_MIN_FRAMES, self.FRAME_FACTOR)
        max_frames = floor_by_factor(min(self.FPS_MAX_FRAMES, total_frames), self.FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, self.FRAME_FACTOR)
        if not (self.FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(f"nframes should in interval [{self.FRAME_FACTOR}, {total_frames}], but got {nframes}.")
        return nframes


    def _crop_video(
            self,
            input_path: str,
            start_time: float,
            end_time: float
        ) -> str:
        """Core video cropping tool with strict FPS consistency checks"""
        try:
            # Validate timestamps
            if start_time < 0 or end_time <= start_time:
                raise ValueError(f"Invalid timestamp: start={start_time}, end={end_time}")

            # Read original video information
            orig_fps, orig_width, orig_height, total_frames, orig_duration = self._get_video_info(input_path)

            # Handle boundaries and calculate the duration of the cropped segment
            start_time = min(max(0, start_time), orig_duration)
            end_time = min(end_time, orig_duration)
            clip_duration = end_time - start_time

            # Create temporary output file
            CUSTOM_TEMP_DIR = "rl/temp_videos/"+datetime.now().strftime("%Y%m%d_%H%M%S")  # Project temporary folder
            os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=CUSTOM_TEMP_DIR)
            output_path = temp_file.name
            temp_file.close()

            max_frames = int(round(clip_duration * orig_fps))  # Safety upper limit (to avoid infinite loops)

            nframes = self.smart_nframes(max_frames, orig_fps)
            crop_video_fps = nframes / clip_duration
            frame_interval = max_frames // nframes

            # Configure encoder
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                # fps=orig_fps,
                fps=crop_video_fps,
                frameSize=(orig_width, orig_height)
            )

            # Seek to start frame
            cap = cv2.VideoCapture(input_path)
            pos_set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * orig_fps))
            if not pos_set_success:
                print(f"Warning: Seeking to start frame failed, reading frame-by-frame...")  
                current_pos = 0
                target_pos = int(start_time * orig_fps)
                while current_pos < target_pos and cap.isOpened():
                    ret, _ = cap.read()
                    if not ret:
                        raise RuntimeError(f"Unable to reach the starting frame (the original video is too short).")
                    current_pos += 1

            # Read and write all frames within the segment (no sampling)
            current_frame_in_clip = 0  # Current frame index within the segment

            while current_frame_in_clip < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Reached end of video early. Expected {max_frames} frames, got {current_frame_in_clip}")  
                    break
                
                if current_frame_in_clip % frame_interval == 0:
                    out.write(frame)
                current_frame_in_clip += 1

            # Release resources in the correct order
            cap.release()
            out.release()  # Flush all frames and write metadata
            cv2.destroyAllWindows()  # Ensure no OpenCV resources are locked

            print(f"Video processing completed. Output: {output_path}")
            # Validate output file
            if not os.path.exists(output_path):
                raise RuntimeError(f"The output file has not been generated: {output_path}")
            file_size = os.path.getsize(output_path)
            if file_size < 1024:
                raise RuntimeError(f"The output file is too small ({file_size} bytes), and there is no valid frame data.")
            return output_path
            
        except Exception as e:
            return f"[Error] Video processing error: {str(e)}" 


    def _extract_timestamp(self, text: str) -> Optional[Tuple[float, float]]:
        """Extract timestamps from model output"""
        clip_match = self.clip_pattern.search(text)
        if not clip_match:
            return None
        
        clip_content = clip_match.group(1).strip()
        try:
            tool_call = json.loads(clip_content)
            if tool_call['name'] == "get_video_clip_frame":
                clip_timestamps = []
                for timestamp in tool_call['arguments']:
                    start_time = float(timestamp['start_time'])
                    end_time = float(timestamp['end_time'])
                    clip_timestamps.append([start_time, end_time])
                return clip_timestamps
            else:
                return None
        except:
            return None


    def check_finished(
        self,
        infer_request: RolloutInferRequest,
        response_choice: 'ChatCompletionResponseChoice',
        current_turn: int
    ) -> bool:
        """Check if the multi-turn video processing is finished"""
        completion = response_choice.message.content
        print(completion)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if answer_match:
            return True
        
        return super().check_finished(infer_request, response_choice, current_turn)


    def step(
        self,
        infer_request: RolloutInferRequest,
        response_choice: 'ChatCompletionResponseChoice',
        current_turn: int
    ) -> Dict:
        """Process each round of video cropping logic"""

        try:
            completion = response_choice.message.content
            original_token_ids = response_choice.token_ids or []
            original_loss_mask = [1] * len(original_token_ids)  # Original model output mask is 1 (for loss calculation)
            
            if len(infer_request.videos) > 0:
                # Original video path
                self.current_video_path = infer_request.videos[0]

            # Extract timestamps and process video cropping
            timestamps = self._extract_timestamp(completion)
            if timestamps is not None:
                timestamps = timestamps[:3] # Maximum of 3 clips per round
            processed_video_paths = []  # Record successfully cropped video paths
            error_info = []

            if timestamps and self.current_video_path:
                print(f"Clip {timestamps} from {self.current_video_path}")
                for start_time, end_time in timestamps:
                    # Execute video cropping with a maximum of 32 frames
                    crop_output = self._crop_video(
                        input_path=self.current_video_path,
                        start_time=start_time,
                        end_time=end_time
                    )
                    # Check if the processed output is a valid video path
                    if os.path.exists(crop_output):
                        processed_video_paths.append(crop_output)
                    else:
                        error_info.append(crop_output)

                # Construct <video> tokens and dialogue content based on the number of cropped videos
                if len(error_info) == 0:
                    # Generate corresponding number of <video> tags (1 video corresponds to 1 <video>)
                    clip_count = len(processed_video_paths)
                    next_content = "<video>\n" * clip_count + CROP_SUCCESS_PROMPT
                else:
                    # Some cropping failed
                    next_content = str(error_info) + CROP_FAIL_PROMPT
            else:
                next_content = "[Error] No valid timestamp found in the model output." + CROP_FAIL_PROMPT

            # Add current round cropping information
            infer_request.messages.append({
                    'role': 'user',
                    'content': next_content,
                })
            if timestamps and len(error_info) == 0:
                infer_request.videos.extend(processed_video_paths)

            # Process token encoding: merge new dialogue content tokens with original tokens
            tokenizer = self.infer_engine.default_template.tokenizer if hasattr(self, 'infer_engine') else None
            if not tokenizer:
                # Fault tolerance: return original tokens when tokenizer is not found to avoid training exceptions
                return {
                    'infer_request': infer_request,
                    'response_token_ids': original_token_ids,
                    'response_loss_mask': original_loss_mask,
                    'rollout_infos': {'error': 'Tokenizer not found', 'generated_clips_count': clip_count}
                }

        except Exception as e:
            print(f"[Error] An exception occurred during video processing: {e}")
            # Fault tolerance: return original tokens when an exception occurs to avoid training interruption

        return {
            'infer_request': infer_request,
            'rollout_infos': {
                'videos': infer_request.videos,
            }
        }


multi_turns['video_processing_scheduler'] = VideoProcessingScheduler


class BaseVideoReward(ORM):
    """Base class for video reward functions to avoid code duplication"""
    
    def __init__(self):
        """Initialize the base reward function"""
        # Timestamp extraction pattern
        self.clip_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    def _parse_timestamps(self, text: str) -> List[Tuple[float, float]]:
        """Extract a list of timestamps from the text"""
        timestamps = []
        clip_matches = self.clip_pattern.findall(text)
        for clip_content in clip_matches:
            try:
                tool_call = json.loads(clip_content.strip())
                if tool_call['name'] == "get_video_clip_frame":
                    for timestamp in tool_call['arguments']:
                        start_time = float(timestamp['start_time'])
                        end_time = float(timestamp['end_time'])
                        timestamps.append((start_time, end_time))
            except:
                continue
        return timestamps

    def _calculate_iou(self, pred: Tuple[float, float], true: Tuple[float, float]) -> float:
        """Calculate the IoU (Intersection over Union) of two timestamp intervals"""
        pred_start, pred_end = pred
        true_start, true_end = true

        # Calculate intersection
        overlap_start = max(pred_start, true_start)
        overlap_end = min(pred_end, true_end)
        overlap = max(0.0, overlap_end - overlap_start)

        # Calculate union
        union = (pred_end - pred_start) + (true_end - true_start) - overlap
        return overlap / union if union > 0 else 0.0

    def clip(self, value, min_value, max_value):
        """Clamp a single value within the range [min_value, max_value]"""
        return max(min_value, min(value, max_value))
    
    def _get_best_iou(self, pred_ts: List[Tuple[float, float]], true_ts: List[Tuple[float, float]]) -> float:
        """Calculate the best matching IoU between predicted and true timestamps"""
        if not pred_ts or not true_ts:
            return 0.0

        total_iou = 0.0
        matched = set()

        # Find the best matching true timestamp for each predicted timestamp
        for pred in pred_ts:
            max_iou = 0.0
            best_idx = -1
            for idx, true in enumerate(true_ts):
                if idx not in matched:
                    iou = self._calculate_iou(pred, true)
                    if iou > max_iou:
                        max_iou = iou
                        best_idx = idx
            if best_idx != -1:
                matched.add(best_idx)
                total_iou += max_iou

        average_iou = total_iou / len(pred_ts) if len(pred_ts) > 0 else 0.0
        iou_reward = self.clip((average_iou - 0.2) / (0.8 - 0.2), 0.0, 1.0)
        return iou_reward

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer content from tags"""
        answer_match = self.answer_pattern.search(text)
        clip_match = self.clip_pattern.search(text)
        if answer_match and not clip_match:
            return answer_match.group(1).strip()
        return None

    def _compute_answer_reward(self, model_answers: List[str], 
        reference_answers: List[str], 
        timestamps_list: List[Tuple[float, float]], 
        data_type_list: List[str]) -> List[float]:
        """Compute item-wise rule-based reward"""
        
        def first_letter(s: str) -> str:
            if not s:
                return ""
            s = s.strip()
            for ch in s:
                if ch.isalpha():
                    return ch
            return ""

        assert len(model_answers) == len(reference_answers)
        rewards = []

        for i in range(len(model_answers)):
            model_resp = model_answers[i]
            ref_resp = reference_answers[i]

            if data_type_list[i] == "qa":
                m = first_letter(model_resp)
                r = first_letter(ref_resp)
                rewards.append(1.0 if m and r and m == r else 0.0)
            elif data_type_list[i] == "grounding":
                matches = re.findall(r"-?\d+\.?\d*", model_resp)
                if len(matches) % 2 == 0:
                    pred_timestamp = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]
                    true_timestamp = timestamps_list[i]
                    rewards.append(self._get_best_iou(pred_timestamp, true_timestamp))
                else:
                    rewards.append(0.0)
            else:
                raise NotImplementedError(f"Unsupported data type: {data_type_list[i]}")

        return rewards

    def normalize_list(self, data_list, epsilon=1e-8):
        """Normalize the list to balance rewards across different tasks"""
        if len(data_list) == 0:
            return []

        mean = np.mean(data_list)
        std = np.std(data_list)
        normalized_data = [(x - mean) / (std + epsilon) for x in data_list]
        normalized_data = [float(1 / (1 + np.exp(-x))) for x in normalized_data]
        return normalized_data

    def _extract_trajectory_data(self, trajectory_ids: List[str], global_trajectories: Dict[str, List[Dict]]):
        """Extract common trajectory data"""
        model_answers = []
        reference_answers = []
        timestamps_list = []
        iou_scores = []
        data_type_list = []
        
        for idx, local_tra_id in enumerate(trajectory_ids):
            trajectory = global_trajectories.get(local_tra_id, [])[-1]
            messages = trajectory.get('messages', [])

            data_type = trajectory.get('data_type', '')
            data_type_list.append(data_type)
            
            true_timestamps = trajectory.get('timestamp', [])
            true_timestamps = [[float(num) for num in sublist] for sublist in true_timestamps]
            timestamps_list.append(true_timestamps)
            
            if data_type == "qa":
                if len(messages) <= 3:
                    iou_scores.append(0.0)
                else:
                    pred_timestamps = []
                    pred_timestamps.extend(self._parse_timestamps(messages[-3].get('content', '')))
                    iou = self._get_best_iou(pred_timestamps, true_timestamps)
                    iou_scores.append(iou)
            else:
                iou_scores.append(0.0)

            pred_answer = self._extract_answer(messages[-1].get('content', ''))
            model_answers.append(pred_answer if pred_answer is not None else "")
            solution = trajectory.get('solution', '')
            reference_answers.append(solution)
            
        return model_answers, reference_answers, timestamps_list, iou_scores, data_type_list


class Accuracy_Reward(BaseVideoReward):
    """Reward function that only computes answer accuracy"""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Compute accuracy reward values (range: 0-1)"""
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs', {})

        model_answers, reference_answers, timestamps_list, _, data_type_list = self._extract_trajectory_data(
            trajectory_ids, global_trajectories
        )

        # Compute answer accuracy
        accuracy_scores = self._compute_answer_reward(
            model_answers, reference_answers, timestamps_list, data_type_list
        )

        return accuracy_scores

# Register reward functions
orms['acc_reward'] = Accuracy_Reward


class IOU_Reward(BaseVideoReward):
    """Reward function that computes penalty-aware IoU reward"""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Compute penalty-aware IoU reward values (range: 0-1)"""
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs', {})

        model_answers, reference_answers, timestamps_list, iou_scores, data_type_list = self._extract_trajectory_data(
            trajectory_ids, global_trajectories
        )

        # Compute answer accuracy
        accuracy_scores = self._compute_answer_reward(
            model_answers, reference_answers, timestamps_list, data_type_list
        )

        # Compute penalty-aware IoU reward
        rewards = []
        for i in range(len(trajectory_ids)):
            if accuracy_scores[i] == 1.0:
                total_reward = iou_scores[i] if iou_scores[i] >= 0.1 else iou_scores[i] - 0.1
            else:
                total_reward = 0
            rewards.append(total_reward)

        return rewards

# Register reward functions
orms['iou_reward'] = IOU_Reward




class FormatReward(ORM):
    """Reward function for checking response format correctness"""

    def __init__(self):
        self.think_tool_pattern = re.compile(r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>$", re.DOTALL)
        self.think_answer_pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
        self.tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def _compute_format_reward(self, model_outputs: List[List[str]]) -> List[float]:
        """Compute format reward: 1.0 if correct, 0.0 if incorrect"""
        format_rewards = []

        for output in model_outputs:
            reward = 1.0
            for response in output:
                # Check if response matches accepted formats
                have_clip_tag = bool(self.think_tool_pattern.fullmatch(response))
                have_answer_tag = bool(self.think_answer_pattern.fullmatch(response))
                
                if not (have_clip_tag or have_answer_tag):
                    reward = 0.0
                    break

                # If clip request is included, validate clip information
                if have_clip_tag:
                    clip_match = self.tool_call_pattern.search(response)
                    try:
                        clip_content = clip_match.group(1).strip() if clip_match else ""
                        tool_call = json.loads(clip_content)
                        assert tool_call['name'] == "get_video_clip_frame"
                        for timestamp in tool_call['arguments']:
                            float(timestamp['start_time'])
                            float(timestamp['end_time'])
                    except Exception:
                        reward = 0.0
                        break
            
            format_rewards.append(reward)
        return format_rewards

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Compute format rewards (range: 0-1)"""
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs', {})
        
        model_outputs = []
        for local_tra_id in trajectory_ids:
            trajectory = global_trajectories.get(local_tra_id, [])[-1]
            messages = trajectory.get('messages', [])
            model_outputs.append([messages[i].get('content', '') for i in range(2, len(messages), 2)])

        return self._compute_format_reward(model_outputs)

# Register reward function
orms['format_reward'] = FormatReward