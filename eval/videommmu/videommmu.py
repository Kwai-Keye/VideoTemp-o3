"""
Video-MMMU benchmark evaluation script.

Data dir : eval/videommmu/data/
  - {Adaptation,Comprehension,Perception}/test-00000-of-00001.parquet
  - video/   (video files)
  - image/   (supplementary images for Adaptation task)
Output   : eval/videommmu/output/test.jsonl

Note: Only multiple-choice questions are evaluated.
      The Adaptation task additionally provides a reference image.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import connect_to_vllm, run_agent_with_sandbox


# ==============================================================================
# Global state (shared across threads)
# ==============================================================================
client = None
MODEL_NAME = None
lock = threading.Lock()

VLLM_BASE_URL = "http://0.0.0.0:8000/v1"
VIDEO_DIR = "eval/videommmu/data/video"
IMAGE_DIR = "eval/videommmu/data/image"
OUTPUT_PATH = "eval/videommmu/output/test.jsonl"
AGENT_RUNS_DIR = "eval/videommmu/agent_runs"

TASKS = ["Adaptation", "Comprehension", "Perception"]
MC_MAP = {i: chr(ord("A") + i) for i in range(14)}  # A–N


# ==============================================================================
# Data helpers
# ==============================================================================
def prepare_input_list(video_dir: str, image_dir: str) -> list:
    """Load all tasks and build the input list (MCQ only)."""
    input_list = []
    for task in TASKS:
        data = pd.read_parquet(f"eval/videommmu/data/{task}/test-00000-of-00001.parquet")
        for i in range(len(data)):
            if data.iloc[i]["question_type"] != "multiple-choice":
                continue

            video_name = data.iloc[i]["id"]
            video_path = os.path.join(video_dir, video_name + ".mp4")
            if not os.path.exists(video_path):
                print(f"⚠️ Video not found: {video_path}")

            question = data.iloc[i]["question"]
            for idx, option in enumerate(data.iloc[i]["options"]):
                question += f"\n{MC_MAP[idx]}. {option}"

            image_path = None
            if task == "Adaptation":
                image_path = os.path.join(image_dir, data.iloc[i]["image"]["path"])

            input_list.append({
                "question": question,
                "gt_answer": data.iloc[i]["answer"],
                "task": task,
                "video_path": video_path,
                "image_path": image_path,
            })

    return input_list


# ==============================================================================
# Worker
# ==============================================================================
def vllm_api_process_(item: dict, output_file: str) -> None:
    """Process a single item and append the result to the output file."""
    if not os.path.exists(item["video_path"]):
        print(f"⚠️ Video not found, skipping: {item['video_path']}")
        return

    conv_history, final_answer = run_agent_with_sandbox(
        client=client,
        model_name=MODEL_NAME,
        user_prompt=item["question"],
        user_video_path=item["video_path"],
        user_image_path=item.get("image_path"),
        output_base_dir=AGENT_RUNS_DIR,
    )

    print("#" * 50)
    print(f"Response: {final_answer}")
    print("#" * 50)

    item["response"] = final_answer
    item["conv_history"] = conv_history

    with lock:
        with open(output_file, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    client, MODEL_NAME = connect_to_vllm(VLLM_BASE_URL)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    input_list = prepare_input_list(VIDEO_DIR, IMAGE_DIR)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(vllm_api_process_, item, OUTPUT_PATH) for item in input_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Waiting for all subprocesses done...")
    executor.shutdown(wait=True)
    print("All subprocesses done.")