"""
MLVU benchmark evaluation script.

Data dir : eval/mlvu/data/
  - test-ground-truth/test_mcq_gt.json  (MCQ ground truth)
  - video/                               (video files)
Output   : eval/mlvu/output/test.jsonl
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
INPUT_FILE = "eval/mlvu/data/test-ground-truth/test_mcq_gt.json"
VIDEO_DIR = "eval/mlvu/data/video"
OUTPUT_PATH = "eval/mlvu/output/test.jsonl"
AGENT_RUNS_DIR = "eval/mlvu/agent_runs"

OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}


# ==============================================================================
# Worker
# ==============================================================================
def vllm_api_process_(item: dict, output_file: str) -> None:
    """Process a single MCQ item and append the result to the output file."""
    video_path = os.path.join(VIDEO_DIR, item["video"])
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found, skipping: {video_path}")
        return

    prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
    prompt += item["question"]

    gt_option = None
    for idx, option in enumerate(item["candidates"]):
        prompt += f"\n{OPTION_MAP[idx]}. {option}"
        if option == item["answer"]:
            gt_option = OPTION_MAP[idx]

    conv_history, final_answer = run_agent_with_sandbox(
        client=client,
        model_name=MODEL_NAME,
        user_prompt=prompt,
        user_video_path=video_path,
        output_base_dir=AGENT_RUNS_DIR,
    )

    print("#" * 50)
    print(f"Response: {final_answer}")
    print("#" * 50)

    item["response"] = final_answer
    item["gt_answer"] = gt_option
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

    with open(INPUT_FILE, "r") as f:
        input_list = json.load(f)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(vllm_api_process_, item, OUTPUT_PATH) for item in input_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Waiting for all subprocesses done...")
    executor.shutdown(wait=True)
    print("All subprocesses done.")