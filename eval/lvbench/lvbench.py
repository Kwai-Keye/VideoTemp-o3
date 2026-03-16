"""
LVBench benchmark evaluation script.

Data dir : eval/lvbench/data/
  - video_info_sample.meta.jsonl  (question metadata)
  - video_chunks/                 (video files, named by key)
Output   : eval/lvbench/output/test.json
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
INPUT_FILE = "eval/lvbench/data/video_info.meta.jsonl"
VIDEO_DIR = "eval/lvbench/data/video_chunks"
OUTPUT_PATH = "eval/lvbench/output/test.json"
AGENT_RUNS_DIR = "eval/lvbench/agent_runs"


# ==============================================================================
# Data helpers
# ==============================================================================
def prepare_input_list(input_file: str, video_dir: str) -> list:
    """Parse the metadata file and build the input list."""
    input_list = []
    for line in open(input_file, "r"):
        item = json.loads(line)
        video_path = os.path.join(video_dir, item["key"] + ".mp4")

        if not os.path.exists(video_path):
            print(f"❌ Video file does not exist: {video_path}")
            sys.exit(1)

        for qa in item["qa"]:
            question = (
                qa["question"]
                .replace("(A) ", "A.")
                .replace("(B) ", "B.")
                .replace("(C) ", "C.")
                .replace("(D) ", "D.")
            )
            input_list.append({
                "video_path": video_path,
                "uid": qa["uid"],
                "question": question,
            })

    return input_list


# ==============================================================================
# Worker
# ==============================================================================
def vllm_api_process_(item: dict, output_file: str) -> None:
    """Process a single QA item and append the result to the output file."""
    conv_history, final_answer = run_agent_with_sandbox(
        client=client,
        model_name=MODEL_NAME,
        user_prompt=item["question"],
        user_video_path=item["video_path"],
        output_base_dir=AGENT_RUNS_DIR,
    )

    print("#" * 50)
    print(f"Response: {final_answer}")
    print("#" * 50)

    with lock:
        with open(output_file, "a") as f:
            f.write(json.dumps({item["uid"]: final_answer}, ensure_ascii=False) + "\n")
            f.flush()


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    client, MODEL_NAME = connect_to_vllm(VLLM_BASE_URL)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    input_list = prepare_input_list(INPUT_FILE, VIDEO_DIR)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(vllm_api_process_, item, OUTPUT_PATH) for item in input_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Waiting for all subprocesses done...")
    executor.shutdown(wait=True)
    print("All subprocesses done.")

    # Merge per-line results into a single dict and overwrite the file
    total_data = {}
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            total_data.update(json.loads(line))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(total_data, f, indent=4)