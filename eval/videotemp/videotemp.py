"""
VideoTemp benchmark evaluation script.

Data file: eval/videotemp/data/data.jsonl
Output   : eval/videotemp/output/test.jsonl

Each item in the JSONL must contain:
  - videos: list of video paths (first element is used)
  - question: the question text
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
INPUT_FILE = "eval/videotemp/data/data.jsonl"
OUTPUT_PATH = "eval/videotemp/output/test.jsonl"
AGENT_RUNS_DIR = "eval/videotemp/agent_runs"


# ==============================================================================
# Worker
# ==============================================================================
def vllm_api_process_(item: dict, output_file: str) -> None:
    """Process a single item and append the result to the output file."""
    video_path = item["videos"][0]
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found, skipping: {video_path}")
        return

    conv_history, final_answer = run_agent_with_sandbox(
        client=client,
        model_name=MODEL_NAME,
        user_prompt=item["question"],
        user_video_path=video_path,
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

    input_list = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            input_list.append(json.loads(line))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(vllm_api_process_, item, OUTPUT_PATH) for item in input_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Waiting for all subprocesses done...")
    executor.shutdown(wait=True)
    print("All subprocesses done.")