"""
Video-MME benchmark evaluation script.

Dataset  : lmms-lab/Video-MME
Data file: eval/videomme/data/test-00000-of-00001.parquet
Output   : eval/videomme/data/videomme/output/test.json
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
DATA_PATH = "eval/videomme/data/test-00000-of-00001.parquet"
OUTPUT_PATH = "eval/videomme/data/videomme/output/test.json"
AGENT_RUNS_DIR = "eval/videomme/agent_runs"


# ==============================================================================
# Data helpers
# ==============================================================================
def load_existing_results(output_path: str) -> set:
    """Load processed (video_id + question_id) pairs to avoid reprocessing."""
    existing_set = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                item = json.loads(line)
                existing_set.add(item["video_id"] + item["question_id"])
    return existing_set


def prepare_input_list(data: pd.DataFrame, existing_set: set) -> list:
    """Build the list of items that still need to be processed."""
    input_list = []
    for i in range(len(data)):
        video_info = {
            "video_id": data["video_id"][i],
            "video_path": f"lmms-lab/Video-MME/data/{data['videoID'][i]}.mp4",
            "duration": data["duration"][i],
            "domain": data["domain"][i],
            "sub_category": data["sub_category"][i],
            "question_id": data["question_id"][i],
            "task_type": data["task_type"][i],
            "question": data["question"][i],
            "options": data["options"][i].tolist(),
            "answer": data["answer"][i],
            "response": "A",
        }

        if video_info["video_id"] + video_info["question_id"] in existing_set:
            continue

        if not os.path.exists(video_info["video_path"]):
            print(f"❌ Error: Video file '{video_info['video_path']}' does not exist.")
            sys.exit(1)

        input_list.append(video_info)

    return input_list


def format_output_data(output_path: str) -> None:
    """Group output records by video and write the final JSON file."""
    video_info: dict = {}
    with open(output_path, "r") as f:
        for line in f:
            item = json.loads(line)
            video_info.setdefault(item["video_id"], []).append(item)

    formatted_data = []
    for values in video_info.values():
        data = values[0]
        formatted_data.append({
            "video_id": data["video_id"],
            "video_path": data["video_path"],
            "duration": data["duration"],
            "domain": data["domain"],
            "sub_category": data["sub_category"],
            "questions": [
                {
                    "question_id": v["question_id"],
                    "task_type": v["task_type"],
                    "question": v["question"],
                    "options": v["options"],
                    "answer": v["answer"],
                    "response": v["response"],
                    "conv_history": v["conv_history"],
                }
                for v in values
            ],
        })

    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)


# ==============================================================================
# Worker
# ==============================================================================
def vllm_api_process_(item: dict, output_file: str) -> None:
    """Process a single video item and append the result to the output file."""
    prompt = item["question"]
    for option in item["options"]:
        prompt += f"\n{option}"

    conv_history, final_answer = run_agent_with_sandbox(
        client=client,
        model_name=MODEL_NAME,
        user_prompt=prompt,
        user_video_path=item["video_path"],
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

    data = pd.read_parquet(DATA_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    existing_set = load_existing_results(OUTPUT_PATH)
    input_list = prepare_input_list(data, existing_set)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(vllm_api_process_, item, OUTPUT_PATH) for item in input_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Waiting for all subprocesses done...")
    executor.shutdown(wait=True)
    print("All subprocesses done.")

    format_output_data(OUTPUT_PATH)