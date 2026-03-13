"""
Unified scoring script for all benchmarks.

Usage:
    python eval/score.py <benchmark>

Supported benchmarks:
    videotemp   - VideoTemp MCQ evaluation (by duration part)
    videotemp-g - VideoTemp temporal grounding evaluation (mIoU / R@k)
    videomme    - Video-MME evaluation (by duration type: short/medium/long)
    lvbench     - LVBench evaluation (by question category)
    mlvu        - MLVU MCQ evaluation
    videommmu   - Video-MMMU evaluation (by task: Adaptation/Comprehension/Perception)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np


# ==============================================================================
# VideoTemp  (MCQ, split by duration)
# ==============================================================================
def eval_videotemp(input_file: str = "eval/videotemp/output/test.jsonl") -> None:
    """Evaluate VideoTemp MCQ results, broken down by video duration."""
    VALID_OPTIONS = {"A", "B", "C", "D", "E"}
    total_cnt: Dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0}
    correct_cnt: Dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0}

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            duration = item["duration"]
            if duration < 3 * 60:
                part = "1"
            elif duration < 10 * 60:
                part = "2"
            elif duration < 20 * 60:
                part = "3"
            else:
                part = "4"

            total_cnt[part] += 1

            response = item["response"]
            if not response or response[0] not in VALID_OPTIONS:
                response = "A"
            if item["answer"][0] == response[0]:
                correct_cnt[part] += 1

    print("=" * 45)
    print("VideoTemp MCQ Evaluation")
    print("=" * 45)
    part_labels = {"1": "<3min", "2": "3-10min", "3": "10-20min", "4": ">20min"}
    for part, label in part_labels.items():
        total = total_cnt[part]
        correct = correct_cnt[part]
        acc = correct / total if total > 0 else 0.0
        print(f"  Part{part} ({label:>9s}): {correct:4d}/{total:4d}  Acc={acc:.2%}")

    all_total = sum(total_cnt.values())
    all_correct = sum(correct_cnt.values())
    print("-" * 45)
    print(f"  Overall             : {all_correct:4d}/{all_total:4d}  Acc={all_correct/all_total:.2%}")


# ==============================================================================
# VideoTemp-Grounding  (temporal grounding, mIoU + R@{0.3,0.5,0.7})
# ==============================================================================
def compute_iou(pred: list, gt: list) -> float:
    """Compute the IoU between a predicted window and a ground-truth window."""
    pred_arr = np.array([pred])
    gt_arr = np.array([gt])
    inter_left = np.maximum(pred_arr[:, 0, None], gt_arr[None, :, 0])
    inter_right = np.minimum(pred_arr[:, 1, None], gt_arr[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred_arr[:, 0, None], gt_arr[None, :, 0])
    union_right = np.maximum(pred_arr[:, 1, None], gt_arr[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = inter / union
    return float(np.max(overlap))


def eval_videotemp_grounding(input_file: str = "eval/videotemp/output/test-g.jsonl") -> None:
    """Evaluate VideoTemp temporal grounding results."""
    THRESHOLDS = [0.3, 0.5, 0.7]
    parts = ["1", "2", "3", "4", "total"]
    target_ts: Dict[str, list] = {p: [] for p in parts}
    predict_ts: Dict[str, list] = {p: [] for p in parts}
    error_count = 0

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            duration = item["duration"]
            if duration < 3 * 60:
                part = "1"
            elif duration < 10 * 60:
                part = "2"
            elif duration < 20 * 60:
                part = "3"
            else:
                part = "4"

            target_ts[part].append(item["timestamp"])
            target_ts["total"].append(item["timestamp"])

            matches = re.findall(r"-?\d+\.?\d*", item["response"])
            if len(matches) >= 2:
                seg = [float(matches[0]), float(matches[1])]
                predict_ts[part].append(seg)
                predict_ts["total"].append(seg)
            else:
                predict_ts[part].append(None)
                predict_ts["total"].append(None)
                error_count += 1

    print("=" * 55)
    print("VideoTemp Grounding Evaluation")
    print("=" * 55)
    part_labels = {"1": "<3min", "2": "3-10min", "3": "10-20min", "4": ">20min", "total": "Total"}
    for part in parts:
        preds = predict_ts[part]
        gts = target_ts[part]
        ious = [
            compute_iou(preds[i], gts[i]) if preds[i] is not None else 0.0
            for i in range(len(preds))
        ]
        miou = float(np.mean(ious)) * 100
        r_at = {thr: sum(1 for s in ious if s > thr) / len(ious) * 100 for thr in THRESHOLDS}
        avg = (miou + sum(r_at.values())) / (1 + len(THRESHOLDS))
        label = part_labels[part]
        print(f"  [{label:>9s}]  mIoU={miou:.2f}%  "
              f"R@0.3={r_at[0.3]:.2f}%  R@0.5={r_at[0.5]:.2f}%  R@0.7={r_at[0.7]:.2f}%  "
              f"Avg={avg:.2f}%")

    if error_count:
        print(f"\n  ⚠ Parse errors (no valid timestamp found): {error_count}")


# ==============================================================================
# Video-MME
# ==============================================================================
VIDEOMME_CATEGORIES = [
    "Knowledge", "Film & Television", "Sports Competition",
    "Artistic Performance", "Life Record", "Multilingual",
]
VIDEOMME_SUB_CATEGORIES = [
    "Humanity & History", "Literature & Art", "Biology & Medicine",
    "Finance & Commerce", "Astronomy", "Geography", "Law", "Life Tip",
    "Technology", "Animation", "Movie & TV Show", "Documentary",
    "News Report", "Esports", "Basketball", "Football", "Athletics",
    "Other Sports", "Stage Play", "Magic Show", "Variety Show",
    "Acrobatics", "Handicraft", "Food", "Fashion", "Daily Life",
    "Travel", "Pet & Animal", "Exercise", "Multilingual",
]
VIDEOMME_TASK_CATEGORIES = [
    "Temporal Perception", "Spatial Perception", "Attribute Perception",
    "Action Recognition", "Object Recognition", "OCR Problems",
    "Counting Problem", "Temporal Reasoning", "Spatial Reasoning",
    "Action Reasoning", "Object Reasoning", "Information Synopsis",
]


def _extract_videomme_answer(s: str) -> str:
    """Extract a single option letter (A-D) from a free-form response."""
    s = s.strip()
    for prefix in [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:", "Answer:", "Option:",
        "The correct answer", "The correct option",
    ]:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    if s and s[0] not in "ABCD":
        return ""
    m = re.search(r"[ABCD]", s)
    return m[0] if m else ""


def eval_videomme(
    results_file: str = "eval/videomme/data/videomme/output/test.json",
    video_types: Optional[Union[List[str], str]] = None,
    return_categories_accuracy: bool = True,
    return_sub_categories_accuracy: bool = False,
    return_task_types_accuracy: bool = False,
) -> None:
    """Evaluate Video-MME results.

    Args:
        results_file: Path to the formatted output JSON produced by videomme.py.
        video_types: Duration bucket(s) to evaluate, e.g. ``"short,medium,long"``.
            If *None*, all unique duration values in the file are used.
        return_categories_accuracy: Print per-domain accuracy.
        return_sub_categories_accuracy: Print per-sub-category accuracy.
        return_task_types_accuracy: Print per-task-type accuracy.
    """
    with open(results_file, "r") as f:
        all_results = json.load(f)

    if video_types is None:
        video_types = sorted({item["duration"] for item in all_results})
    elif isinstance(video_types, str):
        video_types = [v.strip() for v in video_types.split(",")]

    q_type_dict: Dict[str, Dict] = {}
    v_type_dict: Dict[str, Dict] = {}
    v_sub_type_dict: Dict[str, Dict] = {}

    for vt in video_types:
        q_type_dict[vt] = {q: {"correct": 0, "answered": 0} for q in VIDEOMME_TASK_CATEGORIES}
        v_type_dict[vt] = {v: {"correct": 0, "answered": 0} for v in VIDEOMME_CATEGORIES}
        v_sub_type_dict[vt] = {s: {"correct": 0, "answered": 0} for s in VIDEOMME_SUB_CATEGORIES}

    for item in all_results:
        vt = item["duration"]
        if vt not in video_types:
            continue
        domain = item["domain"]
        sub_cat = item["sub_category"]
        for q in item["questions"]:
            extracted = _extract_videomme_answer(q["response"])
            if not extracted:
                continue
            correct = extracted == q["answer"]
            q_type = q["task_type"]
            q_type_dict[vt][q_type]["answered"] += 1
            q_type_dict[vt][q_type]["correct"] += correct
            v_type_dict[vt][domain]["answered"] += 1
            v_type_dict[vt][domain]["correct"] += correct
            v_sub_type_dict[vt][sub_cat]["answered"] += 1
            v_sub_type_dict[vt][sub_cat]["correct"] += correct

    def _pct(correct: int, answered: int) -> str:
        return f"{100 * correct / answered:.1f}%" if answered > 0 else "N/A"

    for vt in video_types:
        print("=" * 45)
        print(f"Video-MME  |  Duration: {vt}")
        print("=" * 45)
        if return_categories_accuracy:
            print("  [Domains]")
            for v in VIDEOMME_CATEGORIES:
                d = v_type_dict[vt][v]
                print(f"    {v:<30s}: {_pct(d['correct'], d['answered'])}")
        if return_sub_categories_accuracy:
            print("  [Sub-Categories]")
            for s in VIDEOMME_SUB_CATEGORIES:
                d = v_sub_type_dict[vt][s]
                print(f"    {s:<30s}: {_pct(d['correct'], d['answered'])}")
        if return_task_types_accuracy:
            print("  [Task Types]")
            for q in VIDEOMME_TASK_CATEGORIES:
                d = q_type_dict[vt][q]
                print(f"    {q:<30s}: {_pct(d['correct'], d['answered'])}")
        total_correct = sum(q_type_dict[vt][q]["correct"] for q in VIDEOMME_TASK_CATEGORIES)
        total_answered = sum(q_type_dict[vt][q]["answered"] for q in VIDEOMME_TASK_CATEGORIES)
        print(f"  Overall: {_pct(total_correct, total_answered)}\n")

    print("=" * 45)
    print("Video-MME  |  Overall (all duration types)")
    print("=" * 45)
    if return_categories_accuracy:
        print("  [Domains]")
        for v in VIDEOMME_CATEGORIES:
            total_c = sum(v_type_dict[vt][v]["correct"] for vt in video_types)
            total_a = sum(v_type_dict[vt][v]["answered"] for vt in video_types)
            print(f"    {v:<30s}: {_pct(total_c, total_a)}")
    if return_sub_categories_accuracy:
        print("  [Sub-Categories]")
        for s in VIDEOMME_SUB_CATEGORIES:
            total_c = sum(v_sub_type_dict[vt][s]["correct"] for vt in video_types)
            total_a = sum(v_sub_type_dict[vt][s]["answered"] for vt in video_types)
            print(f"    {s:<30s}: {_pct(total_c, total_a)}")
    if return_task_types_accuracy:
        print("  [Task Types]")
        for q in VIDEOMME_TASK_CATEGORIES:
            total_c = sum(q_type_dict[vt][q]["correct"] for vt in video_types)
            total_a = sum(q_type_dict[vt][q]["answered"] for vt in video_types)
            print(f"    {q:<30s}: {_pct(total_c, total_a)}")
    grand_correct = sum(
        sum(q_type_dict[vt][q]["correct"] for q in VIDEOMME_TASK_CATEGORIES) for vt in video_types
    )
    grand_answered = sum(
        sum(q_type_dict[vt][q]["answered"] for q in VIDEOMME_TASK_CATEGORIES) for vt in video_types
    )
    print(f"  Overall: {_pct(grand_correct, grand_answered)}")


# ==============================================================================
# LVBench
# ==============================================================================
def eval_lvbench(
    response_path: str = "eval/lvbench/output/test.json",
    meta_data_path: str = "eval/lvbench/data/video_info.meta.jsonl",
) -> None:
    """Evaluate LVBench results by question category."""
    import jsonlines

    with open(response_path) as f:
        model_answers = json.load(f)

    total_qa = 0
    right_num = 0
    category_right: Dict[str, int] = defaultdict(int)
    category_total: Dict[str, int] = defaultdict(int)

    with jsonlines.open(meta_data_path) as reader:
        for meta in reader:
            for qa in meta["qa"]:
                uid = str(qa["uid"])
                if uid not in model_answers:
                    continue
                answer = model_answers[uid]
                if not answer or answer[0] not in "ABCDE":
                    answer = "A"
                correct = answer[0] == qa["answer"]
                for cat in qa["question_type"]:
                    category_total[cat] += 1
                    if correct:
                        category_right[cat] += 1
                if correct:
                    right_num += 1
                total_qa += 1

    print("=" * 45)
    print("LVBench Evaluation")
    print("=" * 45)
    for cat in sorted(category_total.keys()):
        total = category_total[cat]
        correct = category_right[cat]
        print(f"  {cat:<35s}: {correct:4d}/{total:4d}  Acc={correct/total:.2%}")
    print("-" * 45)
    print(f"  {'Overall':<35s}: {right_num:4d}/{total_qa:4d}  Acc={right_num/total_qa:.2%}")


# ==============================================================================
# MLVU
# ==============================================================================
def eval_mlvu(input_file: str = "eval/mlvu/output/test.jsonl") -> None:
    """Evaluate MLVU MCQ results."""
    total = 0
    correct = 0

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            response = item.get("response", "")
            if not response:
                continue
            if item["gt_answer"][0] == response[0]:
                correct += 1

    print("=" * 45)
    print("MLVU Evaluation")
    print("=" * 45)
    print(f"  Total  : {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total:.2%}" if total > 0 else "  No data.")


# ==============================================================================
# Video-MMMU
# ==============================================================================
def eval_videommmu(input_file: str = "eval/videommmu/output/test.jsonl") -> None:
    """Evaluate Video-MMMU results by task type."""
    VALID_OPTIONS = set("ABCDEFGHIJKLMN")
    counts: Dict[str, Dict[str, int]] = {
        task: {"total": 0, "correct": 0}
        for task in ["Adaptation", "Comprehension", "Perception"]
    }

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            task = item["task"]
            response = item.get("response", "")
            if not response or response[0] not in VALID_OPTIONS:
                response = "A"
            counts[task]["total"] += 1
            if item["gt_answer"][0] == response[0]:
                counts[task]["correct"] += 1

    print("=" * 45)
    print("Video-MMMU Evaluation")
    print("=" * 45)
    accs = []
    for task, d in counts.items():
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        accs.append(acc)
        print(f"  {task:<15s}: {d['correct']:4d}/{d['total']:4d}  Acc={acc:.2%}")
    print("-" * 45)
    print(f"  {'Average':<15s}:              Acc={sum(accs)/len(accs):.2%}")


# ==============================================================================
# CLI
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified benchmark scoring script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="benchmark", required=True)

    # videotemp
    p = sub.add_parser("videotemp", help="VideoTemp MCQ scoring")
    p.add_argument("--input_file", default="eval/videotemp/output/test.jsonl",
                   help="Path to the prediction JSONL file (default: %(default)s)")

    # videotemp-g
    p = sub.add_parser("videotemp-g", help="VideoTemp temporal grounding scoring")
    p.add_argument("--input_file", default="eval/videotemp/output/test-g.jsonl",
                   help="Path to the prediction JSONL file (default: %(default)s)")

    # videomme
    p = sub.add_parser("videomme", help="Video-MME scoring")
    p.add_argument("--input_file", default="eval/videomme/data/videomme/output/test.json",
                   help="Path to the formatted output JSON produced by videomme.py (default: %(default)s)")
    p.add_argument("--video_duration_type", default=None,
                   help="Comma-separated duration types to evaluate, e.g. 'short,medium,long'. "
                        "Defaults to all types found in the file.")
    p.add_argument("--return_categories_accuracy", action="store_true",
                   help="Print per-domain accuracy.")
    p.add_argument("--return_sub_categories_accuracy", action="store_true",
                   help="Print per-sub-category accuracy.")
    p.add_argument("--return_task_types_accuracy", action="store_true",
                   help="Print per-task-type accuracy.")

    # lvbench
    p = sub.add_parser("lvbench", help="LVBench scoring")
    p.add_argument("--input_file", default="eval/lvbench/output/test.json",
                   help="Path to the model prediction JSON file (default: %(default)s)")
    p.add_argument("--meta_data_path", default="eval/lvbench/data/video_info.meta.jsonl",
                   help="Path to the dataset metadata JSONL file (default: %(default)s)")

    # mlvu
    p = sub.add_parser("mlvu", help="MLVU scoring")
    p.add_argument("--input_file", default="eval/mlvu/output/test.jsonl",
                   help="Path to the prediction JSONL file (default: %(default)s)")

    # videommmu
    p = sub.add_parser("videommmu", help="Video-MMMU scoring")
    p.add_argument("--input_file", default="eval/videommmu/output/test.jsonl",
                   help="Path to the prediction JSONL file (default: %(default)s)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.benchmark == "videotemp":
        eval_videotemp(args.input_file)
    elif args.benchmark == "videotemp-g":
        eval_videotemp_grounding(args.input_file)
    elif args.benchmark == "videomme":
        eval_videomme(
            results_file=args.input_file,
            video_types=args.video_duration_type,
            return_categories_accuracy=args.return_categories_accuracy,
            return_sub_categories_accuracy=args.return_sub_categories_accuracy,
            return_task_types_accuracy=args.return_task_types_accuracy,
        )
    elif args.benchmark == "lvbench":
        eval_lvbench(args.input_file, args.meta_data_path)
    elif args.benchmark == "mlvu":
        eval_mlvu(args.input_file)
    elif args.benchmark == "videommmu":
        eval_videommmu(args.input_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
