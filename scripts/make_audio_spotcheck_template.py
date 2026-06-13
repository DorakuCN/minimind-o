#!/usr/bin/env python3
"""Create a human listening template for disputed A2A cases."""

import argparse
import json
import os
from datetime import datetime


DEFAULT_CASE_IDS = [
    "a2a_en_food",
    "a2a_en_black_hole",
    "a2a_zh_health_probe",
    "a2a_en_sky_blue",
    "a2a_zh_coffee_probe",
    "a2a_en_animals",
    "a2a_zh_cat_story_probe",
    "a2a_en_ai_fields",
    "a2a_zh_ai_fields_probe",
    "a2a_zh_snow_probe",
]


def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def load_results(run_dir):
    return load_jsonl(os.path.join(run_dir, "results.jsonl"))


def load_asr(run_dir):
    path = os.path.join(run_dir, "asr_eval.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data.get("items", [])}


def short(text, limit=220):
    return (text or "").replace("\n", " ").replace("|", "\\|")[:limit]


def fmt_float(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def audio_path(run_dir, row):
    path = row.get("decoded_audio_path") or ""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return path


def main():
    parser = argparse.ArgumentParser(description="Create audio human spot-check markdown template")
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=DEFAULT_CASE_IDS,
        help="Audio case ids to include; defaults to known disputed A2A cases.",
    )
    args = parser.parse_args()

    a = load_results(args.run_a)
    b = load_results(args.run_b)
    asr_a = load_asr(args.run_a)
    asr_b = load_asr(args.run_b)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Audio Spot-Check Template: {args.label_a} vs {args.label_b}\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Run A: `{args.run_a}`\n")
        f.write(f"- Run B: `{args.run_b}`\n")
        f.write("- Purpose: human listening and semantic adjudication for disputed audio-input cases.\n")
        f.write("- Important: CER measures ASR transcript distance only; it does not measure semantic correctness, refusal behavior, naturalness, or factuality.\n\n")

        f.write("Scoring rubric:\n\n")
        f.write("- `intelligibility`: 0 unintelligible, 1 hard to understand, 2 mostly understandable, 3 clear.\n")
        f.write("- `semantic`: 0 wrong/off-prompt, 1 mostly wrong, 2 partly correct/useful, 3 correct and on-prompt.\n")
        f.write("- `naturalness`: 0 broken/silent/looping, 1 poor, 2 usable, 3 natural enough.\n")
        f.write("- `winner`: `A`, `B`, or `Tie`; judge from listening plus semantic correctness, not CER alone.\n\n")

        f.write("| id | source audio | A audio | B audio | A CER | B CER | A ASR | B ASR | A intelligibility | B intelligibility | A semantic | B semantic | A naturalness | B naturalness | winner | notes |\n")
        f.write("| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")

        for cid in args.case_ids:
            ra = a.get(cid, {})
            rb = b.get(cid, {})
            aa = asr_a.get(cid, {})
            ab = asr_b.get(cid, {})
            base = ra or rb
            if not base:
                continue
            f.write(
                f"| `{cid}` | {base.get('audio_path', '')} | {audio_path(args.run_a, ra)} | {audio_path(args.run_b, rb)} | "
                f"{fmt_float(aa.get('cer'))} | {fmt_float(ab.get('cer'))} | "
                f"{short(aa.get('asr_text') or ra.get('text', ''))} | {short(ab.get('asr_text') or rb.get('text', ''))} | "
                "|  |  |  |  |  |  |  |  |\n"
            )

    print(f"Wrote audio spot-check template: {args.output}")


if __name__ == "__main__":
    main()
