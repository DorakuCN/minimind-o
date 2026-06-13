#!/usr/bin/env python3
"""Generate a markdown blind-review template from batch_validate output dirs."""

import argparse
import json
import os
from datetime import datetime


def load_results(run_dir):
    path = os.path.join(run_dir, "results.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r["id"]: r for r in rows}


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short(text, limit=180):
    return (text or "").replace("\n", " ").replace("|", "\\|")[:limit]


def main():
    parser = argparse.ArgumentParser(description="Create a manual A/B review markdown template")
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = load_results(args.run_a)
    b = load_results(args.run_b)
    sa = load_summary(args.run_a)
    sb = load_summary(args.run_b)
    case_ids = sorted(set(a) | set(b))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Manual Review Template: {args.label_a} vs {args.label_b}\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Run A: `{args.run_a}`\n")
        f.write(f"- Run B: `{args.run_b}`\n")
        f.write(f"- A weight: `{sa.get('weight_path', '')}`\n")
        f.write(f"- B weight: `{sb.get('weight_path', '')}`\n\n")

        f.write("Scoring guidance:\n\n")
        f.write("- Winner: `A`, `B`, or `Tie`\n")
        f.write("- I2A grounding score: `0` mostly wrong, `1` main object only, `2` mostly correct, `3` correct and specific\n")
        f.write("- Audio score: `0` failed, `1` intelligible but poor, `2` usable, `3` good\n\n")

        f.write("| id | type | prompt/media | A text | B text | winner | grounding | audio | notes |\n")
        f.write("| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |\n")
        for cid in case_ids:
            ra = a.get(cid, {})
            rb = b.get(cid, {})
            base = ra or rb
            prompt = base.get("prompt") or base.get("audio_path") or base.get("image_path") or ""
            f.write(
                f"| `{cid}` | {base.get('type', '')} | {short(prompt, 80)} | "
                f"{short(ra.get('text', ''))} | {short(rb.get('text', ''))} |  |  |  |  |\n"
            )

    print(f"Wrote review template: {args.output}")


if __name__ == "__main__":
    main()
