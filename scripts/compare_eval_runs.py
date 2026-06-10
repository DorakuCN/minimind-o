#!/usr/bin/env python3
"""Compare two batch_validate_omni.py summary.json outputs into a markdown report."""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def aggregate_by_type(results):
    by_type = defaultdict(lambda: {
        "total": 0,
        "pass": 0,
        "chars": [],
        "frames": [],
        "repeat": [],
        "special": [],
        "seconds": [],
    })
    for r in results:
        t = r["type"]
        by_type[t]["total"] += 1
        by_type[t]["pass"] += int(r["pass_basic"])
        by_type[t]["chars"].append(r["text_chars"])
        by_type[t]["frames"].append(r["valid_audio_frames"])
        by_type[t]["repeat"].append(r["repeat_score"])
        by_type[t]["special"].append(r["special_code_rate"])
        by_type[t]["seconds"].append(r["seconds"])
    return by_type


def avg(values):
    return sum(values) / max(len(values), 1)


def fmt_delta(a, b, higher_better=False):
    if a is None or b is None:
        return "n/a"
    delta = b - a
    if abs(delta) < 1e-9:
        return "0"
    sign = "+" if delta > 0 else ""
    improved = (delta > 0) if higher_better else (delta < 0)
    marker = " (better)" if improved else " (worse)" if delta != 0 else ""
    return f"{sign}{delta:.4f}{marker}"


def write_report(args, summary_a, summary_b, results_a, results_b):
    by_type_a = aggregate_by_type(results_a)
    by_type_b = aggregate_by_type(results_b)
    case_map_a = {r["id"]: r for r in results_a}
    case_map_b = {r["id"]: r for r in results_b}
    case_ids = sorted(set(case_map_a) | set(case_map_b))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Eval Comparison: {args.label_a} vs {args.label_b}\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Run A: `{summary_a.get('weight_path', args.run_a)}`\n")
        f.write(f"- Run B: `{summary_b.get('weight_path', args.run_b)}`\n")
        f.write(f"- Test set: `{summary_a.get('test_set', summary_b.get('test_set', 'unknown'))}`\n\n")

        f.write("## Overall\n\n")
        f.write("| Metric | A | B | Delta (B-A) |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        f.write(f"| Basic pass | {summary_a['passed']}/{summary_a['total']} | {summary_b['passed']}/{summary_b['total']} | "
                f"{summary_b['passed'] - summary_a['passed']:+d} |\n")
        f.write(f"| Pass rate | {summary_a['pass_rate']:.2%} | {summary_b['pass_rate']:.2%} | "
                f"{fmt_delta(summary_a['pass_rate'], summary_b['pass_rate'], higher_better=True)} |\n")
        f.write(f"| Dtype | {summary_a.get('dtype')} | {summary_b.get('dtype')} | |\n")
        f.write(f"| max_new_tokens | {summary_a.get('max_new_tokens')} | {summary_b.get('max_new_tokens')} | |\n\n")

        f.write("## By Type\n\n")
        f.write("| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | "
                "A avg repeat | B avg repeat | A avg sec | B avg sec |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        all_types = sorted(set(by_type_a) | set(by_type_b))
        for t in all_types:
            ta, tb = by_type_a[t], by_type_b[t]
            f.write(
                f"| {t} | {ta['pass']}/{ta['total']} | {tb['pass']}/{tb['total']} | "
                f"{avg(ta['chars']):.1f} | {avg(tb['chars']):.1f} | "
                f"{avg(ta['frames']):.1f} | {avg(tb['frames']):.1f} | "
                f"{avg(ta['repeat']):.4f} | {avg(tb['repeat']):.4f} | "
                f"{avg(ta['seconds']):.3f} | {avg(tb['seconds']):.3f} |\n"
            )

        f.write("\n## Per-Case\n\n")
        f.write("| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for cid in case_ids:
            ra = case_map_a.get(cid)
            rb = case_map_b.get(cid)
            note = ""
            if ra and rb:
                if ra["pass_basic"] and not rb["pass_basic"]:
                    note = "regression"
                elif rb["pass_basic"] and not ra["pass_basic"]:
                    note = "improved"
            f.write(
                f"| {cid} | "
                f"{ra['pass_basic'] if ra else 'n/a'} | {rb['pass_basic'] if rb else 'n/a'} | "
                f"{ra['text_chars'] if ra else 'n/a'} | {rb['text_chars'] if rb else 'n/a'} | "
                f"{ra['repeat_score'] if ra else 'n/a'} | {rb['repeat_score'] if rb else 'n/a'} | "
                f"{note} |\n"
            )

    print(f"Comparison report written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Compare two batch_validate_omni output directories")
    parser.add_argument("--run_a", required=True, help="Directory with summary.json from run A")
    parser.add_argument("--run_b", required=True, help="Directory with summary.json from run B")
    parser.add_argument("--label_a", default="baseline")
    parser.add_argument("--label_b", default="candidate")
    parser.add_argument(
        "--output",
        default="docs/evaluation_results/compare_eval_report.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    summary_a = load_summary(os.path.join(args.run_a, "summary.json"))
    summary_b = load_summary(os.path.join(args.run_b, "summary.json"))
    results_a = load_results(os.path.join(args.run_a, "results.jsonl"))
    results_b = load_results(os.path.join(args.run_b, "results.jsonl"))
    write_report(args, summary_a, summary_b, results_a, results_b)


if __name__ == "__main__":
    main()
