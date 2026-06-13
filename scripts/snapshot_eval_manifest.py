#!/usr/bin/env python3
"""Snapshot an eval jsonl file and referenced media with sha256 hashes."""

import argparse
import hashlib
import json
import os
from datetime import datetime


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                case = json.loads(line)
                case["_line"] = line_no
                cases.append(case)
    return cases


def file_record(root, path):
    abs_path = os.path.abspath(os.path.join(root, path))
    exists = os.path.exists(abs_path)
    record = {
        "path": path,
        "exists": exists,
        "bytes": os.path.getsize(abs_path) if exists else None,
        "sha256": sha256_file(abs_path) if exists else None,
    }
    return record


def write_markdown(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Eval Manifest Snapshot\n\n")
        f.write(f"- Generated: {manifest['generated_at']}\n")
        f.write(f"- Test set: `{manifest['test_set']['path']}`\n")
        f.write(f"- Test set sha256: `{manifest['test_set']['sha256']}`\n")
        f.write(f"- Cases: {manifest['case_count']}\n")
        f.write(f"- Referenced files: {len(manifest['files'])}\n")
        f.write(f"- Missing files: {len(manifest['missing_files'])}\n\n")

        f.write("## Files\n\n")
        f.write("| path | exists | bytes | sha256 |\n")
        f.write("| --- | ---: | ---: | --- |\n")
        for r in manifest["files"]:
            f.write(
                f"| `{r['path']}` | {r['exists']} | {r['bytes'] if r['bytes'] is not None else ''} | "
                f"`{r['sha256'] or ''}` |\n"
            )

        f.write("\n## Cases\n\n")
        f.write("| line | id | type | tags | prompt/media |\n")
        f.write("| ---: | --- | --- | --- | --- |\n")
        for c in manifest["cases"]:
            media = c.get("audio_path") or c.get("image_path") or c.get("prompt", "")
            media = str(media).replace("\n", " ").replace("|", "\\|")[:100]
            tags = ",".join(c.get("tags", []))
            f.write(f"| {c['_line']} | `{c['id']}` | {c['type']} | {tags} | {media} |\n")


def main():
    parser = argparse.ArgumentParser(description="Create a reproducible manifest for an eval jsonl")
    parser.add_argument("--test_set", default="dataset/eval_muon_mini.jsonl")
    parser.add_argument("--root", default=".", help="Project root used to resolve relative media paths")
    parser.add_argument("--output", required=True, help="Output JSON manifest path")
    parser.add_argument("--markdown", default="", help="Optional output markdown path")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    test_set_path = os.path.abspath(os.path.join(root, args.test_set))
    cases = load_cases(test_set_path)

    paths = []
    for c in cases:
        for key in ("audio_path", "image_path"):
            if c.get(key):
                paths.append(c[key])
    seen = set()
    files = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        files.append(file_record(root, path))

    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": root,
        "test_set": {
            "path": args.test_set,
            "bytes": os.path.getsize(test_set_path),
            "sha256": sha256_file(test_set_path),
        },
        "case_count": len(cases),
        "cases": cases,
        "files": files,
        "missing_files": [r["path"] for r in files if not r["exists"]],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        write_markdown(args.markdown, manifest)
    print(f"Wrote manifest: {args.output}")
    if args.markdown:
        print(f"Wrote markdown: {args.markdown}")
    if manifest["missing_files"]:
        print(f"Missing files: {manifest['missing_files']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
