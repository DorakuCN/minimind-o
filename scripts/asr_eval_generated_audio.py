#!/usr/bin/env python3
"""Optional ASR round-trip evaluation for generated audio files.

By default this uses the repository's SenseVoiceSmall model, matching the
audio encoder family used by MiniMind-O. A faster-whisper backend remains
available for reproducing older reports.
"""

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime

import numpy as np


def load_results(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(text):
    return " ".join((text or "").lower().strip().split())


def cer(ref, hyp):
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    return levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)


def build_output_row(row, audio_path, transcript, meta):
    return {
        "id": row["id"],
        "type": row["type"],
        "audio_path": audio_path,
        "text": row.get("text", ""),
        "asr_text": transcript,
        "cer": round(cer(row.get("text", ""), transcript), 4),
        "wer": round(wer(row.get("text", ""), transcript), 4),
        "language": meta.get("language"),
        "language_probability": meta.get("language_probability"),
    }


def summarize_items(items):
    def summarize_subset(rows):
        valid = [r for r in rows if "cer" in r and "wer" in r]
        if not valid:
            return {
                "count": 0,
                "avg_cer": None,
                "avg_wer": None,
            }
        return {
            "count": len(valid),
            "avg_cer": round(float(np.mean([r["cer"] for r in valid])), 4),
            "avg_wer": round(float(np.mean([r["wer"] for r in valid])), 4),
        }

    types = sorted({r.get("type", "") for r in items if r.get("type")})
    return {
        "overall": summarize_subset(items),
        "by_type": {
            item_type: summarize_subset([r for r in items if r.get("type") == item_type])
            for item_type in types
        },
    }


class SenseVoiceBackend:
    def __init__(self, model_path, device, project_path, use_remote_code, batch_size):
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as e:
            raise SystemExit("funasr is not installed; install it or use --backend whisper") from e

        self.model_path = model_path
        self.device = device
        self.project_path = project_path
        self.batch_size = batch_size
        self.postprocess = rich_transcription_postprocess
        self.remote_code_used = False
        self.remote_code_error = None
        kwargs = {}
        if project_path and use_remote_code:
            project_path = os.path.abspath(project_path)
            model_py = os.path.join(project_path, "model.py")
            if not os.path.exists(model_py):
                raise SystemExit(f"SenseVoice project path does not contain model.py: {project_path}")
            if project_path not in sys.path:
                sys.path.insert(0, project_path)
            kwargs["remote_code"] = model_py
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = AutoModel(
                    model=model_path,
                    trust_remote_code=True,
                    device=device,
                    disable_update=True,
                    **kwargs,
                )
            self.remote_code_used = bool(kwargs)
        except Exception as e:
            if not kwargs:
                raise
            self.remote_code_error = f"{type(e).__name__}: {e}"
            print(f"SenseVoice remote_code failed, falling back to FunASR built-in implementation: {self.remote_code_error}")
            self._unload_project_modules(project_path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = AutoModel(
                    model=model_path,
                    trust_remote_code=True,
                    device=device,
                    disable_update=True,
                )

    @staticmethod
    def _unload_project_modules(project_path):
        if not project_path:
            return
        project_path = os.path.abspath(project_path)
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != project_path]
        for name, module in list(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if module_file and os.path.abspath(module_file).startswith(project_path + os.sep):
                del sys.modules[name]

    def transcribe(self, audio_path):
        return self.transcribe_many([audio_path])[0]

    def transcribe_many(self, audio_paths):
        result = self.model.generate(
            input=audio_paths,
            cache={},
            language="auto",
            use_itn=True,
            batch_size=self.batch_size,
        )
        outputs = []
        for item in result:
            text = self.postprocess(item["text"]).strip() if item else ""
            outputs.append((text, {
                "language": None,
                "language_probability": None,
            }))
        return outputs


class WhisperBackend:
    def __init__(self, model_size, device, compute_type):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise SystemExit("faster-whisper is not installed; install it or use --backend sensevoice") from e

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = "".join(s.text for s in segments).strip()
        return text, {
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
        }


def main():
    parser = argparse.ArgumentParser(description="Run ASR on batch_validate generated audio")
    parser.add_argument("--results", required=True, help="batch_validate results.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--markdown", default="", help="Optional markdown report path")
    parser.add_argument("--backend", choices=["sensevoice", "whisper"], default="sensevoice")
    parser.add_argument("--sensevoice_model", default="model/SenseVoiceSmall", help="SenseVoice model path")
    parser.add_argument("--sensevoice_project", default="/home/genesis/Projects/SenseVoice", help="SenseVoice source checkout containing model.py")
    parser.add_argument("--use_sensevoice_remote_code", action="store_true", help="Load SenseVoice implementation from --sensevoice_project/model.py")
    parser.add_argument("--batch_size", type=int, default=16, help="SenseVoice batch size for short generated clips")
    parser.add_argument("--model_size", default="tiny", help="faster-whisper model size/path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="int8")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")

    rows = load_results(args.results)
    if args.backend == "sensevoice":
        asr = SenseVoiceBackend(
            args.sensevoice_model,
            args.device,
            args.sensevoice_project,
            args.use_sensevoice_remote_code,
            args.batch_size,
        )
        model_label = args.sensevoice_model
    else:
        asr = WhisperBackend(args.model_size, args.device, args.compute_type)
        model_label = args.model_size

    out_rows = [None] * len(rows)
    valid = []
    for idx, row in enumerate(rows):
        audio_path = row.get("decoded_audio_path")
        if not audio_path or not os.path.exists(audio_path):
            out_rows[idx] = {**row, "asr_error": "missing decoded_audio_path"}
            continue
        valid.append((idx, row, audio_path))

    if args.backend == "sensevoice":
        for start in range(0, len(valid), args.batch_size):
            batch = valid[start:start + args.batch_size]
            transcripts = asr.transcribe_many([audio_path for _, _, audio_path in batch])
            for (idx, row, audio_path), (transcript, meta) in zip(batch, transcripts):
                out_rows[idx] = build_output_row(row, audio_path, transcript, meta)
                print(f"{row['id']}: CER={out_rows[idx]['cer']:.4f} WER={out_rows[idx]['wer']:.4f}")
    else:
        for idx, row, audio_path in valid:
            transcript, meta = asr.transcribe(audio_path)
            out_rows[idx] = build_output_row(row, audio_path, transcript, meta)
            print(f"{row['id']}: CER={out_rows[idx]['cer']:.4f} WER={out_rows[idx]['wer']:.4f}")

    out_rows = [r for r in out_rows if r is not None]

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": args.results,
        "backend": args.backend,
        "model": model_label,
        "model_size": args.model_size if args.backend == "whisper" else None,
        "sensevoice_model": args.sensevoice_model if args.backend == "sensevoice" else None,
        "sensevoice_project": args.sensevoice_project if args.backend == "sensevoice" else None,
        "sensevoice_use_remote_code": args.use_sensevoice_remote_code if args.backend == "sensevoice" else None,
        "sensevoice_remote_code_used": getattr(asr, "remote_code_used", None),
        "sensevoice_remote_code_error": getattr(asr, "remote_code_error", None),
        "device": args.device,
        "batch_size": args.batch_size if args.backend == "sensevoice" else None,
        "compute_type": args.compute_type if args.backend == "whisper" else None,
        "summary": summarize_items(out_rows),
        "items": out_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("# ASR Round-Trip Evaluation\n\n")
            f.write(f"- Results: `{args.results}`\n")
            f.write(f"- ASR backend: `{args.backend}`\n")
            f.write(f"- ASR model: `{model_label}`\n")
            f.write(f"- Device: `{args.device}`\n")
            if args.backend == "sensevoice":
                f.write(f"- Batch size: `{args.batch_size}`\n")
            f.write("\n")
            f.write("## Summary\n\n")
            f.write("| Scope | Count | Avg CER | Avg WER |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
            summary = report["summary"]
            overall = summary["overall"]
            f.write(f"| overall | {overall['count']} | {overall['avg_cer']} | {overall['avg_wer']} |\n")
            for item_type, values in summary["by_type"].items():
                f.write(f"| {item_type} | {values['count']} | {values['avg_cer']} | {values['avg_wer']} |\n")
            f.write("\n## Per-Case\n\n")
            f.write("| id | type | CER | WER | ASR text |\n")
            f.write("| --- | --- | ---: | ---: | --- |\n")
            for r in out_rows:
                text = (r.get("asr_text") or r.get("asr_error") or "").replace("|", "\\|")[:160]
                f.write(f"| `{r['id']}` | {r['type']} | {r.get('cer', '')} | {r.get('wer', '')} | {text} |\n")
    print(f"Wrote ASR eval: {args.output}")


if __name__ == "__main__":
    main()
