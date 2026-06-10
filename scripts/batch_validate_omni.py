import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from transformers import AutoTokenizer, MimiModel

from dataset.omni_dataset import OmniDataset
from model.model_omni import MiniMindOmni, OmniConfig
from trainer.trainer_utils import setup_seed


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindOmni(
        OmniConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ),
        audio_encoder_path=args.audio_encoder_dir,
        vision_model_path=args.vision_dir,
    )
    weight_path = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth")
    load_result = model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    model.audio_encoder and model.audio_encoder.to(args.device)
    model.vision_encoder and model.vision_encoder.to(args.device)
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    model = model.to(dtype=dtype_map[args.dtype]).eval().to(args.device)
    mimi = None
    if args.decode_audio:
        mimi = MimiModel.from_pretrained(args.mimi_dir).eval().to(args.device)
    return model, tokenizer, mimi, weight_path, load_result


def repetition_score(text, n=4):
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / max(len(grams), 1)


def build_inputs(case, model, tokenizer, args):
    audio_inputs, audio_lens, pixel_values = None, None, None
    prompt = case.get("prompt", "")

    if case["type"] == "audio":
        mel, valid_len = OmniDataset.process_audio(case["audio_path"], model.audio_processor)
        audio_inputs = mel.unsqueeze(0).to(args.device)
        audio_lens = torch.tensor([valid_len], device=args.device)
        prompt = model.config.audio_special_token * (valid_len or 1)

    elif case["type"] == "image":
        image = Image.open(case["image_path"]).convert("RGB")
        pixel_values = {
            k: v.to(args.device)
            for k, v in model.vision_processor(images=image, return_tensors="pt").items()
        }
        prompt = prompt + "\n\n" + model.config.image_special_token * model.config.image_token_len

    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    input_ids = torch.tensor(tokenizer(chat).data["input_ids"], dtype=torch.long, device=args.device)[None, :]
    return input_ids, audio_inputs, audio_lens, pixel_values


def decode_audio_frames(frames, model, mimi, args, output_path):
    if mimi is None or not frames:
        return None
    codes = [frame for frame in frames if frame and len(frame) == 8]
    if not codes:
        return None
    mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0).to(args.device)
    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
    audio = mimi.decode(filtered).audio_values
    wav_path = output_path.rsplit(".", 1)[0] + ".wav"
    sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)
    AudioSegment.from_wav(wav_path).export(output_path, format="mp3", bitrate="64k")
    os.remove(wav_path)
    return output_path


@torch.no_grad()
def run_case(case, model, tokenizer, mimi, args):
    start = time.time()
    input_ids, audio_inputs, audio_lens, pixel_values = build_inputs(case, model, tokenizer, args)
    text = ""
    frames = []

    stream = model.generate(
        input_ids,
        tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=True,
        return_audio_codes=True,
        audio_inputs=audio_inputs,
        audio_lens=audio_lens,
        pixel_values=pixel_values,
    )
    for y, audio_frame in stream:
        if y is not None:
            decoded = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
            if decoded and decoded[-1] != "�":
                text = decoded
        if audio_frame:
            frames.append(audio_frame)

    valid_frames = [f for f in frames if f and len(f) == 8]
    flat_codes = [int(c) for f in valid_frames for c in f]
    special_rate = sum(c >= 2049 for c in flat_codes) / max(len(flat_codes), 1)
    repeat = repetition_score(text)
    min_chars = case.get("min_chars", 1)
    min_audio_frames = case.get("min_audio_frames", 1)
    pass_text = len(text.strip()) >= min_chars and repeat <= args.max_repeat_score
    pass_audio = (not case.get("expect_audio", True)) or (len(valid_frames) >= min_audio_frames)

    audio_path = None
    if args.decode_audio:
        audio_path = os.path.join(args.output_dir, "audio", f"{case['id']}.mp3")
        audio_path = decode_audio_frames(valid_frames, model, mimi, args, audio_path)

    return {
        "id": case["id"],
        "type": case["type"],
        "tags": case.get("tags", []),
        "prompt": case.get("prompt", ""),
        "audio_path": case.get("audio_path"),
        "image_path": case.get("image_path"),
        "text": text,
        "text_chars": len(text.strip()),
        "audio_frames": len(frames),
        "valid_audio_frames": len(valid_frames),
        "special_code_rate": round(special_rate, 4),
        "repeat_score": round(repeat, 4),
        "seconds": round(time.time() - start, 3),
        "pass_text": pass_text,
        "pass_audio": pass_audio,
        "pass_basic": pass_text and pass_audio,
        "decoded_audio_path": audio_path,
    }


def write_summary(results, args, weight_path, load_result):
    total = len(results)
    passed = sum(r["pass_basic"] for r in results)
    by_type = defaultdict(lambda: {"total": 0, "pass": 0})
    by_tag = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in results:
        by_type[r["type"]]["total"] += 1
        by_type[r["type"]]["pass"] += int(r["pass_basic"])
        for tag in r["tags"]:
            by_tag[tag]["total"] += 1
            by_tag[tag]["pass"] += int(r["pass_basic"])

    summary = {
        "weight_path": weight_path,
        "test_set": args.test_set,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "by_type": dict(by_type),
        "by_tag": dict(by_tag),
        "missing_keys": len(load_result.missing_keys),
        "unexpected_keys": len(load_result.unexpected_keys),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Batch Validation Summary\n\n")
        f.write(f"- Weight: `{weight_path}`\n")
        f.write(f"- Test set: `{args.test_set}`\n")
        f.write(f"- Basic pass: {passed}/{total} ({summary['pass_rate']:.2%})\n")
        f.write(f"- Load missing/unexpected keys: {summary['missing_keys']}/{summary['unexpected_keys']}\n\n")
        f.write(f"- DType: `{args.dtype}`\n")
        f.write(f"- Generation: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}\n\n")
        f.write("| id | type | pass | chars | frames | repeat | special | seconds | text |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in results:
            text = r["text"].replace("\n", " ").replace("|", "\\|")[:120]
            f.write(
                f"| {r['id']} | {r['type']} | {r['pass_basic']} | {r['text_chars']} | "
                f"{r['valid_audio_frames']} | {r['repeat_score']:.4f} | {r['special_code_rate']:.4f} | "
                f"{r['seconds']:.3f} | {text} |\n"
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch validate MiniMind-O checkpoints")
    parser.add_argument("--test_set", default="dataset/eval_muon_mini.jsonl")
    parser.add_argument("--output_dir", default="eval_results/sft_zero_muon_batch")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_zero_muon")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--audio_encoder_dir", default="model/SenseVoiceSmall")
    parser.add_argument("--vision_dir", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--mimi_dir", default="model/mimi")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--max_repeat_score", type=float, default=0.35)
    parser.add_argument("--decode_audio", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.decode_audio:
        os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)
    setup_seed(args.seed)

    cases = load_cases(args.test_set)
    model, tokenizer, mimi, weight_path, load_result = load_model(args)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    results = []

    with open(results_path, "w", encoding="utf-8") as f:
        for i, case in enumerate(cases, start=1):
            result = run_case(case, model, tokenizer, mimi, args)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{i:02d}/{len(cases)}] {result['id']} "
                f"pass={result['pass_basic']} chars={result['text_chars']} "
                f"frames={result['valid_audio_frames']} repeat={result['repeat_score']:.3f}"
            )

    summary = write_summary(results, args, weight_path, load_result)
    print(f"Summary: {summary['passed']}/{summary['total']} basic pass -> {summary['pass_rate']:.2%}")
    print(f"Results: {results_path}")
    print(f"Report: {os.path.join(args.output_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
