#!/usr/bin/env python
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from model.model_omni import MiniMindOmni, OmniConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate MiniMind-O Omni parameter count without loading encoders")
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_hidden_layers", type=int, default=18)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--intermediate_size", type=int, default=6464)
    parser.add_argument("--talker_hidden_size", type=int, default=1024)
    parser.add_argument("--num_talker_hidden_layers", type=int, default=4)
    parser.add_argument("--talker_num_attention_heads", type=int, default=8)
    parser.add_argument("--talker_num_key_value_heads", type=int, default=4)
    parser.add_argument("--talker_head_dim", type=int, default=128)
    parser.add_argument("--talker_intermediate_size", type=int, default=3264)
    parser.add_argument("--use_moe", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        talker_hidden_size=args.talker_hidden_size,
        num_talker_hidden_layers=args.num_talker_hidden_layers,
        talker_num_attention_heads=args.talker_num_attention_heads,
        talker_num_key_value_heads=args.talker_num_key_value_heads,
        talker_head_dim=args.talker_head_dim,
        talker_intermediate_size=args.talker_intermediate_size,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindOmni(cfg, audio_encoder_path="/missing", vision_model_path="/missing")
    ignored = ("audio_encoder.", "vision_encoder.")
    total = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(ignored))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params_non_frozen_encoders={total / 1e6:.2f}M")
    print(f"trainable={trainable / 1e6:.2f}M")
    print(
        "arch="
        f"hidden={cfg.hidden_size},layers={cfg.num_hidden_layers},heads={cfg.num_attention_heads},"
        f"kv={cfg.num_key_value_heads},head_dim={cfg.head_dim},ffn={cfg.intermediate_size},"
        f"talker_hidden={cfg.talker_hidden_size},talker_layers={cfg.num_talker_hidden_layers}"
    )


if __name__ == "__main__":
    main()
