#!/usr/bin/env python3
"""Verify local loss path matches legacy formulas when defaults are used."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from trainer.train_sft_omni import compute_batch_loss, parse_rvq_layer_weights


class DummyAudioLogits(list):
    pass


class DummyResult:
    def __init__(self, logits, audio_logits, aux_loss=0.0):
        self.logits = logits
        self.audio_logits = audio_logits
        self.aux_loss = aux_loss


def legacy_loss(res, labels, audio_labels, loss_fct):
    text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
    text_mask = (labels.view(-1) != -100).float()
    text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)

    audio_loss = res.audio_logits[0].sum() * 0
    for i, al in enumerate(res.audio_logits):
        al_flat = al.view(-1, al.size(-1))
        target_flat = audio_labels[:, i, :].reshape(-1)
        layer_loss = loss_fct(al_flat, target_flat)
        valid_mask = (target_flat != -100).float()
        stop_mask = (target_flat == 2050).float()
        weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)
        msum = valid_mask.sum()
        if msum > 0:
            audio_loss = audio_loss + weighted_loss.sum() / (msum + 1e-9)
    audio_loss = audio_loss / 8
    return text_loss, audio_loss


def main():
    torch.manual_seed(0)
    batch, seq, text_vocab, audio_vocab = 2, 16, 32, 2112
    logits = torch.randn(batch, seq, text_vocab)
    labels = torch.randint(0, text_vocab, (batch, seq))
    labels[:, -4:] = -100

    audio_logits = []
    audio_labels = torch.full((batch, 8, seq), -100, dtype=torch.long)
    for layer in range(8):
        al = torch.randn(batch, seq, audio_vocab)
        audio_logits.append(al)
        valid_len = 10 - layer
        if valid_len > 0:
            audio_labels[:, layer, :valid_len] = torch.randint(0, audio_vocab, (batch, valid_len))
            if layer == 0:
                audio_labels[:, layer, valid_len - 1] = 2050

    res = DummyResult(logits, audio_logits)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    rvq_weights = parse_rvq_layer_weights("1,1,1,1,1,1,1,1")

    old_text, old_audio = legacy_loss(res, labels, audio_labels, loss_fct)
    new_text, new_audio = compute_batch_loss(
        res, labels, audio_labels, loss_fct, rvq_weights, 10.0, "local"
    )

    text_diff = abs(old_text.item() - new_text.item())
    audio_diff = abs(old_audio.item() - new_audio.item())
    print(f"text diff={text_diff:.2e}, audio diff={audio_diff:.2e}")
    assert text_diff < 1e-6, f"text loss mismatch: {text_diff}"
    assert audio_diff < 1e-6, f"audio loss mismatch: {audio_diff}"
    print("OK: local/default loss matches legacy implementation")


if __name__ == "__main__":
    main()
