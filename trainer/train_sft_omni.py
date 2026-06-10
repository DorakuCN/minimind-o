import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import time
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_omni import OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_omni_model, omni_checkpoint, SkipBatchSampler, log_model_params
from trainer.optimizers import build_optimizer

warnings.filterwarnings('ignore')

DEFAULT_RVQ_WEIGHTS = (1.0,) * 8
LEGACY_AUDIO_LAYERS = 8


def parse_rvq_layer_weights(raw):
    weights = [float(x.strip()) for x in raw.split(',')]
    if len(weights) != LEGACY_AUDIO_LAYERS:
        raise ValueError(f"--rvq_layer_weights expects {LEGACY_AUDIO_LAYERS} values, got {len(weights)}")
    return tuple(weights)


def uses_legacy_audio_denominator(rvq_weights):
    return all(abs(w - 1.0) < 1e-9 for w in rvq_weights)


def compute_batch_loss(res, labels, audio_labels, loss_fct, rvq_weights, audio_stop_weight, loss_norm):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    use_global = loss_norm == 'global' and dist.is_initialized()
    stop_multiplier = max(audio_stop_weight - 1.0, 0.0)

    text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
    text_mask = (labels.view(-1) != -100).float()
    local_text_sum = (text_loss_raw * text_mask).sum()
    local_text_count = text_mask.sum()
    if use_global:
        stats = torch.stack([local_text_sum, local_text_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_text_count = stats[1]
        text_loss = (local_text_sum / (global_text_count + 1e-9)) * world_size
    else:
        text_loss = local_text_sum / (local_text_count + 1e-9)

    audio_loss = res.audio_logits[0].sum() * 0
    layer_means = []
    active_weights = []
    legacy = uses_legacy_audio_denominator(rvq_weights)
    for i, al in enumerate(res.audio_logits):
        al_flat = al.view(-1, al.size(-1))
        target_flat = audio_labels[:, i, :].reshape(-1)
        layer_loss = loss_fct(al_flat, target_flat)
        valid_mask = (target_flat != -100).float()
        stop_mask = (target_flat == 2050).float()
        weighted_loss = layer_loss * valid_mask * (1 + stop_mask * stop_multiplier)
        local_layer_sum = weighted_loss.sum()
        local_layer_count = valid_mask.sum()
        if local_layer_count > 0:
            if use_global:
                stats = torch.stack([local_layer_sum, local_layer_count])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                global_layer_count = stats[1]
                layer_mean = (local_layer_sum / (global_layer_count + 1e-9)) * world_size
            else:
                layer_mean = local_layer_sum / (local_layer_count + 1e-9)
            layer_means.append(layer_mean)
            active_weights.append(rvq_weights[i])

    if legacy:
        audio_loss = sum(layer_means) / LEGACY_AUDIO_LAYERS if layer_means else audio_loss
    elif active_weights:
        weight_sum = sum(active_weights)
        audio_loss = sum(w * l for w, l in zip(active_weights, layer_means)) / (weight_sum + 1e-9)
    return text_loss, audio_loss


@torch.no_grad()
def evaluate_loader(loader, max_batches=None):
    model.eval()
    totals = torch.zeros(4, device=args.device, dtype=torch.float64)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = batch
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
            text_loss, audio_loss = compute_batch_loss(
                res, labels, audio_labels, loss_fct, rvq_weights, args.audio_stop_weight, args.loss_norm
            )
            batch_loss = text_loss + audio_loss + res.aux_loss
        totals[0] += batch_loss.item()
        totals[1] += text_loss.item()
        totals[2] += audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0.0
        totals[3] += 1.0
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    if totals[3].item() == 0:
        return None
    count = totals[3].item()
    return {
        "loss": float(totals[0].item() / count),
        "text": float(totals[1].item() / count),
        "audio": float(totals[2].item() / count),
    }


def omni_collate_fn(batch):
    """自定义collate函数，处理变长audio_inputs和pixel_values"""
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)
    else:
        pixel_values = None
    spk_emb = torch.stack(spk_emb)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb


def resolve_rank_sharded_data_path(data_path, local_rank):
    if not dist.is_initialized():
        raise RuntimeError("--rank_sharded_data requires DDP/torchrun")
    world_size = dist.get_world_size()
    rank_paths = []
    for raw_path in data_path.split(','):
        path = raw_path.strip()
        base_dir = os.path.dirname(path) or "."
        stem, ext = os.path.splitext(os.path.basename(path))
        shard_path = os.path.join(
            base_dir,
            "_full_shards",
            f"{stem}.rank{local_rank:02d}-of{world_size:02d}{ext}",
        )
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"rank shard not found: {shard_path}")
        rank_paths.append(shard_path)
    return ",".join(rank_paths)


def validate_equal_batch_count(dataset_len, batch_size, rank_sharded_data):
    if not dist.is_initialized() or not rank_sharded_data:
        return
    local_batches = (dataset_len + batch_size - 1) // batch_size
    current = torch.tensor([dataset_len, local_batches], device=args.device, dtype=torch.long)
    gathered = [torch.zeros_like(current) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, current)
    stats = [(int(t[0].item()), int(t[1].item())) for t in gathered]
    Logger(f'Rank-sharded data stats (rows, batches): {stats}')
    if len({batches for _, batches in stats}) != 1:
        raise ValueError(f"rank-sharded loaders have unequal batch counts: {stats}")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, val_loader=None):
    start_time = time.time()
    last_step = start_step
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_batch = args.batch_size * world_size
    final_loss = final_text_loss = final_audio_loss = 0.0
    warmup_steps = int(args.warmup_ratio * args.epochs * iters)
    total_steps = args.epochs * iters
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    last_val_metrics = None
    for step, (input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        last_step = step
        global_step = epoch * iters + step
        lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
            text_loss, audio_loss = compute_batch_loss(
                res, labels, audio_labels, loss_fct, rvq_weights, args.audio_stop_weight, args.loss_norm
            )
            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            final_loss, final_text_loss, final_audio_loss = current_loss, text_loss_val, audio_loss_val
            current_lr = optimizer.param_groups[-1]['lr']
            steps_done = step - start_step
            samples_per_sec = (steps_done * global_batch) / max(spend_time, 1e-9)
            eta_min = spend_time / max(steps_done, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                f'text: {text_loss_val:.4f}, audio: {audio_loss_val:.4f}, lr: {current_lr:.8f}, '
                f'samples/sec: {samples_per_sec:.2f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "text_loss": text_loss_val,
                    "audio_loss": audio_loss_val,
                    "lr": current_lr,
                    "samples_per_sec": samples_per_sec,
                    "epoch_time": eta_min,
                })

        if val_loader and args.val_interval > 0 and step % args.val_interval == 0:
            last_val_metrics = evaluate_loader(val_loader)
            if is_main_process() and last_val_metrics:
                Logger(
                    f'Val [{epoch+1}/{args.epochs}]({step}/{iters}): '
                    f'loss: {last_val_metrics["loss"]:.4f}, text: {last_val_metrics["text"]:.4f}, '
                    f'audio: {last_val_metrics["audio"]:.4f}'
                )
                if wandb:
                    wandb.log({
                        "val_loss": last_val_metrics["loss"],
                        "val_text_loss": last_val_metrics["text"],
                        "val_audio_loss": last_val_metrics["audio"],
                    })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if omni_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.')}
            torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
            omni_checkpoint(omni_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - start_time
    steps_done = last_step - start_step
    return {
        "epoch": epoch + 1,
        "steps": steps_done,
        "final_loss": round(final_loss, 4),
        "final_text_loss": round(final_text_loss, 4),
        "final_audio_loss": round(final_audio_loss, 4),
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_sec": round((steps_done * global_batch) / max(elapsed, 1e-9), 2),
        "val_metrics": last_val_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-O SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_omni', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"], help="优化器类型")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01, help="AdamW/fallback AdamW权重衰减")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adamw_beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--adamw_eps", type=float, default=1e-8, help="AdamW eps")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon矩阵参数学习率；设为-1时沿用learning_rate")
    parser.add_argument("--muon_weight_decay", type=float, default=0.0, help="Muon矩阵参数权重衰减")
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum")
    parser.add_argument("--muon_nesterov", type=int, default=1, choices=[0, 1], help="Muon是否使用Nesterov")
    parser.add_argument("--muon_ns_steps", type=int, default=5, help="Muon Newton-Schulz迭代步数")
    parser.add_argument("--muon_eps", type=float, default=1e-7, help="Muon Newton-Schulz数值稳定项")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/train_t2a_mini.parquet", help="训练数据路径（parquet格式）")
    parser.add_argument('--rank_sharded_data', default=0, type=int, choices=[0, 1], help="是否按DDP rank加载_full_shards中的parquet分片")
    parser.add_argument("--audio_encoder_dir", type=str, default="../model/SenseVoiceSmall", help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default="../model/siglip2-base-p32-256-ve", help="CLIP视觉模型路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-SFT", help="wandb项目名")
    parser.add_argument("--wandb_run_name", type=str, default="", help="SwanLab run 名称；为空时使用自动命名")
    parser.add_argument("--metrics_path", type=str, default="", help="训练结束写入阶段指标 JSON 的路径")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="LR warmup 占总 step 比例；0 表示关闭")
    parser.add_argument("--loss_norm", type=str, default="local", choices=["local", "global"], help="loss 归一化方式")
    parser.add_argument("--rvq_layer_weights", type=str, default="1,1,1,1,1,1,1,1", help="8 层 RVQ loss 权重，逗号分隔")
    parser.add_argument("--audio_stop_weight", type=float, default=10.0, help="audio stop token(2050) 相对权重")
    parser.add_argument("--val_data_path", type=str, default="", help="验证集 parquet 路径；为空则关闭 val loss")
    parser.add_argument("--val_interval", type=int, default=0, help="每 N step 跑一次 val loss；0 表示关闭")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.muon_lr is not None and args.muon_lr < 0:
        args.muon_lr = None
    rvq_weights = parse_rvq_layer_weights(args.rvq_layer_weights)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = args.wandb_run_name or f"MiniMind-O-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                        audio_encoder_path=args.audio_encoder_dir,
                                        vision_model_path=args.vision_dir,
                                        save_dir=args.save_dir, device=args.device,
                                        freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)

    if args.use_compile == 1:
        model = torch.compile(model)

    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)

    if args.mode == 'audio_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.audio_proj.parameters(): p.requires_grad = True
    elif args.mode == 'vision_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.vision_proj.parameters(): p.requires_grad = True
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable: {trainable:.2f}M | Mode: {args.mode} | Freeze: {args.freeze_backbone} | Compile: {"on" if args.use_compile else "off"}')
    Logger(f'Loss norm: {args.loss_norm} | Warmup ratio: {args.warmup_ratio} | RVQ weights: {args.rvq_layer_weights}')

    val_ds = None
    if args.val_data_path:
        Logger(f'Val path: {args.val_data_path}')
        val_ds = OmniDataset(
            args.val_data_path,
            tokenizer,
            audio_processor=model.audio_processor,
            vision_processor=model.vision_processor,
            max_length=args.max_seq_len,
            image_token_len=model.config.image_token_len,
        )

    # scheduled_sampling 现在会自动保护 image/audio token 的连续性
    effective_data_path = resolve_rank_sharded_data_path(args.data_path, local_rank) if args.rank_sharded_data else args.data_path
    Logger(f'Data path: {effective_data_path}')
    train_ds = OmniDataset(
        effective_data_path,
        tokenizer,
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=model.config.image_token_len
    )

    validate_equal_batch_count(len(train_ds), args.batch_size, bool(args.rank_sharded_data))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() and not args.rank_sharded_data else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model, args)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        raw_model = getattr(model, '_orig_mod', model)
        load_result = raw_model.load_state_dict(ckp_data['model'], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            Logger(f'模型resume加载差异: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}')
        try:
            optimizer.load_state_dict(ckp_data['optimizer'])
        except ValueError as e:
            Logger(f'优化器状态不兼容，已跳过optimizer resume: {e}')
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    training_start = time.time()
    epoch_metrics = []
    total_steps = 0
    val_loader = None
    if val_ds is not None:
        if dist.is_initialized():
            val_sampler = DistributedSampler(val_ds, shuffle=False)
            val_batch_sampler = SkipBatchSampler(val_sampler, args.batch_size, 0)
        else:
            val_batch_sampler = SkipBatchSampler(list(range(len(val_ds))), args.batch_size, 0)
        val_loader = DataLoader(val_ds, batch_sampler=val_batch_sampler, collate_fn=omni_collate_fn, num_workers=args.num_workers, pin_memory=True)
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            metrics = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, val_loader)
        else:
            metrics = train_epoch(epoch, loader, len(loader), 0, wandb, val_loader)
        epoch_metrics.append(metrics)
        total_steps += metrics["steps"]

    if is_main_process() and args.metrics_path:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_elapsed = time.time() - training_start
        last_epoch = epoch_metrics[-1] if epoch_metrics else {}
        metrics_doc = {
            "save_weight": args.save_weight,
            "mode": args.mode,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "learning_rate": args.learning_rate,
            "muon_lr": args.muon_lr,
            "data_path": args.data_path,
            "from_weight": args.from_weight,
            "total_steps": total_steps,
            "elapsed_seconds": round(total_elapsed, 2),
            "samples_per_sec": round((total_steps * args.batch_size * world_size) / max(total_elapsed, 1e-9), 2),
            "final_loss": last_epoch.get("final_loss"),
            "final_text_loss": last_epoch.get("final_text_loss"),
            "final_audio_loss": last_epoch.get("final_audio_loss"),
            "epoch_metrics": epoch_metrics,
            "wandb_run_name": args.wandb_run_name or None,
            "warmup_ratio": args.warmup_ratio,
            "loss_norm": args.loss_norm,
            "rvq_layer_weights": args.rvq_layer_weights,
            "audio_stop_weight": args.audio_stop_weight,
            "val_data_path": args.val_data_path or None,
            "val_interval": args.val_interval,
            "last_val_metrics": last_epoch.get("val_metrics"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_path)), exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_doc, f, ensure_ascii=False, indent=2)
        Logger(f'Stage metrics saved to {args.metrics_path}')

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
