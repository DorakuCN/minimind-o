import torch
from torch import optim

from trainer.trainer_utils import Logger


def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    if g.ndim < 2:
        raise ValueError("Muon update expects at least 2D tensors")

    transposed = g.size(-2) > g.size(-1)
    x = g.to(torch.bfloat16 if g.is_cuda else torch.float32)
    if transposed:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xa = x @ x.mT
        xb = b * xa + c * (xa @ xa)
        x = a * x + xb @ x

    if transposed:
        x = x.mT
    return x.to(dtype=g.dtype)


def muon_update(grad, momentum_buffer, momentum=0.95, nesterov=True, ns_steps=5, eps=1e-7):
    momentum_buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps, eps=eps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape)


class MuonWithAuxAdam(optim.Optimizer):
    """Muon for hidden matrices, AdamW for parameters that should not use Muon."""

    def __init__(self, param_groups):
        for group in param_groups:
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("eps", 1e-7)
                group["params"] = sorted(group["params"], key=lambda p: p.numel(), reverse=True)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("weight_decay", 0.01)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

            update = muon_update(
                grad,
                state["momentum_buffer"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_steps=group["ns_steps"],
                eps=group["eps"],
            )
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])

    def _step_adamw_group(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdamW fallback does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1

            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])

            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1
            p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)


def _is_muon_parameter(name, param):
    if (not param.requires_grad) or param.ndim != 2:
        return False
    lowered = name.lower()
    excluded = ("embed", "lm_head", "head", "norm", "bias", "audio_encoder", "vision_encoder")
    return not any(token in lowered for token in excluded)


def build_optimizer(model, args):
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            [p for _, p in trainable_named_params],
            lr=args.learning_rate,
            weight_decay=args.adamw_weight_decay,
            betas=(args.adamw_beta1, args.adamw_beta2),
            eps=args.adamw_eps,
        )
        optimizer.param_groups[0]["lr_scale"] = 1.0
        Logger(f'Optimizer: AdamW | params={sum(p.numel() for _, p in trainable_named_params) / 1e6:.2f}M')
        return optimizer

    muon_params, adam_params = [], []
    for name, param in trainable_named_params:
        if _is_muon_parameter(name, param):
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_lr = args.muon_lr if args.muon_lr is not None else args.learning_rate
    param_groups = []
    if adam_params:
        param_groups.append(dict(
            params=adam_params,
            lr=args.learning_rate,
            lr_scale=1.0,
            weight_decay=args.adamw_weight_decay,
            betas=(args.adamw_beta1, args.adamw_beta2),
            eps=args.adamw_eps,
            use_muon=False,
        ))
    if muon_params:
        param_groups.append(dict(
            params=muon_params,
            lr=muon_lr,
            lr_scale=muon_lr / args.learning_rate if args.learning_rate != 0 else 1.0,
            weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum,
            nesterov=bool(args.muon_nesterov),
            ns_steps=args.muon_ns_steps,
            eps=args.muon_eps,
            use_muon=True,
        ))

    optimizer = MuonWithAuxAdam(param_groups)
    muon_m = sum(p.numel() for p in muon_params) / 1e6
    adam_m = sum(p.numel() for p in adam_params) / 1e6
    native = 'available' if hasattr(optim, 'Muon') else 'unavailable'
    Logger(f'Optimizer: MuonWithAuxAdam | muon={muon_m:.2f}M | adamw={adam_m:.2f}M | muon_lr={muon_lr:g} | torch.optim.Muon={native}')
    return optimizer
