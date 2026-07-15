"""Trainer tuned for efficient learning under the competition caps."""
import argparse
import math
import time
import torch
from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)

def parameter_groups(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "rms" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        tie_weights=True,
    )
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = torch.optim.AdamW(
        parameter_groups(model),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.steps),
    )

    model.train()
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        scheduler.step()
        losses.append(loss.item())
        step_num = step + 1
        if step_num % args.log_every == 0 or step_num == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step_num:5d}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step_num*1000:.0f} ms/step) "
                  f"lr {opt.param_groups[0]['lr']:.3g}")

    torch.save({
        "model": model.state_dict(),
        "config": {k: getattr(cfg, k) for k in dir(cfg)
                   if not k.startswith("_") and not callable(getattr(cfg, k))},
        "steps": args.steps,
        "train_loss_curve": losses,
    }, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()