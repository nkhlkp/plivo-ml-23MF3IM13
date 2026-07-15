"""Optimized trainer: Gradient accumulation, AdamW, and stable OneCycleLR scheduler."""
import argparse
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--micro_batch", type=int, default=4) # 4 physical
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens (vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Scheduler with fixed tail death
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=args.lr, 
        total_steps=args.steps, 
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=10.0
    )

    model.train()
    t0 = time.time()
    losses = []
    accum_steps = 4 # 4 * 4 = 16 effective batch size
    
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        
        # Gradient Accumulation Loop
        for _ in range(accum_steps):
            x, y = get_batch(ids, cfg.block_size, args.micro_batch, device)
            _, loss = model(x, y)
            loss = loss / accum_steps
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step() 
        
        losses.append(step_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step) | lr: {scheduler.get_last_lr()[0]:.2e}")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()