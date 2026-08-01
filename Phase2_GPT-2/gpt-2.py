import os
import math
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 10000
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) # Q * transpose(K)/sqrt(d)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # it trims like lower symmetric matrix as ones and uppersymmetric matrix as 0
        att = torch.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        # x = self.c_fc(x)
        # print(x.shape)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = token_emb + pos_emb
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            logits_flat = logits.reshape(B * T, self.config.vocab_size)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(
                    logits < v[:, [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        self.train(was_training)
        return idx

    
@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device, eval_iters=20):
    """Estimate loss on train and val splits.
    
    Args:
        eval_iters: Number of batches to evaluate (default: 20)
    """
    model.eval()

    out = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(
                data,
                batch_size=4,
                block_size=config.block_size,
                device=device
            )

            _, loss = model(xb, yb)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


class CosineAnnealingWarmupScheduler:
    """Learning rate scheduler with linear warmup then cosine decay."""
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        base_lr=3e-4,
        min_lr=1e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1


def save_checkpoint(model, optimizer, config, step, loss, filepath, scheduler=None, rng_state=None):
    """Save checkpoint with all training state.
    
    Args:
        scheduler: Optional scheduler to save current_step
        rng_state: Optional random state for reproducibility
    """
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_step"] = scheduler.current_step
    
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, device, scheduler=None):
    """Load checkpoint and restore all training state.
    
    Args:
        scheduler: Optional scheduler to restore current_step
    
    Returns:
        checkpoint dict, start_step
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Ensure model is on the correct device before loading
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Move optimizer state tensors to device to avoid device mismatch errors
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    # Restore scheduler state (critical for LR schedule continuity!)
    if scheduler is not None and "scheduler_step" in checkpoint:
        scheduler.current_step = checkpoint["scheduler_step"]
    
    # Restore random state for reproducibility
    if "rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["rng_state"])
    
    return checkpoint, checkpoint["step"] + 1


if __name__ == "__main__":
    # === CONFIGURATION ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = GPTConfig(
        vocab_size=50257,
        block_size=64,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )

    # === TRAINING HYPERPARAMETERS ===
    num_steps = 1000
    batch_size = 8
    learning_rate = 3e-4
    eval_interval = 100
    save_interval = 500
    warmup_steps = 100

    # === INITIALIZE MODEL & OPTIMIZER ===
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        base_lr=learning_rate,
        min_lr=1e-5
    )

    # === LOAD DATA ===
    enc = tiktoken.get_encoding("gpt2")

    with open("text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = enc.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)

    print(f"Characters: {len(text):,}")
    print(f"Tokens: {len(data):,}")

    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # === RESUME FROM CHECKPOINT IF EXISTS ===
    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists("checkpoint.pt"):
        print("Loading checkpoint...")
        checkpoint, start_step = load_checkpoint(
            "checkpoint.pt", model, optimizer, device, scheduler=scheduler
        )
        print(
            f"Resumed from step {checkpoint['step']} "
            f"(loss={checkpoint['loss']:.4f}, "
            f"scheduler_step={scheduler.current_step})"
        )

    # === TRAINING LOOP ===
    for step in range(start_step, num_steps):
        # Evaluate
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device, eval_iters=20)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:4d} | "
                f"lr={current_lr:.6f} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                rng_state = torch.random.get_rng_state()
                save_checkpoint(
                    model, optimizer, config, step,
                    losses["val"], "best.pt",
                    scheduler=scheduler,
                    rng_state=rng_state
                )
                print(f"✓ New best model! val={best_val_loss:.4f}")

        # Forward pass
        xb, yb = get_batch(
            train_data,
            batch_size,
            config.block_size,
            device
        )

        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Debug: Print gradient norm occasionally
        if step % (eval_interval * 2) == 0 and step > 0:
            print(f"  → grad_norm={total_norm:.4f}")

        # Save checkpoint (latest)
        if step % save_interval == 0:
            rng_state = torch.random.get_rng_state()
            save_checkpoint(
                model, optimizer, config, step,
                loss.item(), "checkpoint.pt",
                scheduler=scheduler,
                rng_state=rng_state
            )

    # Save final checkpoint
    rng_state = torch.random.get_rng_state()
    save_checkpoint(
        model, optimizer, config, num_steps - 1,
        loss.item(), "checkpoint.pt",
        scheduler=scheduler,
        rng_state=rng_state
    )

    # === GENERATION ===
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)
    prompt = "The company"
    prompt_idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    generated_idx = model.generate(
        prompt_idx, max_new_tokens=100, temperature=0.9, top_k=50
    )
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerated text:")
    print("-" * 60)
    print(enc.decode(generated_idx[0].tolist()))
    print("-" * 60)
