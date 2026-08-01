import torch
import tiktoken

def get_batch(data, batch_size, block_size):
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

    return x, y

enc = tiktoken.get_encoding("gpt2")

with open("savary_and_sage_company.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = enc.encode(text)

data = torch.tensor(tokens, dtype=torch.long)

print("Characters:", len(text))
print("Tokens:", len(data))

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Train:", train_data.shape)
print("Validation:", val_data.shape)

xb, yb = get_batch(
    train_data,
    batch_size=4,
    block_size=128
)

print("X:", xb.shape)
print("Y:", yb.shape)

print("\nFirst X:")
print(xb[0])

print("\nFirst Y:")
print(yb[0])


















































import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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

if __name__ == "__main__":
    config = GPTConfig()

    model = GPT(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=50257,
        block_size=64,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )

    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    num_steps = 1000
    batch_size = 8
    enc = tiktoken.get_encoding("gpt2")

    with open("savary_and_sage_company.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = enc.encode(text)

    data = torch.tensor(tokens, dtype=torch.long)

    print("Characters:", len(text))
    print("Tokens:", len(data))

    n = int(0.9 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    for step in range(num_steps):

        xb, yb = get_batch(
            train_data,
            batch_size,
            config.block_size,
            device
        )

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f}"
            )
    









    # n = int(0.9 * len(data))

    # train_data = data[:n]
    # val_data = data[n:]

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=3e-4
    # )

    # xb, yb = get_batch(
    # train_data,
    # batch_size=4,
    # block_size=config.block_size
    # )

    # logits, loss = model(xb, yb)

    # # logits, loss = model(idx, targets)

    # # print("Loss:", loss.item())

    # # w = model.lm_head.weight

    # # print("Weight-before:", w[0, 0].item())

    # optimizer.zero_grad()

    # loss.backward()

    # # print("Gradient:", w.grad[0,0].item())

    # optimizer.step()

    # # print("Weight-after:", w[0, 0].item())

    # # print("Gradient after:", w.grad[0, 0].item())