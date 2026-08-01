# trainable parameters value for GPT-2 Small
| Parameter    | Value |
| ------------ | ----- |
| Layers       | 12    |
| Heads        | 12    |
| d_model      | 768   |
| d_head       | 64    |
| Vocabulary   | 50257 |
| Max Position | 1024  |
| FFN Hidden   | 3072  |

# this is for our Model
vocab_size = 10000
block_size = 128
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1

# dimensions
 X
[B,T,C]
[2,10,768]
     │
     │ QKV Linear
     ▼
[B,T,3C]
[2,10,2304]
     │
     │ split
     ▼
Q,K,V
[B,T,C]
[2,10,768]
     │
     │ split heads
     ▼
[B,H,T,D]
[2,12,10,64]
     │
     │ Q @ Kᵀ
     ▼
Attention Scores
[B,H,T,T]
[2,12,10,10]
     │
     │ mask + softmax
     ▼
Attention Weights
[B,H,T,T]
[2,12,10,10]
     │
     │ @ V
     ▼
[B,H,T,D]
[2,12,10,64]
     │
     │ concatenate heads
     ▼
[B,T,C]
[2,10,768]
     │
     │ c_proj
     ▼
[B,T,C]
[2,10,768]


# steps 
Step 1  Configuration
↓
Step 2 Token Embeddings
↓
Step 3 LayerNorm
↓
Step 4 Single Attention Head
↓
Step 5 Multi-Head Attention
↓
Step 6 Feed Forward (MLP)
↓
Step 7 GPT Block
↓
Step 8 Stack GPT Blocks
↓
Step 9 Final LayerNorm
↓
Step 10 LM Head
↓
Step 11 Training Loop
↓
Mini GPT-2 🎉



One more important distinction

# Inside Attention:

Token 1 ←→ Token 2 ←→ Token 3

# Tokens exchange information.

Inside MLP:

Token 1 → its own MLP
Token 2 → its own MLP
Token 3 → its own MLP

No token communication.

Therefore one GPT block performs:

LN → ATTENTION
     "What information should I gather?"

Residual
     "Add what I learned."

LN → MLP
     "What should I compute from it?"

Residual
     "Add what I computed."