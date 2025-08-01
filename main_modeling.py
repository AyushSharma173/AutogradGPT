"""
Clean Transformer Language Model Implementation
Educational version for CS336 - Stanford

This module implements a complete transformer language model with:
- Custom neural network layers (Linear, Embedding, RMSNorm)
- Multi-head self-attention with RoPE
- SwiGLU feedforward networks
- Training loop with evaluation
- AdamW optimizer implementation
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Optional
from einops import rearrange
from tokenizer import Tokenizer
import wandb


# =============================================================================
# BASIC NEURAL NETWORK COMPONENTS
# =============================================================================

class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.embedding_lookup = Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.embedding_lookup, 0.0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_lookup[token_ids]


class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.rms_weights = Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is dimensions (B, T, D)

        orig_dtype = x.dtype

        x32 = x.to(torch.float32)
        
        # FLOPs: B * T * D
        sq_x = x32.pow(2)

        # FLOPs: B*T*(D-1) + B*T
        x_2_mean = sq_x.mean(dim=-1, keepdim=True) 

        # FLOPs: B*T
        norm_plus_eps = x_2_mean + self.eps

        # FLOPs: B*T
        rms = torch.sqrt(norm_plus_eps)

        # FLOPs: B*T*D
        x_norm = x32 / rms

        # FLOPs: B*T*D
        y = x_norm* self.rms_weights

        return y.to(orig_dtype)


# -------------------------------------------------------------------
# FINAL TOTAL FLOPs for RMSNorm forward:
#
#   = (1)    B * T * D         # pow(2)
#   + (2)    B * T * (D-1)     # sum for mean
#   + (2)    B * T             # divide for mean
#   + (3)    B * T             # add eps
#   + (4)    B * T             # sqrt
#   + (5)    B * T * D         # divide by rms
#   + (6)    B * T * D         # multiply by weight
#   --------------------------------------------
#   = 3 * B * T * D + B * T * (D-1) + 3 * B * T
#
# For large D, this is approximately:
#   ≈ 4 * B * T * D
# -------------------------------------------------------------------

# For layernorm, this is = 8 * B * T * D


# =============================================================================
# ATTENTION MECHANISM COMPONENTS
# =============================================================================

class RopeEmbeddings(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        inv_freq = inv_freq.to(device)

        # Precompute position encodings
        positions = torch.arange(max_seq_len).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        # Split into even and odd indices
        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices

        # Apply rotation
        rotated_even = x1 * cos_pos - x2 * sin_pos
        rotated_odd = x1 * sin_pos + x2 * cos_pos

        # Recombine
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated = rotated.flatten(-2)

        return rotated


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None):
    scores = torch.einsum("...qd, ...kd->...qk", Q, K)
    scores = scores / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", attn_weights, V)

    return output


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        if theta is not None:
            self.rope = RopeEmbeddings(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        factory_kwargs = {"device": device, "dtype": dtype}

        # Weight matrices for Q, K, V, and output projection
        self.q_weights = nn.Parameter(torch.empty(self.d_model, self.d_model, **factory_kwargs))
        self.k_weights = nn.Parameter(torch.empty(self.d_model, self.d_model, **factory_kwargs))
        self.v_weights = nn.Parameter(torch.empty(self.d_model, self.d_model, **factory_kwargs))
        self.o_weights = nn.Parameter(torch.empty(self.d_model, self.d_model, **factory_kwargs))

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.q_weights)
        torch.nn.init.xavier_uniform_(self.k_weights)
        torch.nn.init.xavier_uniform_(self.v_weights)
        torch.nn.init.xavier_uniform_(self.o_weights)

    def forward(self, in_features: torch.Tensor, use_rope=False):
        # Compute Q, K, V
        Q = torch.einsum("b l d, k d -> b l k", in_features, self.q_weights)
        K = torch.einsum("b l d, k d -> b l k", in_features, self.k_weights)
        V = torch.einsum("b l d, k d -> b l k", in_features, self.v_weights)

        b, l, k = Q.shape
        h = self.num_heads

        # Reshape for multi-head attention
        Q = rearrange(Q, "b l (h k) -> b h l k", h=h)
        K = rearrange(K, "b l (h k) -> b h l k", h=h)
        V = rearrange(V, "b l (h k) -> b h l k", h=h)

        # Apply RoPE if specified
        if use_rope:
            pos = torch.arange(l, device=in_features.device)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # Create causal mask
        seq_len = in_features.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Attention computation
        context = scaled_dot_product_attention(Q, K, V, mask)
        context = rearrange(context, "b h l k -> b l (h k)")

        # Output projection
        output = torch.einsum("b l d, m d -> b l m", context, self.o_weights)

        return output


# =============================================================================
# FEEDFORWARD NETWORK
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation function with gating mechanism."""
    
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        if d_ff is not None:
            self.d_ff = d_ff
        else:
            # Standard scaling for SwiGLU
            raw_dff = (8 * d_model) / 3
            d_ff = math.ceil(raw_dff / 64) * 64

        self.w1 = Linear(d_model, d_ff)  # Gate projection
        self.w2 = Linear(d_ff, d_model)  # Output projection
        self.w3 = Linear(d_model, d_ff)  # Value projection

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        
        a = self.w1(x32)  # Gate
        b = self.w3(x32)  # Value
        
        gated = F.silu(a) * b  # SwiGLU activation
        output = self.w2(gated)
        
        return output.to(orig_dtype)


# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_pos=None):
        # Multi-head self-attention with residual connection
        h = self.ln1(x)
        attn_out = self.attn(h, use_rope=True)
        x = x + attn_out

        # Feedforward network with residual connection
        h = self.ln2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        
        return x


class TransformerLanguageModel(nn.Module):
    """Complete Transformer Language Model."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 theta: float, vocab_size: int, context_length: int, num_layers: int,
                 device=None, dtype=None):
        super().__init__()
        
        # Token embedding
        self.tok_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Language modeling head
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.tok_emb(token_ids)

        # Pass through each transformer block
        pos = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, token_pos=pos)

        # Final normalization and projection to vocabulary
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits

    def top_p_filtering(self, probs, top_p=0.9):
        """Apply top-p (nucleus) sampling filter."""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        cutoff = cumulative_probs > top_p
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        
        filtered_probs = sorted_probs.masked_fill(cutoff, 0.0)
        probs = torch.zeros_like(probs)
        probs[sorted_indices] = filtered_probs
        
        return probs

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_p=0.9):
        """Generate text using the model."""
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                probs = self.top_p_filtering(probs, top_p)

            # Re-normalize and sample
            if probs.sum() == 0:
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

        return input_ids


# =============================================================================
# LOSS FUNCTION AND OPTIMIZER
# =============================================================================

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling."""
    logits = rearrange(logits, "b s v -> (b s) v")
    targets = rearrange(targets, "b s -> (b s)")

    # Numerically stable computation
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log() + m

    logit_true = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_p_true = logit_true - log_sum_exp.squeeze(-1)

    return (-log_p_true).mean()


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                t = state["t"] + 1

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # Update parameters
                with torch.no_grad():
                    if wd != 0:
                        p.data -= lr * wd * p.data  # Decoupled weight decay
                    p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                state["t"] = t
                
        return loss


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def gradient_clipping(parameters, max_norm, epsilon=1e-6):
    """Apply gradient clipping to prevent exploding gradients."""
    grads = [p.grad for p in parameters if p.grad is not None and p.requires_grad]
    
    if not grads:
        return
        
    flat_grads = torch.cat([g.view(-1) for g in grads])
    grad_norm = torch.norm(flat_grads, p=2)

    if grad_norm > max_norm:
        scale = max_norm / (grad_norm + epsilon)
        for p in parameters:
            if p.grad is not None and p.requires_grad:
                p.grad.mul_(scale)


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """Load random batches of training data."""
    n = len(x)
    starts = np.random.randint(0, n - context_length, size=batch_size)

    inputs = np.stack([x[s:s + context_length] for s in starts])
    targets = np.stack([x[s + 1:s + 1 + context_length] for s in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model, optimizer, iteration, filepath):
    """Save model checkpoint."""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


def evaluate_model(model, val_arr, batch_size, context_length, device):
    """Evaluate model on validation data."""
    model.eval()
    with torch.no_grad():
        inputs, targets = data_loading(val_arr, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
    model.train()
    return loss.item()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_llm(train_path: str, val_path: str, batch_size: int, max_steps: int,
              ckpt_path: str, tokenizer: Tokenizer, d_model: int, num_heads: int,
              d_ff: int, max_seq_len: int, theta: float, vocab_size: int,
              context_length: int, num_layers: int, device=None, dtype=None,
              clip_grad: float = 1.0, resume: Optional[str] = None,
              eval_every: int = 100, lr: float = 1e-3):
    """Complete training loop for the language model."""
    
    start_time = time.time()

    # Load data
    train_arr = np.load(train_path, mmap_mode="r")
    val_arr = np.load(val_path, mmap_mode="r")

    # Initialize model
    model = TransformerLanguageModel(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len,
        theta=theta, vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, device=device, dtype=dtype
    )

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # Resume from checkpoint if specified
    if resume:
        iteration = load_checkpoint(resume, model, optimizer)
    else:
        iteration = 0

    # Training loop
    for step in range(iteration, max_steps):
        optimizer.zero_grad()
        model = model.to(device)

        # Get batch
        inputs, targets = data_loading(train_arr, batch_size, context_length, device)

        # Forward pass
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        # Backward pass
        loss.backward()
        gradient_clipping(model.parameters(), clip_grad)
        optimizer.step()
        iteration += 1

        # Evaluation
        val_loss = None
        if iteration % eval_every == 0:
            val_loss = evaluate_model(model, val_arr, batch_size, context_length, device)
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Step {iteration}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

        # Logging
        wall_time = time.time() - start_time
        if step % 10 == 0:  # Print less frequently
            print(f"Step {step}: Train Loss = {loss.item():.4f}")

        # Generate sample text occasionally
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                start_tokens = tokenizer.encode(" ")
                input_ids = torch.tensor([start_tokens], device=device)
                generated_ids = model.generate(input_ids, max_new_tokens=50)
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"Generated: {generated_text}")
            model.train()

        # Log to wandb
        log_dict = {
            "step": step,
            "train_loss": loss.item(),
            "wall_time": wall_time,
        }
        if val_loss is not None:
            log_dict["val_loss"] = val_loss
        wandb.log(log_dict)


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

def sweep_run():
    """Function for hyperparameter sweeps with wandb."""
    wandb.init()
    config = wandb.config
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    train_llm(
        train_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
        val_path="../data/TinyStoriesV2-GPT4-valid-tok-10pct.npy",
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        ckpt_path="../checkpoints/model.pt",
        tokenizer=tokenizer,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.context_length,
        theta=10000,
        vocab_size=vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        device=device,
        dtype=None,
        clip_grad=1.0,
        resume=None,
        eval_every=config.eval_every,
        lr=config.lr,
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize W&B
    wandb.init(
        project="cs336-transformer",
        name="Clean Transformer Training",
        config={
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 1024,
            "num_layers": 4,
            "lr": 1e-3,
            "batch_size": 16,
            "context_length": 128,
            "eval_every": 100,
        }
    )

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])
    print(f"Tokenizer vocab length: {len(tokenizer.vocab)}")

    # Training
    train_llm(
        train_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
        val_path="../data/TinyStoriesV2-GPT4-valid-tok-10pct.npy",
        batch_size=16,
        max_steps=1000,
        ckpt_path="../checkpoints/model.pt",
        tokenizer=tokenizer,
        eval_every=100,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        max_seq_len=256,
        theta=10000,
        vocab_size=len(tokenizer.vocab),
        context_length=128,
        num_layers=4,
        device=device,
        lr=1e-3,
    ) 