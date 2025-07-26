import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange
from tokenizer import Tokenizer
import wandb
import time

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        std_sq = 2/(out_features + in_features)
        a = -3 * std_sq**0.5
        b = 3 * std_sq**0.5
        # torch.nn.init.trunc_normal_(self.weight, 0, std_sq, a=a, b=b)
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
            torch.empty(
                (self.num_embeddings, self.embedding_dim),
                **factory_kwargs
            )
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
        self.rms_weights = Parameter(
            torch.ones(
                self.d_model,
                **factory_kwargs
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = ( x32 / rms ) * self.rms_weights

        return y.to(orig_dtype)
        
        


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        if d_ff is not None:
            self.d_ff = d_ff
        else:
            raw_dff = (8 * d_model) / 3
            d_ff = math.ceil(raw_dff / 64) * 64

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x:torch.Tensor):
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        a = self.w1(x32)
        b = self.w3(x32)

        gated = F.silu(a) * b
        out = self.w2(gated)

        return out.to(orig_dtype)
    


class RopeEmbeddings(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        inv_freq = inv_freq.to(device)

        positions = torch.arange(max_seq_len).unsqueeze(1)

        angles = positions * inv_freq.unsqueeze(0)

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    
        cos_pos = self.cos[token_positions]  # shape: (..., seq_len, d_k//2)
        sin_pos = self.sin[token_positions]


        x1 = x[..., ::2]  # even indices
        x2 = x[..., 1::2] # odd indices

        rotated_even = x1 * cos_pos - x2 * sin_pos
        rotated_odd  = x1 * sin_pos + x2 * cos_pos

        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)  # (..., seq_len, d_k//2, 2)
        rotated = rotated.flatten(-2)  # (..., seq_len, d_k)

        return rotated




def softmax(x: torch.Tensor, dimension: int):
    expx = torch.exp(x - x.max(dim=dimension, keepdim=True).values)
    exp_sum = expx.sum(dim=dimension, keepdim=True)
    return expx/exp_sum



def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V:torch.Tensor, mask: Optional[torch.tensor] = None):
    scores = torch.einsum("...qd, ...kd->...qk", Q, K)
    scores = scores/math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(mask==False, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.einsum("...qk,...kd->...qd", attn_weights, V)


    return output



class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, theta =None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads  # typically same as d_k

        if theta is not None:
            self.rope = RopeEmbeddings(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_weights = nn.Parameter(
            torch.empty(self.d_model, self.d_model, **factory_kwargs)
        )
        self.k_weights = nn.Parameter(
            torch.empty(self.d_model, self.d_model, **factory_kwargs)
        )
        self.v_weights = nn.Parameter(
            torch.empty(self.d_model, self.d_model, **factory_kwargs)
        )
        self.o_weights = nn.Parameter(
            torch.empty(self.d_model, self.d_model, **factory_kwargs)  # standard [proj to d_model]
        )

        # Optional: better initialization than random junk
        torch.nn.init.xavier_uniform_(self.q_weights)
        torch.nn.init.xavier_uniform_(self.k_weights)
        torch.nn.init.xavier_uniform_(self.v_weights)
        torch.nn.init.xavier_uniform_(self.o_weights)




    def forward(self, in_features: torch.Tensor, use_rope = False):
        # print(f"Dimensions of infeatures: {in_features.shape}, dimensions of Q: {Q.shape}")
        Q = torch.einsum("b l d, k d -> b l k", in_features, self.q_weights)
        K = torch.einsum("b l d, k d -> b l k", in_features, self.k_weights)
        V = torch.einsum("b l d, k d -> b l k", in_features, self.v_weights)

        b, l, k = Q.shape
        h = self.num_heads

        Q = rearrange(Q, "b l (h k) -> b h l k", h=h)
        K = rearrange(K, "b l (h k) -> b h l k", h=h)
        V = rearrange(V, "b l (h k) -> b h l k", h=h)


        # Apply RoPE
        if use_rope:
            pos = torch.arange(l, device=in_features.device) 
            Q = self.rope(Q, pos)                     # expects [B, H, L, d_k]
            K = self.rope(K, pos)



        # Removed verbose debug prints
        seq_len = in_features.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0) 

        context = scaled_dot_product_attention(Q, K, V, mask)

        context = rearrange(context, "b h l k -> b l (h k)")

        out = torch.einsum("b l d, m d -> b l m", context, self.o_weights)

        return out



class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)   # w1,w3 gate -> w2 back to d_model

    def forward(self, x, token_pos=None):
        # --- MHSA ---
        h = self.ln1(x)
        # h = x
        attn_out = self.attn(h, use_rope=True)  # apply RoPE inside attn on Q,K only
        x = x + attn_out

        # --- FFN ---
        # h = self.ln2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        return x



class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 1) Token embedding
        self.tok_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2) Stack of Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
                for _ in range(num_layers)
            ]
        )

        # 3) Final RMSNorm (pre-norm architecture needs this)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # 4) LM head (projects to vocab)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.context_length = context_length  # may be useful for asserts/generation

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, S) int64
        returns:  (B, S, vocab_size) logits
        """
        x = self.tok_emb(token_ids)  # (B, S, d_model)

        # RoPE needs positions; build once and pass down if your blocks use it.
        pos = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)

        for blk in self.blocks:
            x = blk(x, token_pos=pos)

        # x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)
        return logits


    def top_p_filtering(self, probs, top_p=0.9):
        # probs: (vocab_size,)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        # Create mask: keep the smallest set where cumsum >= top_p
        cutoff = cumulative_probs > top_p
        # Shift mask right to keep at least one token
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        filtered_probs = sorted_probs.masked_fill(cutoff, 0.0)
        # Scatter back to original indices
        probs = torch.zeros_like(probs)
        probs[sorted_indices] = filtered_probs
        return probs

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_p=0.9):
        for _ in range(max_new_tokens):
            logits = self(input_ids)  # (1, seq_len, vocab)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                probs = self.top_p_filtering(probs, top_p)

            # Re-normalize in case some probs are set to zero
            if probs.sum() == 0:
                # Fallback: uniform random
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

            # Optionally: stop early if <|endoftext|> (e.g. token 0)
            # if next_token.item() == tokenizer.byte2id("<|endoftext|>"):
            #     break

        return input_ids


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:


    logits = rearrange(logits, "b s v->(b s) v")
    targets = rearrange(targets, "b s->(b s)")

    # logits: (N, V), targets: (N,)
    m = logits.max(dim=-1, keepdim=True).values              # stabilize
    shifted = logits - m
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log() + m  # = logsumexp

    # log p_true = logit_true - logsumexp
    logit_true = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_p_true = logit_true - log_sum_exp.squeeze(-1)

    loss_ = (-log_p_true).mean()

    print(f"loss: {loss_}")

    return loss_




class AdamW(torch.optim.Optimizer):
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
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                t = state["t"] + 1

                # moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                with torch.no_grad():
                    # decoupled weight decay
                    if wd != 0:
                        p.data -= lr * wd * p.data
                    p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                state["t"] = t
        return loss



def learning_rate_schedule(t, alpha_min, alpha_max, t_w, t_c):
    if t < t_w:
        return (t/t_w)*alpha_max

    if t >= t_w and t <= t_c:
        return alpha_min + 0.5*(1 + math.cos((t-t_w)/(t_c-t_w)*math.pi))*(alpha_max - alpha_min)

    else:
        return alpha_min



def gradient_clipping(parameters, max_norm, epsilon=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None and p.requires_grad]

    # Flatten and concatenate all gradients
    flat_grads = torch.cat([g.view(-1) for g in grads])

    grad_norm = torch.norm(flat_grads, p=2)

    if grad_norm > max_norm:
        scale = max_norm / (grad_norm + epsilon)
        for p in parameters:
            if p.grad is not None and p.requires_grad:
                p.grad.mul_(scale)

    return parameters
    

import numpy as np


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(x)
    max_start = n - context_length - 1

    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + 1 + context_length] for s in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets



def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")  # add device handling if needed
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


def evaluate_model(model, val_arr, batch_size, context_length, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = data_loading(val_arr, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        return loss.item()

def train_llm(
        # Data Params
        train_path: str,
        val_path: str,
        batch_size: int,
        max_steps: int,
        ckpt_path: str,
        tokenizer: Tokenizer,
        


        # Model Params
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device=None,
        dtype=None,



        # Training Params
        clip_grad: float = 1.0,
        alpha_min: float = 1e-5,
        alpha_max: float = 3e-4,
        t_w: int = 2000,
        t_c: int = 100_000,
        resume: Optional[str] = None,
        eval_every: int = 100,
        lr: float = 1e-3,  

):
    

    start_time = time.time() 


    train_arr = np.load(train_path, mmap_mode="r")
    val_arr = np.load(val_path, mmap_mode="r")
    


    inputs, targets = data_loading(train_arr, batch_size, context_length, device)

    model = TransformerLanguageModel(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )

    
    
    inputs, targets = data_loading(train_arr, batch_size, context_length, device)


    # optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


    if resume:
        iteration = load_checkpoint(resume, model, optimizer)
    else:
        iteration = 0


    for step in range(iteration, max_steps):
        print(f"Step {step}")
        
        optimizer.zero_grad()

        model = model.to(device)


        inputs, targets = data_loading(train_arr, batch_size, context_length, device)

        logits = model(inputs)

        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), clip_grad)

        optimizer.step()
        iteration += 1
        
        # Initialize val_loss as None
        val_loss = None
        
        # Evaluation every eval_every steps
        if iteration % eval_every == 0:
            print(f"Evaluating at step {iteration}")
            val_loss = evaluate_model(model, val_arr, batch_size, context_length, device)
            print(f"Step {iteration}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
    
        # Print train loss every step
        wall_time = time.time() - start_time
        print(f"Step {step}: Train Loss = {loss.item():.4f}")

        # Generate text every step (optional - you might want to do this less frequently)
        model.eval()
        with torch.no_grad():
            start_tokens = tokenizer.encode(" ")
            input_ids = torch.tensor([start_tokens], device=device)
            generated_ids = model.generate(input_ids, max_new_tokens=50)
            print(f"Generated ids: {generated_ids}" )
            print(f"Generated ids shape: {generated_ids.shape}")
            print(f"Generated text: {tokenizer.decode(generated_ids[0].tolist())}")
        model.train()
        
        # Log to wandb - include val_loss when available
        log_dict = {
            "step": step,
            "train_loss": loss.item(),
            "wall_time": wall_time,
        }
        if val_loss is not None:
            log_dict["val_loss"] = val_loss
            
        wandb.log(log_dict)






def sweep_run():
    import torch
    import wandb
    import numpy as np
    from tokenizer import Tokenizer

    # ✅ FIRST: call wandb.init()
    wandb.init()

    # Now you can access wandb.config safely
    config = wandb.config
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    train_llm(
        train_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
        val_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
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
        alpha_min=1e-5,
        alpha_max=3e-4,
        t_w=2000,
        t_c=100_000,
        resume=None,
        eval_every=config.eval_every,
        lr=config.lr,
    )




if __name__ == "__main__":
    # Check MPS availability first
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])
    
    print(f"Tokenizer vocab length: {len(tokenizer.vocab)}")
        
    train_arr = np.load("../data/TinyStoriesV2-GPT4-train-tok-1pct.npy", mmap_mode="r")
    print(f"Total number of tokens in training set: {len(train_arr):,}")


    train_llm(
        train_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
        val_path="../data/TinyStoriesV2-GPT4-train-tok-1pct.npy",
        batch_size=16,       # Smaller batch for MPS
        max_steps=1000,      # Fewer steps for testing
        ckpt_path="../checkpoints/model.pt",
        tokenizer=tokenizer,
        eval_every=100,      # More frequent evaluation
        d_model=256,         # Smaller model
        num_heads=8,         # Fewer heads
        d_ff=1024,           # Smaller FFN
        max_seq_len=256,     # Shorter sequences
        theta=10000,
        vocab_size=len(tokenizer.vocab),  # Use actual tokenizer vocab size
        context_length=128,  # Much shorter context
        num_layers=4,        # Fewer layers
        device=device,       # 🚀 Auto-detect best device
    )






