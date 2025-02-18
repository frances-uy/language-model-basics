#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import regex as re
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    # Using PyTorch's implementation
    ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, d_model)
    )
    
    # Load weights
    ffn[0].weight.data = weights['w1.weight']
    ffn[2].weight.data = weights['w2.weight']
    
    return ffn(in_features)

def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    # Using PyTorch's scaled dot-product attention
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    
    if pdrop is not None:
        attn = F.dropout(attn, p=pdrop)
        
    return torch.matmul(attn, V)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    # Using PyTorch's MultiheadAttention
    mha = nn.MultiheadAttention(d_model, num_heads, dropout=attn_pdrop, batch_first=True)
    
    # Convert individual head weights to combined weights
    head_dim = d_model // num_heads
    
    # Combine Q, K, V projection weights
    q_weights = torch.stack([weights[f'q_heads.{i}.weight'] for i in range(num_heads)])
    k_weights = torch.stack([weights[f'k_heads.{i}.weight'] for i in range(num_heads)])
    v_weights = torch.stack([weights[f'v_heads.{i}.weight'] for i in range(num_heads)])
    
    mha.in_proj_weight.data = torch.cat([
        q_weights.reshape(-1, d_model),
        k_weights.reshape(-1, d_model),
        v_weights.reshape(-1, d_model)
    ])
    
    mha.out_proj.weight.data = weights['output_proj.weight']
    
    output, _ = mha(in_features, in_features, in_features)
    return output

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    # Using PyTorch's TransformerEncoderLayer
    transformer_block = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_ff,
        dropout=residual_pdrop,
        activation='gelu',
        batch_first=True,
        norm_first=True  # Pre-norm architecture
    )
    
    # Load weights
    transformer_block.self_attn.in_proj_weight.data = weights['attn.q_proj.weight']
    transformer_block.self_attn.out_proj.weight.data = weights['attn.output_proj.weight']
    transformer_block.linear1.weight.data = weights['ffn.w1.weight']
    transformer_block.linear2.weight.data = weights['ffn.w2.weight']
    transformer_block.norm1.weight.data = weights['ln1.weight']
    transformer_block.norm2.weight.data = weights['ln2.weight']
    
    return transformer_block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    # Using GPT2-like model from Hugging Face
    config = {
        "vocab_size": vocab_size,
        "n_positions": context_length,
        "n_embd": d_model,
        "n_layer": num_layers,
        "n_head": num_heads,
        "n_inner": d_ff,
        "attn_pdrop": attn_pdrop,
        "resid_pdrop": residual_pdrop,
    }
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
    
    # Load weights
    model.transformer.wte.weight.data = weights['token_embeddings.weight']
    model.transformer.wpe.weight.data = weights['position_embeddings.weight']
    
    for i in range(num_layers):
        layer = model.transformer.h[i]
        prefix = f'layers.{i}.'
        
        # Load attention weights
        layer.attn.c_attn.weight.data = weights[prefix + 'attn.q_proj.weight']
        layer.attn.c_proj.weight.data = weights[prefix + 'attn.output_proj.weight']
        
        # Load FFN weights
        layer.mlp.c_fc.weight.data = weights[prefix + 'ffn.w1.weight']
        layer.mlp.c_proj.weight.data = weights[prefix + 'ffn.w2.weight']
        
        # Load normalization weights
        layer.ln_1.weight.data = weights[prefix + 'ln1.weight']
        layer.ln_2.weight.data = weights[prefix + 'ln2.weight']
    
    model.transformer.ln_f.weight.data = weights['ln_final.weight']
    model.lm_head.weight.data = weights['lm_head.weight']
    
    outputs = model(in_indices)
    return outputs.logits

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    # Implementation of RMSNorm
    variance = in_features.pow(2).mean(-1, keepdim=True)
    in_features = in_features * torch.rsqrt(variance + eps)
    return in_features * weights['weight']

def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    return F.gelu(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Simple random sampling implementation
    total_length = len(dataset) - context_length
    start_indices = torch.randint(0, total_length, (batch_size,))
    
    x = torch.stack([
        torch.from_numpy(dataset[idx:idx + context_length]) 
        for idx in start_indices
    ]).to(device)
    
    y = torch.stack([
        torch.from_numpy(dataset[idx + 1:idx + context_length + 1])
        for idx in start_indices
    ]).to(device)
    
    return x, y

def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    return F.softmax(in_features, dim=dim)

def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    return F.cross_entropy(inputs, targets)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)

def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    return torch.optim.AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        # Linear warmup
        return max_learning_rate * (it / warmup_iters)
    elif it > cosine_cycle_iters:
        # After cycle, return min learning rate
        return min_learning_rate
    else:
        # Cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    # Using Hugging Face's GPT2Tokenizer as base
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Override vocab and merges
    tokenizer.vocab = {bytes.decode('utf-8'): idx for idx, bytes in vocab.items()}
    tokenizer.merges = [" ".join(map(lambda x: x.decode('utf-8'), merge)) for merge in merges]
    
    if special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    return tokenizer

import os
from tokenizers import ByteLevelBPETokenizer
from typing import List, Tuple, Dict

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], **kwargs) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer using Hugging Face's `ByteLevelBPETokenizer`.

    Args:
        input_path (str): Path to the text file for training the tokenizer.
        vocab_size (int): Maximum vocabulary size (includes initial byte vocab, merges, and special tokens).
        special_tokens (List[str]): List of additional special tokens.

    Returns:
        vocab (Dict[int, bytes]): Mapping from token ID (int) to token bytes.
        merges (List[Tuple[bytes, bytes]]): Ordered list of BPE merges.
    """

    # Define output directory
    output_dir = "bpe_tokenizer"

    # Initialize a ByteLevelBPETokenizer with add_prefix_space=True for GPT-2 compatibility
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    # Convert input_path to string if it's a PosixPath
    input_path = str(input_path)

    # Train the tokenizer
    tokenizer.train(
        files=[input_path],
        vocab_size=vocab_size,
        min_frequency=2,  # Prevents rare merges
        special_tokens=special_tokens
    )

    # Ensure the directory exists before saving
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer.save_model(output_dir)

    # **Verify that merges.txt and vocab.json exist**
    merges_path = os.path.join(output_dir, "merges.txt")
    vocab_path = os.path.join(output_dir, "vocab.json")

    if not os.path.exists(merges_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer files were not saved: {merges_path} or {vocab_path} missing.")

    # Load trained tokenizer
    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

    # Extract vocabulary (ID -> byte representation)
    vocab = {idx: bytes(token, encoding='utf-8') for token, idx in tokenizer.get_vocab().items()}

    # Fix: Read merges.txt manually while preserving order
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first line which contains metadata
            token1, token2 = line.strip().split()
            merges.append((token1.encode('utf-8'), token2.encode('utf-8')))

    return vocab, merges