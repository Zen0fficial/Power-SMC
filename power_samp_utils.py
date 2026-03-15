"""Utilities for autoregressive power sampling and prompt formatting.

This module provides helper routines used by benchmark scripts:
- token-level sampling wrappers
- MCMC-style power sampling loops
- prompt construction for different chat model families
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from constants import *
from grader_utils.parse_utils import parse_answer


class AutoregressiveSampler:
    """Thin wrapper around a causal LM for token-level next-token log-probs."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    @torch.no_grad()
    def next_token(self, prefix):
        """Return log-probabilities for the next token given a tokenized prefix."""
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = (
            torch_prefix
            if torch_prefix.size(1) <= self.block_size
            else torch_prefix[:, -self.block_size :]
        )
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


def normalize(dist):
    """Return normalized probabilities from logits-like scores."""
    probs = F.softmax(dist, dim=-1)
    return probs


def dist_product(logit_p, logit_q):
    """Combine two logit vectors additively (log-space product)."""
    return logit_p + logit_q


def dist_temp_scale(logit_p, temp):
    """Apply temperature scaling in logit space."""
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)


def naive_temp(p: AutoregressiveSampler, context, temp, seq_len):
    """Generate a continuation with temperature sampling and token log-probs."""
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (
        (1 / temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx))
        .view(-1)
        .tolist()
    )
    log_probs_norm = (
        torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()
    )

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


def max_swap(
    p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16
):
    """Greedy-acceptance variant of autoregressive MCMC power sampling."""
    c = len(context)
    print(f"Temp: {temp}")
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(
                p, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def mcmc_power_samp(
    p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16
):
    """Metropolis-Hastings autoregressive power sampler over generated suffixes."""
    c = len(context)
    print(f"alpha: {1 / temp}")
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(
                p, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    """Create a model-specific prompt string/chat template from a question."""
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    return format_str
