"""Utilities for bank-based Monte Carlo MH sampling of the power exponent alpha.

This module targets posterior-style inference over alpha given observed
question/answer pairs under the Power-SMC sequence-level power distribution.
Unlike the exchange sampler, the likelihood-ratio estimate is built from
reusable weighted particle banks anchored at the current alpha. Each prompt keeps
one Power-SMC particle bank, estimates log Z_{alpha'} / Z_alpha by importance
reweighting, and only refreshes low-ESS banks after accepted moves.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from constants import BASE, COT, PROMPT
from smc_samp_utils import SMCSamplingConfig, smc_power_sample_memopt


MODEL_NAME_MAP = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "phi_grpo": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def format_prompt(question, model, tokenizer, cot=True):
    """Create a model-specific prompt string/chat template from a question."""
    if model == "qwen":
        format_str = PROMPT + question
        format_str += COT if cot else BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        format_str += COT if cot else BASE

    elif model == "qwen_instruct":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi_grpo":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "tulu":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "llama":
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    else:
        raise ValueError(f"Unsupported model key '{model}'.")

    return format_str


@dataclass
class AlphaObservation:
    """A single observed question/answer pair and its exact base-model score."""

    question: str
    answer: str
    prompt_text: str
    prompt_ids: torch.Tensor
    observed_logprob: float
    observed_num_tokens: int


@dataclass
class ParticleBank:
    """One cached weighted particle bank for a single prompt."""

    anchor_alpha: float
    cum_logp: torch.Tensor
    log_normalized_w: torch.Tensor
    ended_with_eos: torch.Tensor


@dataclass
class AlphaMCMHConfig:
    """Configuration for bank-based Monte Carlo MH over alpha."""

    initial_alpha: float = 4.0
    num_steps: int = 20
    proposal_std: float = 0.2
    alpha_min: float = 1.0001
    alpha_max: float = 8.0
    prior_type: str = "flat"
    prior_mean: float = 4.0
    prior_std: float = 1.0

    max_new_tokens: int = 2048
    bank_particles: int = 64
    ess_threshold: float = 0.5
    refresh_ess_threshold: float = 0.5
    block_size: int = 64
    alpha_ramp_tokens: int = 400
    min_new_tokens: int = 0
    use_cow_cache: bool = True
    shared_prompt_cache: bool = True
    append_eos_to_observed: bool = True


def resolve_model_name(model: str, model_id: Optional[str] = None) -> str:
    """Resolve a short model key to a Hugging Face model id."""
    if model_id:
        return model_id
    if model not in MODEL_NAME_MAP:
        raise ValueError(f"Unknown model key '{model}'. Pass --model_id to override.")
    return MODEL_NAME_MAP[model]


def _choose_field(example: Dict[str, Any], explicit_key: Optional[str], fallbacks) -> str:
    """Select a field from an example, preferring an explicit key when provided."""
    if explicit_key is not None:
        if explicit_key not in example:
            raise KeyError(f"Missing field '{explicit_key}' in example: {example.keys()}")
        value = example[explicit_key]
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        return value

    for key in fallbacks:
        if key in example:
            value = example[key]
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            return value
    raise KeyError(f"Could not find any of {fallbacks} in example: {example.keys()}")


def load_qa_examples(
    path: str,
    limit: Optional[int] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load a batch of question/answer examples from JSON or JSONL."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if data_path.suffix == ".jsonl":
        raw_examples = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    raw_examples.append(json.loads(line))
    else:
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            raw_examples = payload
        elif isinstance(payload, dict) and isinstance(payload.get("examples"), list):
            raw_examples = payload["examples"]
        else:
            raise ValueError("Expected a JSON list or a dict with an 'examples' list.")

    if limit is not None:
        raw_examples = raw_examples[:limit]

    examples = []
    for example in raw_examples:
        question = _choose_field(example, question_key, ("question", "prompt", "input"))
        answer = _choose_field(
            example,
            answer_key,
            ("answer", "response", "completion", "output"),
        )
        examples.append({"question": question, "answer": answer})
    return examples


def _tokenize_prompt_and_completion(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    append_eos: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize a prompt and continuation, keeping the prompt as a strict prefix."""
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    full_ids = tokenizer(
        prompt_text + answer_text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    prompt_len = prompt_ids.numel()
    if full_ids.numel() < prompt_len or not torch.equal(full_ids[:prompt_len], prompt_ids):
        raise ValueError(
            "Prompt tokenization is not a strict prefix of prompt+answer tokenization. "
            "Try adding an explicit whitespace/newline before answers or adjusting the prompt format."
        )

    completion_ids = full_ids[prompt_len:]
    eos_id = tokenizer.eos_token_id
    if append_eos:
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is required when append_eos=True.")
        if completion_ids.numel() == 0 or int(completion_ids[-1].item()) != int(eos_id):
            completion_ids = torch.cat(
                [completion_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0
            )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
    }


@torch.no_grad()
def teacher_forced_logprob(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> float:
    """Compute log p_theta(completion | prompt) token-by-token with KV caching."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if completion_ids.dim() == 1:
        completion_ids = completion_ids.unsqueeze(0)
    if prompt_ids.size(0) != 1 or completion_ids.size(0) != 1:
        raise ValueError("teacher_forced_logprob expects a single prompt and a single completion.")

    if completion_ids.numel() == 0:
        return 0.0

    out = model(input_ids=prompt_ids, use_cache=True)
    past = getattr(out, "past_key_values", None)
    last_logits = out.logits[:, -1, :]

    total = 0.0
    for token_id in completion_ids[0]:
        logprobs = F.log_softmax(last_logits.float(), dim=-1)
        total += float(logprobs[0, int(token_id.item())].item())
        out = model(
            input_ids=token_id.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

    return total


def build_alpha_observations(
    model,
    tokenizer,
    examples: Iterable[Dict[str, str]],
    model_key: str,
    cot: bool = True,
    append_eos: bool = True,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> List[AlphaObservation]:
    """Precompute exact base-model log-probs for a batch of observed Q/A pairs."""
    if device is None:
        device = next(model.parameters()).device

    iterator = examples
    if progress:
        iterator = tqdm(list(examples), desc="Scoring observed answers")

    observations: List[AlphaObservation] = []
    for example in iterator:
        question = example["question"]
        answer = example["answer"]
        prompt_text = format_prompt(question, model_key, tokenizer, cot)
        encoded = _tokenize_prompt_and_completion(
            tokenizer,
            prompt_text,
            answer,
            append_eos=append_eos,
        )
        prompt_ids = encoded["prompt_ids"].to(device)
        completion_ids = encoded["completion_ids"].to(device)
        observed_logprob = teacher_forced_logprob(model, prompt_ids, completion_ids)
        observations.append(
            AlphaObservation(
                question=question,
                answer=answer,
                prompt_text=prompt_text,
                prompt_ids=prompt_ids.unsqueeze(0),
                observed_logprob=observed_logprob,
                observed_num_tokens=int(completion_ids.numel()),
            )
        )
    return observations


def _log_alpha_prior(alpha: float, cfg: AlphaMCMHConfig) -> float:
    """Evaluate the log prior over alpha."""
    if alpha <= cfg.alpha_min or alpha >= cfg.alpha_max:
        return -math.inf
    if cfg.prior_type == "flat":
        return 0.0
    if cfg.prior_type == "normal":
        z = (alpha - cfg.prior_mean) / cfg.prior_std
        return -0.5 * z * z - math.log(cfg.prior_std) - 0.5 * math.log(2.0 * math.pi)
    raise ValueError(f"Unsupported prior_type '{cfg.prior_type}'.")


def _propose_alpha(
    current_alpha: float,
    cfg: AlphaMCMHConfig,
    rng: np.random.Generator,
) -> float:
    """Random-walk proposal on log(alpha - alpha_min)."""
    shift = current_alpha - cfg.alpha_min
    if shift <= 0.0:
        raise ValueError("current_alpha must be strictly greater than alpha_min.")
    eta = math.log(shift)
    eta_prop = eta + float(rng.normal(0.0, cfg.proposal_std))
    return cfg.alpha_min + math.exp(eta_prop)


def _log_q_reverse_minus_forward(
    current_alpha: float,
    proposed_alpha: float,
    cfg: AlphaMCMHConfig,
) -> float:
    """Hastings correction for the shifted-log random walk."""
    return math.log(proposed_alpha - cfg.alpha_min) - math.log(
        current_alpha - cfg.alpha_min
    )


def _observed_logprob_sum(observations: List[AlphaObservation]) -> float:
    """Return the batch sum of observed base-model log-probabilities."""
    return float(sum(obs.observed_logprob for obs in observations))


def _effective_sample_size_from_log_weights(log_normalized_w: torch.Tensor) -> float:
    """Compute ESS from normalized log-weights."""
    w = torch.exp(log_normalized_w)
    return float(1.0 / torch.sum(w * w).item())


def _build_sampling_config(alpha: float, cfg: AlphaMCMHConfig) -> SMCSamplingConfig:
    """Construct an SMC config for building or refreshing particle banks."""
    return SMCSamplingConfig(
        max_new_tokens=cfg.max_new_tokens,
        alpha=alpha,
        n_particles=cfg.bank_particles,
        ess_threshold=cfg.ess_threshold,
        temperature=1.0 / alpha,
        block_size=cfg.block_size,
        force_eos_after_done=True,
        alpha_ramp_tokens=cfg.alpha_ramp_tokens,
        min_new_tokens=cfg.min_new_tokens,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        min_tokens_to_keep=1,
        penalize_prompt=False,
        hard_truncation=True,
        soft_truncation_value=-50.0,
        stop_on_boxed=False,
        boxed_check_window_tokens=256,
        use_cow_cache=cfg.use_cow_cache,
        shared_prompt_cache=cfg.shared_prompt_cache,
    )


def _make_particle_bank(
    alpha: float,
    smc_out: Dict[str, Any],
    eos_id: Optional[int],
) -> ParticleBank:
    """Build a lightweight particle bank from one SMC run."""
    log_normalized_w = torch.log(smc_out["w"].detach().clamp_min(1e-32)).clone()
    cum_logp = smc_out["cum_logp"].detach().clone()
    ended_with_eos = torch.zeros_like(cum_logp, dtype=torch.bool)
    if eos_id is not None:
        seqs = smc_out["sequences"]
        ended_with_eos = seqs[:, -1].detach().eq(int(eos_id)).clone()
    return ParticleBank(
        anchor_alpha=float(alpha),
        cum_logp=cum_logp,
        log_normalized_w=log_normalized_w,
        ended_with_eos=ended_with_eos,
    )


@torch.no_grad()
def _refresh_particle_bank(
    model,
    tokenizer,
    observation: AlphaObservation,
    alpha: float,
    cfg: AlphaMCMHConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Run the upstream Power-SMC sampler once and convert the result into a bank."""
    proposal_cfg = _build_sampling_config(alpha, cfg)
    smc_out = smc_power_sample_memopt(
        model,
        tokenizer,
        observation.prompt_ids,
        proposal_cfg,
        generator=generator,
    )
    eos_id = tokenizer.eos_token_id
    bank = _make_particle_bank(alpha, smc_out, eos_id)
    num_truncated_particles = 0
    if eos_id is not None:
        num_truncated_particles = int((~bank.ended_with_eos).sum().item())
    return {
        "bank": bank,
        "num_truncated_particles": num_truncated_particles,
    }


@torch.no_grad()
def initialize_particle_banks(
    model,
    tokenizer,
    observations: List[AlphaObservation],
    alpha: float,
    cfg: AlphaMCMHConfig,
    generator: Optional[torch.Generator] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Build one initial particle bank per prompt at the current alpha."""
    iterator = observations
    if progress:
        iterator = tqdm(observations, desc="Initializing particle banks", leave=False)

    banks: List[ParticleBank] = []
    num_truncated_particles = 0
    for obs in iterator:
        refresh_info = _refresh_particle_bank(
            model,
            tokenizer,
            obs,
            alpha=alpha,
            cfg=cfg,
            generator=generator,
        )
        banks.append(refresh_info["bank"])
        num_truncated_particles += int(refresh_info["num_truncated_particles"])

    init_ess = [
        _effective_sample_size_from_log_weights(bank.log_normalized_w) for bank in banks
    ]
    return {
        "banks": banks,
        "num_truncated_particles": num_truncated_particles,
        "min_ess": min(init_ess) if init_ess else None,
        "mean_ess": float(sum(init_ess) / len(init_ess)) if init_ess else None,
    }


def _estimate_log_z_ratio_from_bank(
    bank: ParticleBank,
    alpha_new: float,
) -> Dict[str, Any]:
    """Estimate log Z_{alpha_new} / Z_{anchor_alpha} and the reweighted bank."""
    raw_log_w = bank.log_normalized_w + (float(alpha_new) - bank.anchor_alpha) * bank.cum_logp
    if not torch.isfinite(raw_log_w).any():
        return {
            "log_z_ratio": None,
            "bank": None,
            "ess": None,
        }

    log_z_ratio = torch.logsumexp(raw_log_w, dim=0)
    if not torch.isfinite(log_z_ratio):
        return {
            "log_z_ratio": None,
            "bank": None,
            "ess": None,
        }

    log_norm = raw_log_w - log_z_ratio
    if not torch.isfinite(log_norm).all():
        return {
            "log_z_ratio": None,
            "bank": None,
            "ess": None,
        }

    reweighted_bank = ParticleBank(
        anchor_alpha=float(alpha_new),
        cum_logp=bank.cum_logp,
        log_normalized_w=log_norm,
        ended_with_eos=bank.ended_with_eos,
    )
    return {
        "log_z_ratio": float(log_z_ratio.item()),
        "bank": reweighted_bank,
        "ess": _effective_sample_size_from_log_weights(log_norm),
    }


@torch.no_grad()
def estimate_mcmh_log_likelihood_ratio(
    observations: List[AlphaObservation],
    current_banks: List[ParticleBank],
    current_alpha: float,
    proposed_alpha: float,
    cfg: AlphaMCMHConfig,
    observed_sum: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate the batch log-likelihood ratio with reusable particle banks."""
    if len(current_banks) != len(observations):
        raise ValueError("current_banks must align one-to-one with observations.")
    if observed_sum is None:
        observed_sum = _observed_logprob_sum(observations)
    for bank in current_banks:
        if not math.isclose(bank.anchor_alpha, current_alpha, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("All current_banks must be anchored at the current alpha.")

    log_z_ratio_estimate = 0.0
    promptwise_log_z_ratio: List[Optional[float]] = []
    promptwise_ess: List[Optional[float]] = []
    proposed_banks: List[Optional[ParticleBank]] = []
    valid = True

    refresh_threshold = float(cfg.refresh_ess_threshold) * float(cfg.bank_particles)
    num_low_ess_banks = 0
    num_high_ess_banks = 0

    for bank in current_banks:
        estimate = _estimate_log_z_ratio_from_bank(bank, proposed_alpha)
        log_ratio_i = estimate["log_z_ratio"]
        ess_i = estimate["ess"]
        promptwise_log_z_ratio.append(log_ratio_i)
        promptwise_ess.append(ess_i)
        proposed_banks.append(estimate["bank"])

        if log_ratio_i is None or ess_i is None:
            valid = False
            num_low_ess_banks += 1
            continue

        log_z_ratio_estimate += float(log_ratio_i)
        if ess_i < refresh_threshold:
            num_low_ess_banks += 1
        else:
            num_high_ess_banks += 1

    log_like_ratio = -math.inf
    if valid:
        log_like_ratio = (proposed_alpha - current_alpha) * observed_sum - log_z_ratio_estimate

    finite_ess = [ess for ess in promptwise_ess if ess is not None]
    return {
        "observed_sum": observed_sum,
        "log_like_ratio": log_like_ratio,
        "log_z_ratio_estimate": log_z_ratio_estimate if valid else None,
        "promptwise_log_z_ratio": promptwise_log_z_ratio,
        "promptwise_ess": promptwise_ess,
        "min_prompt_ess": min(finite_ess) if finite_ess else None,
        "mean_prompt_ess": (
            float(sum(finite_ess) / len(finite_ess)) if finite_ess else None
        ),
        "num_high_ess_banks": num_high_ess_banks,
        "num_low_ess_banks": num_low_ess_banks,
        "proposed_banks": proposed_banks,
    }


@torch.no_grad()
def update_particle_banks_after_accept(
    model,
    tokenizer,
    observations: List[AlphaObservation],
    proposed_banks: List[Optional[ParticleBank]],
    promptwise_ess: List[Optional[float]],
    alpha: float,
    cfg: AlphaMCMHConfig,
    generator: Optional[torch.Generator] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Refresh only low-ESS banks after an accepted alpha move."""
    if len(proposed_banks) != len(observations):
        raise ValueError("proposed_banks must align one-to-one with observations.")
    if len(promptwise_ess) != len(observations):
        raise ValueError("promptwise_ess must align one-to-one with observations.")

    refresh_threshold = float(cfg.refresh_ess_threshold) * float(cfg.bank_particles)

    iterator = zip(observations, proposed_banks, promptwise_ess)
    if progress:
        iterator = tqdm(
            list(iterator),
            desc="Updating particle banks",
            leave=False,
        )

    next_banks: List[ParticleBank] = []
    num_reused_banks = 0
    num_refreshed_banks = 0
    num_truncated_refreshed_particles = 0

    for obs, bank, ess in iterator:
        should_refresh = bank is None or ess is None or ess < refresh_threshold
        if not should_refresh:
            next_banks.append(bank)
            num_reused_banks += 1
            continue

        refresh_info = _refresh_particle_bank(
            model,
            tokenizer,
            obs,
            alpha=alpha,
            cfg=cfg,
            generator=generator,
        )
        next_banks.append(refresh_info["bank"])
        num_refreshed_banks += 1
        num_truncated_refreshed_particles += int(refresh_info["num_truncated_particles"])

    return {
        "banks": next_banks,
        "num_reused_banks": num_reused_banks,
        "num_refreshed_banks": num_refreshed_banks,
        "num_truncated_refreshed_particles": num_truncated_refreshed_particles,
    }


def run_alpha_mcmh_sampler(
    model,
    tokenizer,
    observations: List[AlphaObservation],
    cfg: AlphaMCMHConfig,
    seed: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run approximate bank-based Monte Carlo MH over alpha."""
    if not observations:
        raise ValueError("At least one observation is required to sample alpha.")
    if cfg.initial_alpha <= cfg.alpha_min or cfg.initial_alpha >= cfg.alpha_max:
        raise ValueError("initial_alpha must lie strictly inside (alpha_min, alpha_max).")
    if cfg.bank_particles <= 0:
        raise ValueError("bank_particles must be positive.")
    if cfg.refresh_ess_threshold < 0.0:
        raise ValueError("refresh_ess_threshold must be non-negative.")

    rng = np.random.default_rng(seed)
    torch_generator = None
    if seed is not None:
        device = observations[0].prompt_ids.device
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(int(seed))

    current_alpha = float(cfg.initial_alpha)
    current_log_prior = _log_alpha_prior(current_alpha, cfg)
    if not math.isfinite(current_log_prior):
        raise ValueError("initial_alpha has zero prior density under the chosen prior.")

    observed_sum = _observed_logprob_sum(observations)
    init_info = initialize_particle_banks(
        model,
        tokenizer,
        observations,
        alpha=current_alpha,
        cfg=cfg,
        generator=torch_generator,
        progress=progress,
    )
    current_banks = init_info["banks"]

    trace = []
    accepts = 0

    step_iter = range(cfg.num_steps)
    if progress:
        step_iter = tqdm(step_iter, desc="Sampling alpha (MCMH)")

    for step_idx in step_iter:
        alpha_before_step = current_alpha
        proposed_alpha = _propose_alpha(current_alpha, cfg, rng)
        proposed_log_prior = _log_alpha_prior(proposed_alpha, cfg)
        proposal_delta_alpha = proposed_alpha - alpha_before_step

        if not math.isfinite(proposed_log_prior):
            trace.append(
                {
                    "step": step_idx,
                    "alpha_before_step": alpha_before_step,
                    "current_alpha": current_alpha,
                    "proposed_alpha": proposed_alpha,
                    "accepted": False,
                    "log_accept_ratio": -math.inf,
                    "log_like_ratio": -math.inf,
                    "observed_sum": observed_sum,
                    "log_z_ratio_estimate": None,
                    "promptwise_log_z_ratio": None,
                    "promptwise_ess": None,
                    "min_prompt_ess": None,
                    "mean_prompt_ess": None,
                    "num_high_ess_banks": None,
                    "num_low_ess_banks": None,
                    "num_reused_banks": 0,
                    "num_refreshed_banks": 0,
                    "num_truncated_refreshed_particles": 0,
                    "proposal_delta_alpha": proposal_delta_alpha,
                    "abs_proposal_delta_alpha": abs(proposal_delta_alpha),
                }
            )
            continue

        ratio_info = estimate_mcmh_log_likelihood_ratio(
            observations,
            current_banks=current_banks,
            current_alpha=current_alpha,
            proposed_alpha=proposed_alpha,
            cfg=cfg,
            observed_sum=observed_sum,
        )
        log_accept_ratio = (
            ratio_info["log_like_ratio"]
            + proposed_log_prior
            - current_log_prior
            + _log_q_reverse_minus_forward(current_alpha, proposed_alpha, cfg)
        )

        accepted = math.log(float(rng.uniform())) < log_accept_ratio
        num_reused_banks = 0
        num_refreshed_banks = 0
        num_truncated_refreshed_particles = 0

        if accepted:
            update_info = update_particle_banks_after_accept(
                model,
                tokenizer,
                observations,
                proposed_banks=ratio_info["proposed_banks"],
                promptwise_ess=ratio_info["promptwise_ess"],
                alpha=proposed_alpha,
                cfg=cfg,
                generator=torch_generator,
                progress=False,
            )
            current_banks = update_info["banks"]
            current_alpha = proposed_alpha
            current_log_prior = proposed_log_prior
            accepts += 1
            num_reused_banks = update_info["num_reused_banks"]
            num_refreshed_banks = update_info["num_refreshed_banks"]
            num_truncated_refreshed_particles = update_info[
                "num_truncated_refreshed_particles"
            ]

        trace.append(
            {
                "step": step_idx,
                "alpha_before_step": alpha_before_step,
                "current_alpha": current_alpha,
                "proposed_alpha": proposed_alpha,
                "accepted": accepted,
                "log_accept_ratio": log_accept_ratio,
                "log_like_ratio": ratio_info["log_like_ratio"],
                "observed_sum": ratio_info["observed_sum"],
                "log_z_ratio_estimate": ratio_info["log_z_ratio_estimate"],
                "promptwise_log_z_ratio": ratio_info["promptwise_log_z_ratio"],
                "promptwise_ess": ratio_info["promptwise_ess"],
                "min_prompt_ess": ratio_info["min_prompt_ess"],
                "mean_prompt_ess": ratio_info["mean_prompt_ess"],
                "num_high_ess_banks": ratio_info["num_high_ess_banks"],
                "num_low_ess_banks": ratio_info["num_low_ess_banks"],
                "num_reused_banks": num_reused_banks,
                "num_refreshed_banks": num_refreshed_banks,
                "num_truncated_refreshed_particles": num_truncated_refreshed_particles,
                "proposal_delta_alpha": proposal_delta_alpha,
                "abs_proposal_delta_alpha": abs(proposal_delta_alpha),
            }
        )

        if progress:
            step_iter.set_postfix(
                alpha=f"{current_alpha:.3f}",
                acc=f"{accepts / (step_idx + 1):.2f}",
                low_ess=ratio_info["num_low_ess_banks"],
            )

    return {
        "config": asdict(cfg),
        "num_observations": len(observations),
        "initial_alpha": cfg.initial_alpha,
        "final_alpha": current_alpha,
        "acceptance_rate": accepts / max(len(trace), 1),
        "observed_logprob_sum": observed_sum,
        "initial_bank_num_truncated_particles": init_info["num_truncated_particles"],
        "initial_bank_min_ess": init_info["min_ess"],
        "initial_bank_mean_ess": init_info["mean_ess"],
        "trace": trace,
    }
