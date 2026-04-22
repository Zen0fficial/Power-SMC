"""Run bank-based Monte Carlo MH alpha sampling on a batch of question/answer pairs."""

import argparse
import json
import os

import torch
import transformers

from alpha_mcmh_utils import (
    AlphaMCMHConfig,
    build_alpha_observations,
    load_qa_examples,
    resolve_model_name,
    run_alpha_mcmh_sampler,
)


def is_local_model_path(model_str: str) -> bool:
    """Return True when model_str points to an existing local directory."""
    return os.path.isdir(model_str)


def enable_hf_offline_mode() -> None:
    """Force Hugging Face libraries into offline mode for the current process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sample alpha with reusable Power-SMC particle banks for a batch of observed question/answer pairs."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/MATH500.json",
        help="Path to a JSON or JSONL file containing question/answer examples.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/alpha_mcmh_sampler.json",
        help="Where to write the sampler trace as JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen_math",
        choices=[
            "qwen",
            "qwen_instruct",
            "qwen_math",
            "phi",
            "tulu",
            "qwen_math_grpo",
            "phi_grpo",
            "llama",
        ],
        help="Short model key used for prompt formatting and default HF model resolution.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional Hugging Face model id override.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default=None,
        help="Optional question field name in the dataset.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default=None,
        help="Optional answer field name in the dataset.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=8,
        help="Number of examples from the dataset to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model loading and scoring.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load tokenizer/model files from local disk.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face libraries into offline mode for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the alpha chain and Power-SMC bank refreshes.",
    )
    parser.add_argument(
        "--cot",
        dest="cot",
        action="store_true",
        help="Use the chain-of-thought prompt variant.",
    )
    parser.add_argument(
        "--no-cot",
        dest="cot",
        action="store_false",
        help="Use the shorter direct-answer prompt variant.",
    )
    parser.set_defaults(cot=True)

    parser.add_argument("--initial_alpha", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--proposal_std", type=float, default=0.2)
    parser.add_argument("--alpha_min", type=float, default=1.0001)
    parser.add_argument("--alpha_max", type=float, default=8.0)
    parser.add_argument(
        "--prior_type",
        type=str,
        default="flat",
        choices=["flat", "normal"],
    )
    parser.add_argument("--prior_mean", type=float, default=4.0)
    parser.add_argument("--prior_std", type=float, default=1.0)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--bank_particles",
        "--n_particles",
        dest="bank_particles",
        type=int,
        default=64,
        help="Number of Power-SMC particles per prompt bank.",
    )
    parser.add_argument(
        "--ess_threshold",
        type=float,
        default=0.5,
        help="Internal ESS threshold used inside each Power-SMC refresh.",
    )
    parser.add_argument(
        "--refresh_ess_threshold",
        type=float,
        default=0.5,
        help="Refresh accepted banks whose reweighted ESS falls below this fraction of bank_particles.",
    )
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--alpha_ramp_tokens", type=int, default=400)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument(
        "--append_eos_to_observed",
        dest="append_eos_to_observed",
        action="store_true",
        help="Append EOS when scoring observed answers.",
    )
    parser.add_argument(
        "--no-append_eos_to_observed",
        dest="append_eos_to_observed",
        action="store_false",
        help="Do not append EOS when scoring observed answers.",
    )
    parser.set_defaults(append_eos_to_observed=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    model_name = resolve_model_name(args.model, args.model_id)
    if args.offline:
        enable_hf_offline_mode()
    examples = load_qa_examples(
        args.data_path,
        limit=args.num_examples,
        question_key=args.question_key,
        answer_key=args.answer_key,
    )

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    local_files_only = args.local_files_only or is_local_model_path(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=False, local_files_only=local_files_only
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
        local_files_only=local_files_only,
    ).to(args.device)
    model.eval()

    observations = build_alpha_observations(
        model,
        tokenizer,
        examples,
        model_key=args.model,
        cot=args.cot,
        append_eos=args.append_eos_to_observed,
        progress=True,
    )

    cfg = AlphaMCMHConfig(
        initial_alpha=args.initial_alpha,
        num_steps=args.num_steps,
        proposal_std=args.proposal_std,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        prior_type=args.prior_type,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        max_new_tokens=args.max_new_tokens,
        bank_particles=args.bank_particles,
        ess_threshold=args.ess_threshold,
        refresh_ess_threshold=args.refresh_ess_threshold,
        block_size=args.block_size,
        alpha_ramp_tokens=args.alpha_ramp_tokens,
        min_new_tokens=args.min_new_tokens,
        append_eos_to_observed=args.append_eos_to_observed,
    )
    result = run_alpha_mcmh_sampler(
        model,
        tokenizer,
        observations,
        cfg,
        seed=args.seed,
        progress=True,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved alpha trace to {args.save_path}")
    print(f"Final alpha: {result['final_alpha']:.4f}")
    print(f"Acceptance rate: {result['acceptance_rate']:.3f}")


if __name__ == "__main__":
    main()
