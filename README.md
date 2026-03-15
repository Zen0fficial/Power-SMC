# Power-SMC

Codebase for experiments related to:

**Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning**  
arXiv: https://arxiv.org/abs/2602.10273

---

## Overview

This repository focuses on **training-free reasoning-time sampling** methods for LLMs, including:

- Standard sampling baselines
- Power sampling via autoregressive MCMC
- Sequence-level SMC power sampling (memory-optimized implementation)

Benchmarks in this folder include:

- **MATH500**
- **GSM8K**
- **GPQA-Diamond**
- HumanEval helper graders/utilities

---

## Repository Scope (this folder)

This README describes the contents of the folder:

- [power-smc](power-smc)

Core scripts:

- [power-smc/power_samp_math.py](power-smc/power_samp_math.py) — MATH benchmark driver
- [power-smc/power_samp_gsm.py](power-smc/power_samp_gsm.py) — GSM8K benchmark driver
- [power-smc/power_samp_gpqa.py](power-smc/power_samp_gpqa.py) — GPQA benchmark driver
- [power-smc/power_samp_utils.py](power-smc/power_samp_utils.py) — shared generation/sampling helpers
- [power-smc/smc_samp_utils_noschedule_plus_halt_opt.py](power-smc/smc_samp_utils_noschedule_plus_halt_opt.py) — SMC implementation

Support files:

- [power-smc/constants.py](power-smc/constants.py) — prompt templates/constants
- [power-smc/grader_utils](power-smc/grader_utils) — grading and parsing utilities
- [power-smc/myjob.slurm](power-smc/myjob.slurm) — example Slurm job script

---

## Method Summary

The target sequence-level distribution is:

$$
\pi_\alpha(y\mid x) \propto p_\theta(y\mid x)^\alpha,\quad \alpha > 1
$$

This code compares practical samplers that approximate or target this sharpened distribution with different latency/quality trade-offs.

---

## Environment Setup

### 1) Python environment

Use Python 3.10+ (recommended), then install dependencies:

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `tqdm`
- `sympy`
- `pylatexenc`

Optional formatting/dev tools:

- `ruff`

### 2) Model access

The scripts load models from Hugging Face Hub (e.g., Qwen, Phi, Tulu, Llama variants). Make sure your environment has access credentials if needed.

---

## Data

Expected local files/datasets include:

- `data/MATH500.json`
- `openai/gsm8k` via `datasets`
- `fingertap/GPQA-Diamond` via `datasets`

HumanEval helper files are under [power-smc/grader_utils](power-smc/grader_utils).

---

## Running Experiments

From [power-smc](power-smc):

- MATH:
  - `python -u power_samp_math.py --model qwen --temperature 0.25 --mcmc_steps 10`
- GSM8K:
  - `python -u power_samp_gsm.py --model qwen --temperature 0.25 --mcmc_steps 10`
- GPQA:
  - `python -u power_samp_gpqa.py --model qwen --temperature 0.25 --mcmc_steps 10`

---

## Key Configuration Knobs

Common options across scripts:

- `--model`
- `--temperature`
- `--mcmc_steps`
- `--cot`
- `--device`
- `--batch_idx`

SMC-specific settings are defined in `SMCSamplingConfig` in:

- [power-smc/smc_samp_utils_noschedule_plus_halt_opt.py](power-smc/smc_samp_utils_noschedule_plus_halt_opt.py)

---

## Reproducibility Notes

- Set fixed random seeds in your runtime launcher if deterministic comparisons are required.
- Keep model versions/checkpoints consistent across runs.
- Record GPU type and driver/CUDA versions alongside metrics.

---

## GitHub Readiness and Cleanup Policy

This repository was prepared with **non-functional cleanup only**:

- formatting
- comments/docstrings
- file organization/readability
- README and repository hygiene files

No logic, algorithm, or core experimental behavior was intentionally changed.

---

## Citation

If you use this code, please cite:

```bibtex
@article{azizi2026powersmc,
  title={Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning},
  author={Azizi, Seyedarmin and Potraghloo, Erfan Baghaei and Ahmadi, Minoo and Kundu, Souvik and Pedram, Massoud},
  journal={arXiv preprint arXiv:2602.10273},
  year={2026}
}
```

---

## License

Add your preferred repository license file (`LICENSE`) before public release.
