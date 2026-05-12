# TTRL on Qwen2.5-Omni (audio + video)

Test-Time Reinforcement Learning extended to open-ended audio-visual QA on
Qwen2.5-Omni-3B, on top of [VERL](https://github.com/volcengine/verl).

Four orthogonal reward modes dispatched by a single env var (`TTRL_TASK_TYPE`):

| `TTRL_TASK_TYPE` | Reward source | Task shape |
|---|---|---|
| `math` / `video_qa` / *(unset)* | Majority-vote (string-counting) over answer-extracted rollouts | Closed-form / MCQ. **Original TTRL paper recipe.** |
| `open_ended_video` | BGE/Qwen3/MPNet embedding-medoid cosine-sim to gold | Open-ended free-text |
| `judge_open_ended` | **LLM-as-judge** — policy itself rates rollouts 0-10 against the BGE-medoid (no gold) | Open-ended free-text |
| `simple_cluster` / `evolrl_cluster` | **Cluster vote** — k-means on rollout embeddings → modal cluster wins | Open-ended, cluster-based |

All four are independent. The new modes degrade gracefully back to vanilla
TTRL for MCQ — see the env var matrix below.

## Quickstart

```bash
conda activate verl312
cd /path/to/TTRL
export PYTHONPATH="$PWD/verl:$PYTHONPATH"
```

## Run an ablation

Each `slurm/*.sh` script is a complete, runnable config. Submit with `sbatch slurm/<name>.sh`.

### 1. Vanilla TTRL (MCQ / closed-form math)

```bash
# AIME math (text-only Qwen2.5-Math-1.5B, BGE encoder, majority-vote on \boxed{})
sbatch slurm/aime.sh

# AIME with cluster-vote (binary), 1.5B math model
sbatch slurm/aime_simple_cluster.sh

# MMAU (audio MCQ)
sbatch slurm/mmau_omni.sh
```

### 2. Open-ended cluster vote on daily_omni (video + audio)

```bash
# Binary cluster vote (default). Single line to flip to continuous medoid below.
sbatch slurm/daily_omni_simple_cluster.sh

# Continuous-medoid reward (smooth [0,1] instead of {0,1}):
TTRL_CLUSTER_CONTINUOUS=1 sbatch slurm/daily_omni_simple_cluster.sh

# Different encoder (paraphrase-tuned MPNet, much smaller than Qwen3-4B):
TTRL_OE_ENCODER=mpnet sbatch slurm/daily_omni_simple_cluster.sh

# Combined:
TTRL_CLUSTER_CONTINUOUS=1 TTRL_OE_ENCODER=mpnet sbatch slurm/daily_omni_simple_cluster.sh

# EVOL-RL (banded reward + novelty + clip-higher + entropy reg)
sbatch slurm/daily_omni_evolrl_cluster.sh
```

### 3. LLM-as-judge on daily_omni (judge_v3 — vLLM + aux monitoring)

```bash
sbatch slurm/daily_omni_judge_v3.sh

# Legacy judge_v2 (HF rollout, no aux monitoring) for reproducing the +3.2 MMAU run:
sbatch slurm/daily_omni_judge_v2.sh
```

### 4. Open-ended embedding-medoid (no cluster, no judge)

```bash
sbatch slurm/daily_omni_open.sh
```

### 5. Text-only TTRL on Qwen2.5-3B base + UltraFeedback (instruction following)

Faster iteration loop — text-only, single A6 partition, ~3-5 min/step.
Tests whether cluster-vote TTRL can elicit instruction-following from a
base model with no labels.

```bash
# Convert UltraFeedback prompts into VERL JSON (one-time, ~30s)
python verl/scripts/build_ultrafeedback_ttrl.py \
  --src /data/sls/scratch/mvideet/datasets/UltraFeedback \
  --out-dir verl/data/UltraFeedback-TTRL \
  --train-n 4000 --test-n 500 --sanity-n 50

# Submit training (Qwen2.5-3B base + continuous cluster + Qwen3-4B encoder)
sbatch slurm/ultrafeedback_simple_cluster.sh
```

Eval target: AlpacaEval 2.0 length-controlled win-rate vs GPT-4-Turbo
(base ~5-15%, SFT ~30-40%, DPO ~40-55%).

## Env-var matrix

| Env var | Default | Effect | Affects vanilla TTRL? |
|---|---|---|---|
| `TTRL_TASK_TYPE` | `video_qa` | Dispatch: `math` / `video_qa` / `open_ended_video` / `judge_open_ended` / `simple_cluster` / `evolrl_cluster` | Selects path |
| `TTRL_OE_ENCODER` | `bge` | Embedding encoder: `bge` (BGE-small) / `qwen3` (Qwen3-Embedding-4B) / `mpnet` (paraphrase-mpnet-base-v2) | Only when embedding is used |
| `TTRL_OE_DEVICE` | auto | Force encoder to `cpu` or `cuda` | No |
| `TTRL_OE_MAX_LEN` | 512 (bge) / 1024 (qwen3) / 384 (mpnet) | Encoder max tokens | No |
| `TTRL_CLUSTER_CONTINUOUS` | `0` | `1` → continuous medoid reward `(cos+1)/2` instead of binary `{0,1}` | No (only `simple_cluster`) |
| `TTRL_CLUSTER_K_MIN` / `K_MAX` | `2` / `4` | k-means K range (auto-selected) | No |
| `TTRL_AUX_DETERMINISTIC` | `1` | Compute BLEU/ROUGE-L/EM/contains-EM per rollout vs gold, log as `train/aux_*` | No (only fires inside cluster/judge paths) |
| `TTRL_AUX_GPT_JUDGE` | `0` | Async GPT-4o-mini judge against gold, log as `train/aux_gpt_judge_mean`. ~$4.50 / 300-step run | No |
| `TTRL_AUX_GPT_MODEL` | `gpt-4o-mini-2024-07-18` | Override judge model | No |
| `TTRL_AUX_GPT_CONCURRENCY` | `8` | Parallel API calls | No |
| `TTRL_LOG_DROP_PATTERNS` | *(empty)* | Comma-separated list; metrics whose keys contain any pattern are dropped from W&B/Tensorboard (console keeps everything) | No |
| `TTRL_JUDGE_MAX_NEW_TOKENS` | `8` | Judge response length (only `judge_open_ended`) | No |
| `TTRL_JUDGE_NEUTRAL_FALLBACK` | `0.5` | Score when judge output unparseable | No |
| `TTRL_DEBUG` / `TTRL_OE_DEBUG` / `TTRL_JUDGE_DEBUG` | `0` | Verbose logging | No |
| `OPENAI_API_KEY` | – | Required only when `TTRL_AUX_GPT_JUDGE=1`. Sourced from `~/.openai_key` in SLURM scripts. | No |

**Graceful degradation guarantee:** with `TTRL_TASK_TYPE` set to `math` /
`video_qa` / unset, none of the new env vars have any effect — the trainer
runs the original majority-vote TTRL recipe. Verified in dispatch at
`verl/verl/trainer/ppo/ttrl_utils.py:103-114`.

## Offline eval pipeline (GPT-4o-mini judge against gold)

Two-step: dump rollouts from saved checkpoints, then judge.

```bash
# 1. Dump rollouts (autodetects text-only vs multimodal from base model config)
python verl/scripts/dump_rollouts.py \
  --ckpt-dir /path/to/checkpoints/<exp>/<date>/<run> \
  --test-file verl/data/AIME-TTT/test.json \
  --base-model /data/sls/scratch/mvideet/models/Qwen2.5-Math-1.5B \
  --steps 100 200 300 --eval-baseline \
  --output rollouts.jsonl \
  --max-new-tokens 2048 --eval-n 1 --eval-temperature 0.6 \
  --gold-key answer \
  --suffix-prompt $'\nPlease reason step by step, and put your final answer within \\boxed{}.'

# 2. Judge with GPT-4o-mini
source ~/.openai_key
python verl/scripts/judge_rollouts_jsonl.py \
  --rollouts rollouts.jsonl \
  --judge-mode openai --judge-model gpt-4o-mini-2024-07-18 \
  --output judged.jsonl --csv-output results.csv --judge-max-new-tokens 8
```

Wrapper SLURM scripts that do both steps end-to-end:

```bash
sbatch slurm/dump_and_rejudge_aime_gpt4omini.sh        # AIME, step 0 + 300
sbatch slurm/dump_and_rejudge_mmau_gpt4omini.sh        # MMAU
sbatch slurm/dump_and_rejudge_omni_step100_gpt4omini.sh  # daily_omni val20, step 0 + 100
```

## Diagnostics

```bash
# Single-GPU gradient/forward-pass sanity test on Qwen2.5-Omni-3B:
#   T1 forward determinism, T2 grads + element-wise param update, T3 mask sanity
sbatch slurm/diag_grad.sh
```

## Available SLURM scripts (by purpose)

| Category | Script | Notes |
|---|---|---|
| **Open-ended cluster** | `daily_omni_simple_cluster.sh` | Simple binary/continuous cluster, vLLM |
| | `daily_omni_evolrl_cluster.sh` | Full EVOL-RL (banded + novelty + clip-higher) |
| **Open-ended judge** | `daily_omni_judge.sh` | Original HF-rollout judge |
| | `daily_omni_judge_v2.sh` | T=1.0 + max_resp=1024 (the +3.2 MMAU recipe) |
| | `daily_omni_judge_v3.sh` | judge_v2 + vLLM + aux monitoring + W&B filter |
| **Open-ended other** | `daily_omni_open.sh` | Embedding-medoid (no judge, no cluster) |
| | `daily_omni_text.sh` / `daily_omni_text_omni.sh` | Text-only baselines |
| | `daily_omni_ttrl_gspo.sh` | GSPO advantage estimator |
| | `daily_omni_ttrv_fromscratch.sh` | MCQ-style TTRV |
| | `daily_omni_cwa.sh` | Confidence-weighted advantage |
| **Math (AIME/AMC)** | `aime.sh` / `aime_simple_cluster.sh` / `aime_sft.sh` | Various AIME ablations |
| | `amc.sh` / `amc_omni.sh` | AMC variants |
| **MMAU (audio MCQ)** | `mmau_omni.sh` / `mmau_open.sh` / `mmau_judge_v2.sh` | MMAU ablations |
| **OmniBench** | `omnibench.sh` / `omnivideo_text.sh` | OmniBench |
| **SFT baselines** | `oneshot_sft.sh` / `majorityVote_sft.sh` / `pseudolabel_*.sh` | SFT comparison runs |
| **Eval** | `dump_and_rejudge_*.sh` | Dump + judge pipeline |
| | `eval_*.sh` | Checkpoint eval scripts |
| **Diagnostics** | `diag_grad.sh` | Gradient flow / forward determinism / mask sanity |
| | `test_vllm_omni.sh` | 5-step vLLM-Omni smoke test |

## Qwen2.5-Omni integration notes (historical reference)

Original wiring of **Qwen2.5-Omni (Thinker/Talker)** into VERL — preserved
for context. These patches are now part of main.

- **Local model path**: `verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh` sets `BACKBONE_PATH=...`.
- **HF rollout: disable audio output**: `verl/verl/workers/rollout/hf_rollout.py` adds `return_audio=False`. `model.disable_talker()` saves memory.
- **HF rollout: multimodal inputs**: reads `non_tensor_batch["multi_modal_inputs"]`, concatenates, moves to device, casts floats to bfloat16.
- **TTRL reward shape**: `verl/verl/utils/dataset/rl_omni_dataset.py` sets `row_dict["reward_model"] = {"style": "rule", "ground_truth": ...}` and ensures `row_dict["data_source"]` is always present.
- **Thinker forward for PPO**: `verl/verl/workers/actor/dp_actor.py` + `dp_critic.py` go through Thinker (top-level Omni wrapper is generation-focused). With FSDP param offload, wrap in `FSDP.summon_full_params(...)`.
- **Votes vs samples**: `n_votes_per_prompt` rollouts → majority-vote GT → downsample to `n_samples_per_prompt`. HF gotcha: `verl/verl/workers/fsdp_workers.py` strips `kwargs["n"]` for HF rollout.

## vLLM rollout for Qwen2.5-Omni (current)

vLLM rollout for Qwen2.5-Omni with `use_audio_in_video=True` requires three
local patches to `vllm/model_executor/models/qwen2_5_omni_thinker.py` (in
the conda env) and six patches to `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
(in this repo). The verl-side ones are committed; the vLLM-side ones live
in the conda env and would be lost on env rebuild.

Upstream PR drafts (for the verl-side patches and the three vLLM-side
patches) live locally under `docs/pr_drafts/` but are gitignored. Ask
the maintainer for the drafts if you want to upstream.

## Code layout

```
verl/verl/trainer/ppo/
  ttrl_utils.py                 # TTRL_TASK_TYPE dispatch (apply_ttrl_gt)
  ttrl_evolrl_cluster_vote.py   # simple_cluster + evolrl_cluster
  ttrl_judge_vote.py            # judge_open_ended (LLM-as-judge)
  ttrl_open_ended_vote.py       # open_ended_video (embedding-medoid)
  ttrl_aux_metrics.py           # BLEU/ROUGE-L/EM/GPT-judge monitoring

verl/verl/utils/reward_score/
  ttrl_open_ended/embedding.py  # encoder dispatch (bge / qwen3 / mpnet)
  ttrl_judge/__init__.py        # reward_func (score_map lookup + BGE fallback)

verl/verl/workers/rollout/vllm_rollout/
  vllm_rollout_spmd.py          # multimodal patches + resilience wrapper

verl/verl/utils/
  tracking.py                   # W&B noise filter (TTRL_LOG_DROP_PATTERNS)

verl/scripts/
  dump_rollouts.py              # Offline rollout dumper (text + multimodal)
  judge_rollouts_jsonl.py       # GPT/local-model judge over rollouts.jsonl
  diagnose_grad_path.py         # Single-GPU forward/backward diagnostics
  eval_mmau_offline.py          # FSDP shard merger + Omni loader

slurm/                          # All SLURM submission scripts
docs/                           # (pr_drafts/ exists locally, gitignored)
```

## Things on other branches not yet merged to main

- `ttrl-cg` — Confidence-gated TTRL (`TTRL_CG_ENABLE=1` filters training prompts by majority confidence band).
- `ngram-soft-voting` — N-gram soft-consensus voting (embedding-free aggregation).
- `ttrv` — TTRV MCQ-flavored ablations (AIME → MATH-L1 pivot, T-tuning, HF rollout fallback).
