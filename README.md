### Quickstart

- **Env**: `conda activate verl310`
- **Run (SLURM)**: `sbatch slurm/daily_omni.sh`
- **Run (interactive)**: `bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh`

### Qwen2.5-Omni integration notes (what was changed + where)

This repo wires **Qwen2.5-Omni (Thinker/Talker)** into VERL PPO/TTRL with HF rollout + multimodal (video/audio) inputs.

- **Local model path (avoid hub timeouts)**: set in `verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh` via `BACKBONE_PATH=...`.

- **HF rollout: disable audio output (Talker)**
  - **Why**: `model.disable_talker()` saves memory; then `generate()` must use `return_audio=False`.
  - **Where**: `verl/verl/workers/rollout/hf_rollout.py` adds `return_audio=False` when running Omni conditional generation wrappers.

- **HF rollout: pass multimodal inputs into `generate()`**
  - **Why**: without passing video/audio tensors, the model behaves like text-only and produces unrelated text.
  - **Where**: `verl/verl/workers/rollout/hf_rollout.py` reads `prompts.non_tensor_batch["multi_modal_inputs"]`, concatenates, moves to device, and casts floats to `bfloat16`, then forwards them into `self.module.generate(...)`.
  - **Also**: `verl/verl/trainer/ppo/ray_trainer.py` was updated so `multi_modal_inputs` survives the batch plumbing into rollout.

- **TTRL reward input shape (`reward_model` KeyError)**
  - **Why**: reward manager expects `non_tensor_batch["reward_model"]["ground_truth"]`.
  - **Where**: `verl/verl/utils/dataset/rl_omni_dataset.py` sets
    - `row_dict["reward_model"] = {"style": "rule", "ground_truth": ...}`
    - and ensures `row_dict["data_source"]` is always present.

- **Qwen2.5-Omni forward pass for PPO logprobs/critic (Thinker)**
  - **Problem**: the top-level Omni wrapper is generation-focused; forward for training must go through **Thinker**.
  - **Where**:
    - `verl/verl/workers/actor/dp_actor.py`
    - `verl/verl/workers/critic/dp_critic.py`
  - **Important**: when FSDP param offload is enabled, calling `thinker` directly bypasses FSDP and can crash with “tensor data not allocated”. The fix wraps thinker-forward in `FSDP.summon_full_params(...)`.

- **TTRL: votes vs samples with HF rollout**
  - **Codepath**: `verl/verl/trainer/ppo/ray_trainer.py` generates `n_votes_per_prompt`, does majority-vote GT, then downsamples to `n_samples_per_prompt`.
  - **HF gotcha**: `verl/verl/workers/fsdp_workers.py` strips `kwargs["n"]` for HF rollout, so HF uses `actor_rollout_ref.rollout.n` only.
  - **Fix**: in `verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh`, set:
    - `actor_rollout_ref.rollout.n = ${N_VOTES_PER_PROMPT}`
    - and keep `ttrl.n_samples_per_prompt = ${N_SAMPLES_PER_PROMPT}` for training.

- **SLURM wrapper restored**
  - **Where**: `slurm/daily_omni.sh` (requests GPUs/CPUs/mem, activates conda, runs the example script; logs to `slurm/out/` + `slurm/err/`).

- **GitHub push: ignore and purge huge wheel blobs**
  - **Ignore patterns**: `.gitignore` ignores `*.whl*` and the vendor wheel filenames under `verl/`.
  - **If already committed**: history must be rewritten (e.g. `git filter-repo`) or GitHub rejects pushes (>100MB blobs).

