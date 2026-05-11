"""
Minimal standalone reproducer for the merge_interleaved_embeddings bug
in vLLM 0.16's qwen2_5_omni_thinker.py.

Run with the buggy version:
    python repro_minimal.py
Expect: RuntimeError: shape mismatch ... cannot be broadcast ...

Apply the alternating-pair fix → expect: 'OK' printed.

This script does not require a GPU or model download — it constructs
synthetic inputs and exercises only the categorization + scatter logic.
"""

import torch


def merge_interleaved_embeddings_buggy(
    inputs_embeds, multimodal_embeddings, is_video, is_audio, is_multimodal,
    num_video, num_audio,
):
    """Current vLLM 0.16 implementation (verbatim, simplified)."""
    video_embeds, audio_embeds, other_embeds = [], [], []
    video_remaining = num_video
    audio_remaining = num_audio

    for emb in multimodal_embeddings:
        n = emb.shape[0]
        if video_remaining > 0 and n <= video_remaining:
            video_embeds.append(emb)
            video_remaining -= n
        elif audio_remaining > 0 and n <= audio_remaining:
            audio_embeds.append(emb)
            audio_remaining -= n
        else:
            other_embeds.append(emb)

    if video_embeds:
        video_positions = is_video.nonzero(as_tuple=True)[0]
        inputs_embeds[video_positions] = torch.cat(video_embeds, dim=0)
    if audio_embeds:
        audio_positions = is_audio.nonzero(as_tuple=True)[0]
        inputs_embeds[audio_positions] = torch.cat(audio_embeds, dim=0)
    return inputs_embeds


def merge_interleaved_embeddings_fixed(
    inputs_embeds, multimodal_embeddings, is_video, is_audio, is_multimodal,
    num_video, num_audio,
):
    """With the alternating-pair categorization fix."""
    video_embeds, audio_embeds, other_embeds = [], [], []
    video_remaining = num_video
    audio_remaining = num_audio

    n_embs = len(multimodal_embeddings)
    used_alternating = False
    if n_embs >= 2 and n_embs % 2 == 0 and num_video > 0 and num_audio > 0:
        cand_v = sum(multimodal_embeddings[i].shape[0] for i in range(0, n_embs, 2))
        cand_a = sum(multimodal_embeddings[i].shape[0] for i in range(1, n_embs, 2))
        if cand_v == num_video and cand_a == num_audio:
            video_embeds = list(multimodal_embeddings[0::2])
            audio_embeds = list(multimodal_embeddings[1::2])
            video_remaining = audio_remaining = 0
            used_alternating = True

    if not used_alternating:
        for emb in multimodal_embeddings:
            n = emb.shape[0]
            if video_remaining > 0 and n <= video_remaining:
                video_embeds.append(emb)
                video_remaining -= n
            elif audio_remaining > 0 and n <= audio_remaining:
                audio_embeds.append(emb)
                audio_remaining -= n
            else:
                other_embeds.append(emb)

    if video_embeds:
        video_positions = is_video.nonzero(as_tuple=True)[0]
        inputs_embeds[video_positions] = torch.cat(video_embeds, dim=0)
    if audio_embeds:
        audio_positions = is_audio.nonzero(as_tuple=True)[0]
        inputs_embeds[audio_positions] = torch.cat(audio_embeds, dim=0)
    return inputs_embeds


def make_inputs():
    """2 batched prompts, each with (video, audio) interleaved.
    These exact sizes triggered the bug in production with verl GRPO."""
    embeds = [
        torch.zeros(8970, 16),  # V_p1 — visual frames of prompt 1
        torch.ones(375, 16),    # A_p1 — audio chunks of prompt 1
        torch.zeros(5100, 16),  # V_p2 — visual frames of prompt 2
        torch.ones(75, 16),     # A_p2 — audio chunks of prompt 2
    ]
    num_video = 8970 + 5100
    num_audio = 375 + 75
    seq_len = num_video + num_audio
    is_video = torch.zeros(seq_len, dtype=torch.bool)
    is_audio = torch.zeros(seq_len, dtype=torch.bool)
    is_video[:num_video] = True
    is_audio[num_video:] = True
    is_multimodal = is_video | is_audio
    inputs_embeds = torch.zeros(seq_len, 16)
    return inputs_embeds, embeds, is_video, is_audio, is_multimodal, num_video, num_audio


if __name__ == "__main__":
    import sys

    print("Buggy version:")
    args = make_inputs()
    try:
        merge_interleaved_embeddings_buggy(*args)
        print("  (no error — repro failed?)")
    except RuntimeError as e:
        print(f"  RuntimeError: {e}")

    print("Fixed version:")
    args = make_inputs()
    try:
        out = merge_interleaved_embeddings_fixed(*args)
        # Sanity: video positions filled with zeros, audio with ones
        assert out[: 8970 + 5100].sum() == 0
        assert out[8970 + 5100 :].sum() == (375 + 75) * 16
        print("  OK ✓")
    except (RuntimeError, AssertionError) as e:
        print(f"  Fix failed: {e}")
        sys.exit(1)
