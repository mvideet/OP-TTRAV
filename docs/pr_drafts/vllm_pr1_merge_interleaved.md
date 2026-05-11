# PR Draft 1: vllm-project/vllm — fix Qwen2.5-Omni `merge_interleaved_embeddings` greedy categorization

## Title

`[Bugfix] Qwen2.5-Omni: fix merge_interleaved_embeddings categorization for batched prompts with use_audio_in_video=True`

## Summary

When batching multiple prompts that each contain interleaved video+audio (`use_audio_in_video=True`), `merge_interleaved_embeddings` in `qwen2_5_omni_thinker.py` misclassifies an audio embedding as video, producing a shape mismatch:

```
RuntimeError: shape mismatch: value tensor of shape [9420, 2048] cannot be broadcast
              to indexing result of shape [14070, 2048]
```

The root cause is the greedy categorization loop's "fits in remaining bucket" check. When embeddings arrive as `[V_p1, A_p1, V_p2, A_p2, …]` and `A_p1`'s size happens to be `≤ video_remaining` after consuming `V_p1`, it gets dropped into the video bucket instead of the audio one.

## Repro

Single-prompt configs (where `n_embeddings == 2` with sizes `[V, A]`) happen to work because `V` exhausts `video_remaining` cleanly, and `A` then has nowhere to go but `audio_remaining`. The bug only triggers with **batched prompts (`n_embeddings ≥ 4`)** where the second video's size is small enough that the audio embedding from the first prompt fits in the remaining video bucket.

Concrete failing case from a verl GRPO rollout:

```
num_video=14070, num_audio=450, n_embeddings=4
embed_shapes=[(8970, 2048), (375, 2048), (5100, 2048), (75, 2048)]
                ↑ V_p1       ↑ A_p1       ↑ V_p2       ↑ A_p2

Greedy categorization (current):
  emb[0]=8970: video_remaining=14070, 8970≤14070 → video. video_remaining=5100
  emb[1]=375:  video_remaining=5100,  375≤5100  → video (WRONG). video_remaining=4725
  emb[2]=5100: video_remaining=4725,  5100>4725 AND 5100>audio(450) → other_embeds
  emb[3]=75:   video_remaining=4725,  75≤4725 → video. video_remaining=4650

Result: video_embeds_total=9420, num_video=14070 → 4650-position gap → assert fails
```

## Fix

Recognize the alternating-pair pattern that the standard Qwen2.5-Omni MultiModalDataParser produces for batched `use_audio_in_video=True` inputs. Try this categorization first; fall back to the existing greedy logic if the alternating-pair sums don't match.

```python
n_embs = len(multimodal_embeddings)
used_alternating = False
if n_embs >= 2 and n_embs % 2 == 0 and num_video > 0 and num_audio > 0:
    cand_v = sum(multimodal_embeddings[i].shape[0] for i in range(0, n_embs, 2))
    cand_a = sum(multimodal_embeddings[i].shape[0] for i in range(1, n_embs, 2))
    if cand_v == num_video and cand_a == num_audio:
        video_embeds = list(multimodal_embeddings[0::2])
        audio_embeds = list(multimodal_embeddings[1::2])
        video_remaining = 0
        audio_remaining = 0
        used_alternating = True

if not used_alternating:
    # ... existing greedy loop ...
```

Conservative: only takes the alternating path when sums match exactly, so the existing single-prompt (`n_embs == 2`) cases keep working through the same path.

## Test

Add a regression test in `tests/models/multimodal/processing/test_qwen2_5_omni.py`:

```python
def test_merge_interleaved_embeddings_batched_alternating():
    """Regression: batched prompts with use_audio_in_video=True must
    correctly distinguish video and audio embeddings even when an
    audio chunk is small enough to fit in the remaining video bucket
    of a different prompt."""
    import torch
    from vllm.model_executor.models.qwen2_5_omni_thinker import (
        merge_interleaved_embeddings,
    )

    # 2 prompts: each contributes (video, audio) embeddings in order.
    embeds = [
        torch.zeros(8970, 2048),  # V_p1
        torch.ones(375, 2048),    # A_p1
        torch.zeros(5100, 2048),  # V_p2
        torch.ones(75, 2048),     # A_p2
    ]
    num_video = 8970 + 5100
    num_audio = 375 + 75

    # Build is_video / is_audio masks at correct counts.
    seq_len = num_video + num_audio
    is_video = torch.zeros(seq_len, dtype=torch.bool)
    is_audio = torch.zeros(seq_len, dtype=torch.bool)
    is_video[:num_video] = True
    is_audio[num_video:] = True
    is_multimodal = is_video | is_audio
    inputs_embeds = torch.zeros(seq_len, 2048)

    out = merge_interleaved_embeddings(
        inputs_embeds, embeds, is_video, is_audio, is_multimodal,
        num_video, num_audio,
    )

    # Should not raise; video positions filled with 0.0, audio with 1.0
    assert out[:num_video].sum() == 0  # video embeddings (zeros)
    assert out[num_video:].sum() == num_audio * 2048  # audio embeddings (ones)
```

## Affected versions

vLLM 0.16.0 confirmed broken. Behavior likely the same in 0.16.1+ since the categorization logic appears unchanged in the master branch as of writing.

## Diff

```diff
--- a/vllm/model_executor/models/qwen2_5_omni_thinker.py
+++ b/vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -170,17 +170,30 @@ def merge_interleaved_embeddings(...):
-    # Categorize embeddings by modality based on token counts.
-    # ...
     video_embeds: list[torch.Tensor] = []
     audio_embeds: list[torch.Tensor] = []
     other_embeds: list[torch.Tensor] = []
     video_remaining = num_video
     audio_remaining = num_audio

-    for emb in multimodal_embeddings:
-        n = emb.shape[0]
-        if video_remaining > 0 and n <= video_remaining:
-            video_embeds.append(emb)
-            video_remaining -= n
-        elif audio_remaining > 0 and n <= audio_remaining:
-            audio_embeds.append(emb)
-            audio_remaining -= n
-        else:
-            other_embeds.append(emb)
+    # When batched prompts contribute (video, audio) pairs in order, the
+    # total embedding list arrives as [V_p1, A_p1, V_p2, A_p2, ...]. The
+    # original greedy "fits in video bucket" check misclassifies a small
+    # audio embedding as video when video_remaining > audio embedding size.
+    # Try the alternating-pair pattern first; fall back to greedy only if
+    # it doesn't match the expected totals.
+    n_embs = len(multimodal_embeddings)
+    used_alternating = False
+    if n_embs >= 2 and n_embs % 2 == 0 and num_video > 0 and num_audio > 0:
+        cand_v = sum(multimodal_embeddings[i].shape[0] for i in range(0, n_embs, 2))
+        cand_a = sum(multimodal_embeddings[i].shape[0] for i in range(1, n_embs, 2))
+        if cand_v == num_video and cand_a == num_audio:
+            video_embeds = list(multimodal_embeddings[0::2])
+            audio_embeds = list(multimodal_embeddings[1::2])
+            video_remaining = 0
+            audio_remaining = 0
+            used_alternating = True
+
+    if not used_alternating:
+        for emb in multimodal_embeddings:
+            n = emb.shape[0]
+            if video_remaining > 0 and n <= video_remaining:
+                video_embeds.append(emb)
+                video_remaining -= n
+            elif audio_remaining > 0 and n <= audio_remaining:
+                audio_embeds.append(emb)
+                audio_remaining -= n
+            else:
+                other_embeds.append(emb)
```
