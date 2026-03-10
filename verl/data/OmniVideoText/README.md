# OmniVideoText

Text-only version of the OmniVideo dataset for TTRL experiments. Each entry contains only textual fields (question, answer, metadata); no `audio_file` or `video_file` paths.

Use this to evaluate whether TTRL on text-only data (without video/audio) can improve performance on the same QA tasks, as a baseline or ablation.

## Files

- `train.json`, `test.json` – full splits
- `train_sanity.json`, `test_sanity.json` – smaller sanity-check splits
- `migrate_to_text.py` – script to regenerate from OmniVideo (strips `audio_file`, `video_file`)

## Usage

Point your TTRL config to OmniVideoText JSON files instead of OmniVideo. The `RLOMNIDataset` loader supports text-only entries: when `video_file` is absent, it builds messages with question text only.

Example (in your shell script or config):

```bash
# Use OmniVideoText instead of OmniVideo
+data.train_files='["verl/data/OmniVideoText/train_sanity.json"]'
+data.val_files='["verl/data/OmniVideoText/test_sanity.json"]'
```
