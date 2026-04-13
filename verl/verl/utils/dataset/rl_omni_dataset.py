# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# Audio sample rate for Qwen3-Omni
QWEN_OMNI_SAMPLE_RATE = 16000


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLOMNIDataset(Dataset):
    """
    Load and preprocess RLOMNI data from Parquet/JSON files for Qwen3-Omni.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Handles images/videos/audio via Qwen3OmniMoeProcessor.
    - Extracts audio from video files automatically.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet/JSON file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Qwen3OmniMoeProcessor for multimodal inputs.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.video_file_key = config.get("video_file_key", "video_file")  # For direct video paths
        self.image_file_key = config.get("image_file_key", "image_file")  # For direct image paths (e.g. OmniBench)
        self.audio_file_key = config.get("audio_file_key", "audio_file")  # For direct audio paths (audio-only datasets)
        self.question_key = config.get("question_key", "question")  # For question text
        self.answer_key = config.get("answer_key", "answer")  # For ground truth answer
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.suffix_prompt = config.get("suffix_prompt", None)
        self.enable_thinking = config.get("enable_thinking", None)
        
        # Qwen3-Omni specific settings
        self.use_audio_in_video = config.get("use_audio_in_video", True)
        self.audio_sample_rate = config.get("audio_sample_rate", QWEN_OMNI_SAMPLE_RATE)
        self.use_omnivideo_text = config.get("use_omnivideo_text", False)

        # Video downsampling knobs (passed through to qwen_omni_utils / qwen_vl_utils)
        self.video_fps = config.get("video_fps", None)        # e.g. 1.0 (default in lib: 2.0)
        self.video_max_frames = config.get("video_max_frames", None)  # e.g. 32  (default in lib: 768)
        self.video_min_frames = config.get("video_min_frames", None)  # e.g. 4   (default in lib: 4)
        self.video_max_pixels = config.get("video_max_pixels", None)  # per-frame max pixels

        # Audio truncation: clip audio to at most this many seconds (None = full duration)
        self.max_audio_duration = config.get("max_audio_duration", None)  # e.g. 30.0
        
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        """
        Downloads/copies the data files to a local cache directory for processing.

        If `use_origin_parquet` is True, uses the original data file list; 
        otherwise, uses (and potentially overwrites) the dataset's current data files list.
        For each file, ensures the file is available locally (copying if needed) using `copy_to_local`, 
        and updates the path in `self.data_files`. This helps to ensure distributed training and 
        preprocessing can access local files efficiently, possibly leveraging shared memory if enabled.
        """
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            # Support both parquet and json files
            if data_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        if self.suffix_prompt:
            print(f"Apply suffix prompt {self.suffix_prompt}")
            # Use num_proc=None (main process, no subprocess) — string-append is trivial and
            # forking inside a Ray actor causes SIGTERM from inherited Ray handles.
            self.dataframe = self.dataframe.map(self._add_suffix_to_entry, num_proc=None)

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc.copy())  # Use copy to avoid modifying original
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    if self.use_omnivideo_text:
                        return len(
                            processor(text=[raw_prompt], return_tensors="pt", padding=True)["input_ids"][0]
                        )
                    images = (
                        [process_image(image) for image in doc.get(image_key, [])] if image_key in doc else None
                    )
                    videos = (
                        [process_video({"video": v}) for v in doc.get(video_key, [])] if video_key in doc else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    if self.question_key in doc:
                        messages = self._build_messages(doc.copy())
                        raw_prompt = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                        return len(tokenizer.encode(raw_prompt, add_special_tokens=False))
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=None,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _video_sampling_kwargs(self) -> dict:
        """Return a dict of fps/frame/pixel overrides to inject into video content elements."""
        kw = {}
        if self.video_fps is not None:
            kw["fps"] = self.video_fps
        if self.video_max_frames is not None:
            kw["max_frames"] = self.video_max_frames
        if self.video_min_frames is not None:
            kw["min_frames"] = self.video_min_frames
        if self.video_max_pixels is not None:
            kw["max_pixels"] = self.video_max_pixels
        return kw

    def _build_messages(self, example: dict):
        """
        Build chat messages from example data.
        
        Supports four formats:
        1. Standard format with 'prompt' key containing chat messages
        2. OmniVideo format with 'question', 'video_file' keys (video+audio multimodal)
        3. Audio-only format with 'question', 'audio_file' keys (audio multimodal, no video)
        4. OmniVideoText format with 'question' only (text-only, no video/audio)
        """
        video_kw = self._video_sampling_kwargs()

        # Check if this is OmniVideo / Audio-only / OmniVideoText format (has question)
        if self.question_key in example:
            question = example.get(self.question_key, "")
            video_file = example.get(self.video_file_key, "")
            audio_file = example.get(self.audio_file_key, "")
            image_file = example.get(self.image_file_key, "")

            if image_file and not self.use_omnivideo_text:
                # Image (+audio): e.g. OmniBench
                content_list = [{"type": "image", "image": image_file}]
                if audio_file:
                    content_list.append({"type": "audio", "audio": audio_file})
                content_list.append({"type": "text", "text": question})
                messages = [{"role": "user", "content": content_list}]
            elif video_file and not self.use_omnivideo_text:
                # OmniVideo: include video (content_list for multimodal processor)
                video_elem = {"type": "video", "video": video_file, **video_kw}
                content_list = [
                    video_elem,
                    {"type": "text", "text": question},
                ]
                messages = [{"role": "user", "content": content_list}]
            elif audio_file and not self.use_omnivideo_text:
                # Audio-only: include audio element for Qwen2.5-Omni processor
                content_list = [
                    {"type": "audio", "audio": audio_file},
                    {"type": "text", "text": question},
                ]
                messages = [{"role": "user", "content": content_list}]
            else:
                # OmniVideoText: text-only, use plain string for text-model chat template
                messages = [{"role": "user", "content": question}]
            return messages
        
        # Standard format with prompt key
        messages: list = example.pop(self.prompt_key) if self.prompt_key in example else []

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video", **video_kw})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages
    
    def _add_suffix_to_entry(self, entry):
        # OmniVideo format: entry has question_key / video_file_key, no prompt list
        if self.question_key in entry:
            entry[self.question_key] = entry[self.question_key] + self.suffix_prompt
            return entry
        # Standard format: entry has prompt_key (list of messages)
        if self.prompt_key in entry:
            entry[self.prompt_key][-1]["content"] = entry[self.prompt_key][-1]["content"] + self.suffix_prompt
        return entry
    
    def __getitem__(self, item):
        """
        Get a single item from the dataset with multimodal processing.
        
        For Qwen3-Omni, this handles:
        - Video processing (extracting frames)
        - Audio extraction from video using official qwen_omni_utils.process_mm_info
        - Processing through Qwen3OmniMoeProcessor
        """
        row_dict: dict = dict(self.dataframe[item])  # Make a copy to avoid modifying original
        messages = self._build_messages(row_dict.copy())
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            if self.enable_thinking is not None:
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
            else:
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            multi_modal_data = {}
            images = None
            videos = None
            audios = None

            if self.use_omnivideo_text:
                try:
                    model_inputs = self.processor(text=[raw_prompt], return_tensors="pt", padding=True)
                except TypeError:
                    model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            else:
                # Qwen3OmniMoeProcessor has feature_extractor and uses qwen_omni_utils;
                # Qwen2_5OmniProcessor also has feature_extractor but uses a different pipeline.
                is_qwen3_omni = (
                    "Qwen3Omni" in self.processor.__class__.__name__
                    and hasattr(self.processor, 'feature_extractor')
                    and self.processor.feature_extractor is not None
                )

                if is_qwen3_omni:
                    # Use official qwen_omni_utils.process_mm_info for Qwen3-Omni
                    # This handles audio extraction from video automatically
                    from qwen_omni_utils import process_mm_info

                    audios, images, videos = process_mm_info(
                        messages,
                        use_audio_in_video=self.use_audio_in_video,
                    )

                    if audios and self.max_audio_duration is not None:
                        max_samples = int(self.audio_sample_rate * self.max_audio_duration)
                        audios = [a[:max_samples] for a in audios]

                    # Store multi_modal_data for reference
                    if images:
                        multi_modal_data["image"] = images
                    if videos:
                        multi_modal_data["video"] = videos
                    if audios:
                        multi_modal_data["audio"] = audios

                    # Use Qwen3-Omni processor with all modalities
                    model_inputs = self.processor(
                        text=[raw_prompt],
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=self.use_audio_in_video,
                    )
                else:
                    # Standard Qwen2-VL / Qwen2.5-Omni style processing
                    # Handle images: from image list (image_key) or single file (image_file_key)
                    if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                        images = [process_image(image) for image in row_dict.get(self.image_key, [])]
                        multi_modal_data["image"] = images
                    elif self.image_file_key in row_dict and row_dict.get(self.image_file_key, None) is not None:
                        image_path = row_dict.get(self.image_file_key)
                        from PIL import Image as PILImage
                        img = PILImage.open(image_path).convert("RGB")
                        images = [img]
                        multi_modal_data["image"] = images

                    # Handle videos
                    if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                        video_list = row_dict.get(self.video_key, [])
                        videos = [process_video({"video": v} if isinstance(v, str) else v) for v in video_list]
                        multi_modal_data["video"] = [video.numpy() for video in videos]
                    elif self.video_file_key in row_dict and row_dict.get(self.video_file_key, None) is not None:
                        video_path = row_dict.get(self.video_file_key)
                        videos = [process_video({"video": video_path})]
                        multi_modal_data["video"] = [video.numpy() for video in videos]

                    # Handle audio inputs: use audio_file if available,
                    # otherwise extract audio from video .mp4 directly
                    audio_file = row_dict.get(self.audio_file_key, None)
                    if not audio_file and videos is not None and self.use_audio_in_video:
                        # Fall back to extracting audio from the video file
                        video_path = row_dict.get(self.video_file_key, None)
                        if video_path:
                            audio_file = video_path
                    if audio_file:
                        import librosa
                        waveform, _ = librosa.load(audio_file, sr=self.audio_sample_rate)
                        if self.max_audio_duration is not None:
                            max_samples = int(self.audio_sample_rate * self.max_audio_duration)
                            waveform = waveform[:max_samples]
                        # Pad very short audio to avoid avg_pool1d crash in audio_tower
                        # (needs at least ~0.5s for Whisper feature extraction + pooling)
                        min_samples = int(self.audio_sample_rate * 1.0)
                        if len(waveform) < min_samples:
                            import numpy as _np
                            waveform = _np.pad(waveform, (0, min_samples - len(waveform)))
                        audios = [waveform]
                        multi_modal_data["audio"] = audios

                    # Route to correct processor call based on available modalities
                    if images is None and videos is None and audios is None:
                        # Text-only
                        try:
                            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt", padding=True)
                        except TypeError:
                            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                    elif audios is not None and videos is not None:
                        # Video + audio (OmniVideo with use_audio_in_video)
                        model_inputs = self.processor(
                            text=[raw_prompt], videos=videos, audio=audios,
                            return_tensors="pt", padding=True,
                            use_audio_in_video=self.use_audio_in_video,
                        )
                    elif audios is not None and images is not None and videos is None:
                        # Image + audio (e.g. OmniBench)
                        model_inputs = self.processor(
                            text=[raw_prompt], images=images, audio=audios,
                            return_tensors="pt", padding=True,
                        )
                    elif audios is not None and images is None and videos is None:
                        # Audio-only (Qwen2.5-Omni / MMAU)
                        model_inputs = self.processor(
                            text=[raw_prompt], audio=audios, return_tensors="pt", padding=True,
                            use_audio_in_video=False,
                        )
                    else:
                        model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            # Diagnostic: confirm which modalities the processor produced
            import os as _os
            if _os.environ.get("OMNI_INPUT_DEBUG") == "1":
                _mm_keys = {k: (v.shape if hasattr(v, 'shape') else type(v).__name__) for k, v in model_inputs.items()}
                _has_vid = any('video' in k or 'pixel_values_video' in k for k in model_inputs)
                _has_aud = any('input_features' in k or 'feature_attention_mask' in k for k in model_inputs)
                print(f"[MM_VERIFY] id={row_dict.get('id','?')} | video={_has_vid} audio={_has_aud} | "
                      f"input_ids={input_ids.shape} | processor_keys={_mm_keys}")

            # Preserve second_per_grid_ts and feature_attention_mask for M-RoPE computation
            second_per_grid_ts = model_inputs.pop("second_per_grid_ts", None)
            feature_attention_mask = model_inputs.get("feature_attention_mask", None)

            # Store multi_modal_data and multi_modal_inputs
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            
            # Store use_audio_in_video flag for model forward pass
            row_dict["multi_modal_inputs"]["use_audio_in_video"] = self.use_audio_in_video

        else:
            if self.enable_thinking is not None:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
            else:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Handle position_ids based on processor type
        if self.processor is not None:
            processor_class_name = getattr(self.processor, 'image_processor', None)
            processor_class_name = processor_class_name.__class__.__name__ if processor_class_name else ""
            
            main_processor_class = self.processor.__class__.__name__
            is_qwen2_5_omni = "Qwen2_5Omni" in main_processor_class or "Qwen25Omni" in main_processor_class
            
            if is_qwen2_5_omni:
                from verl.models.transformers.qwen2_5_omni import get_rope_index as get_rope_index_omni

                audio_seqlens = None
                if feature_attention_mask is not None:
                    audio_seqlens = torch.sum(feature_attention_mask, dim=1)

                position_ids = [
                    get_rope_index_omni(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        attention_mask=attention_mask[0],
                        use_audio_in_video=self.use_audio_in_video,
                        audio_seqlens=audio_seqlens,
                        second_per_grids=second_per_grid_ts,
                    )
                ]  # (1, 3, seq_len)
            elif "Qwen2VLImageProcessor" in processor_class_name or "Qwen3Omni" in processor_class_name:
                from verl.models.transformers.qwen2_vl import get_rope_index

                position_ids = [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ]  # (1, 3, seq_len)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # Store ground truth answer for reward computation in the expected format
        # NaiveRewardManager expects: non_tensor_batch["reward_model"]["ground_truth"]
        if self.answer_key in row_dict:
            row_dict["reward_model"] = {
                "style": "rule",
                "ground_truth": row_dict.get(self.answer_key)
            }
        
        # Set data_source for reward function routing
        # Use "source" field if available, otherwise default to "omnivideo"
        row_dict["data_source"] = row_dict.get("source", row_dict.get("data_source", "omnivideo"))
        
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", row_dict.get("id", item))
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict.get("data_source", "unknown"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
