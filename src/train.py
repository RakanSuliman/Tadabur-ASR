"""
train.py
Fine-tunes openai/whisper-medium on the Tadabur Quranic ASR dataset.

Strategy:
    - Load N shards from network volume into RAM at startup (one-time cost)
    - Train entirely from RAM — no disk seeks during training
    - Saves checkpoints back to network volume

Usage:
    python train.py
"""

import os
import io
import glob
import torch
import librosa
import numpy as np
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate as evaluate_module

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME      = "openai/whisper-medium"
DATASET_PATH    = "/workspace/tadabur/data"
OUTPUT_DIR      = "/workspace/whisper-medium-tadabur"
LANGUAGE        = "arabic"
TASK            = "transcribe"
SAMPLE_RATE     = 16000
MAX_AUDIO_SECS  = 30

BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 4    # effective batch = 32
MAX_STEPS        = 20000
WARMUP_STEPS     = 500
LEARNING_RATE    = 1e-5
SAVE_STEPS       = 1000
EVAL_STEPS       = 1000
LOGGING_STEPS    = 100
MAX_SHARDS       = 10   # ~9GB RAM — well within 251GB available
EVAL_SHARDS      = 1    # 1 shard for eval, take first 200 samples

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Processor ────────────────────────────────────────────────────────────

print("Loading processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor         = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_shards_into_ram(parquet_dir, max_audio_secs, max_shards=None, max_samples=None):
    """
    Load parquet shards fully into RAM.
    Stores audio bytes + text — no disk access during training.
    Uses pyarrow scalar access to bypass ArrowNotImplementedError.
    """
    samples = []
    parquet_files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    if max_shards:
        parquet_files = parquet_files[:max_shards]

    print(f"Loading {len(parquet_files)} shard(s) into RAM...")

    for shard_idx, shard_path in enumerate(parquet_files):
        shard_samples = 0
        try:
            pf = pq.ParquetFile(shard_path)
            for batch in pf.iter_batches(batch_size=128):
                durations  = batch.column("ayah_duration_s")
                texts      = batch.column("text_ar_uthmani")
                audio_col  = batch.column("audio")

                for i in range(len(batch)):
                    try:
                        duration = durations[i].as_py()
                        if not (0 < duration <= max_audio_secs):
                            continue

                        text = texts[i].as_py()
                        if not text:
                            continue

                        audio_bytes = audio_col[i]["bytes"].as_py()
                        if not audio_bytes:
                            continue

                        samples.append({
                            "audio_bytes": audio_bytes,
                            "text"       : text,
                        })
                        shard_samples += 1

                        if max_samples and len(samples) >= max_samples:
                            print(f"  Shard {shard_idx+1}: +{shard_samples} → total {len(samples)}")
                            return samples

                    except Exception:
                        continue

        except Exception as e:
            print(f"  Warning: could not read {shard_path}: {e}")
            continue

        print(f"  Shard {shard_idx+1}/{len(parquet_files)}: "
              f"+{shard_samples} samples → total {len(samples)}")

    return samples


class TadaburDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        attempts = 0
        cur = idx
        while attempts < 10:
            try:
                s = self.samples[cur]
                audio_array, _ = librosa.load(
                    io.BytesIO(s["audio_bytes"]),
                    sr=SAMPLE_RATE, mono=True
                )
                if len(audio_array) < SAMPLE_RATE * 0.5:
                    raise ValueError("Too short")

                input_features = feature_extractor(
                    audio_array,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="np"
                ).input_features[0]

                labels = tokenizer(
                    s["text"],
                    return_tensors="np"
                ).input_ids[0]

                return {"input_features": input_features, "labels": labels}

            except Exception:
                attempts += 1
                cur = (cur + 1) % len(self.samples)

        raise RuntimeError(f"Could not load sample near index {idx}")


# ── Load Data into RAM ────────────────────────────────────────────────────────

print("Loading training data into RAM...")
train_samples = load_shards_into_ram(
    DATASET_PATH,
    max_audio_secs=MAX_AUDIO_SECS,
    max_shards=MAX_SHARDS,
)
dataset = TadaburDataset(train_samples)
print(f"Train dataset: {len(dataset):,} samples in RAM")

print("Loading eval data into RAM...")
eval_samples = load_shards_into_ram(
    DATASET_PATH,
    max_audio_secs=MAX_AUDIO_SECS,
    max_shards=EVAL_SHARDS,
    max_samples=200,
)
eval_dataset = TadaburDataset(eval_samples)
print(f"Eval dataset: {len(eval_dataset):,} samples in RAM")

# ── Data Collator ─────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── Metric ────────────────────────────────────────────────────────────────────

wer_metric = evaluate_module.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    sample_path = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        for ref, pred in zip(label_str[:10], pred_str[:10]):
            f.write(f"Reference : {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("-" * 50 + "\n")

    return {"wer": wer}

# ── Load Model ────────────────────────────────────────────────────────────────

print("Loading whisper-medium...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language           = LANGUAGE
model.generation_config.task               = TASK
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

# ── Training Arguments ────────────────────────────────────────────────────────

print("Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,

    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,

    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",

    predict_with_generate=True,
    generation_max_length=255,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),

    dataloader_num_workers=0,
    push_to_hub=False,
)

# ── Trainer ───────────────────────────────────────────────────────────────────

print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# ── Train ─────────────────────────────────────────────────────────────────────

print("=" * 50)
print(f"Model      : {MODEL_NAME}")
print(f"Output     : {OUTPUT_DIR}")
print(f"Max steps  : {MAX_STEPS}")
print(f"Batch size : {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
print(f"FP16       : True")
print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Train size : {len(dataset):,} samples")
print(f"Eval size  : {len(eval_dataset):,} samples")
print("=" * 50)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
print("Done!")