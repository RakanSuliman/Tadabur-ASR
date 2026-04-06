"""
test_pipeline.py
Tests the full pipeline locally BEFORE running on RunPod.
Uses the preview.parquet file (already downloaded) — no GPU needed.

Run this on your local machine first to catch any errors cheaply.

Usage:
    python test_pipeline.py

What it tests:
    1. Dependencies     — all required packages installed
    2. Data loading     — preview.parquet loads correctly
    3. Preprocessing    — audio extraction + feature extraction works
    4. Model loading    — Whisper loads (CPU, no training)
    5. Transcription    — single sample transcription works
    6. Surah matching   — fuzzy matching works
    7. Data collator    — batching works correctly
    8. Evaluate imports — evaluate.py imports work
    9. Inference imports — inference.py imports work
"""

import os
import sys
import time
import traceback

# ── Test Runner ───────────────────────────────────────────────────────────────

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SKIP  = "⚠️  SKIP"

results = []

def test(name, fn):
    print(f"\n{'─'*60}")
    print(f"Testing: {name}")
    start = time.time()
    try:
        msg = fn()
        elapsed = time.time() - start
        print(f"{PASS} — {msg} ({elapsed:.2f}s)")
        results.append((name, True, msg))
    except Exception as e:
        elapsed = time.time() - start
        print(f"{FAIL} — {e} ({elapsed:.2f}s)")
        traceback.print_exc()
        results.append((name, False, str(e)))

# ── Test 1: Dependencies ──────────────────────────────────────────────────────

def test_dependencies():
    import importlib
    required = [
        "transformers", "datasets", "torch", "librosa",
        "numpy", "pandas", "soundfile", "evaluate",
        "jiwer", "rapidfuzz", "gradio", "tqdm", "accelerate"
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise ImportError(f"Missing packages: {', '.join(missing)}\nRun: pip install {' '.join(missing)}")

    import torch
    cuda = "CUDA available" if torch.cuda.is_available() else "CPU only (expected locally)"
    return f"All {len(required)} packages found. {cuda}"

test("1. Dependencies", test_dependencies)

# ── Test 2: Preview Parquet Loading ──────────────────────────────────────────

def test_data_loading():
    import pandas as pd

    # Try to find preview.parquet
    possible_paths = [
        "preview.parquet",
        "../preview.parquet",
        os.path.join(os.path.dirname(__file__), "preview.parquet"),
        os.path.join(os.path.dirname(__file__), "..", "preview.parquet"),
    ]

    preview_path = None
    for path in possible_paths:
        if os.path.exists(path):
            preview_path = path
            break

    if not preview_path:
        raise FileNotFoundError(
            "preview.parquet not found. Download from:\n"
            "https://huggingface.co/datasets/FaisaI/tadabur/resolve/main/preview.parquet\n"
            "and place it in the same folder as this script."
        )

    df = pd.read_parquet(preview_path)

    required_cols = ["reciter_id", "surah_id", "ayah_id", "ayah_duration_s",
                     "text_ar_simple", "text_ar_uthmani", "metadata", "audio"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Store for later tests
    global PREVIEW_DF, PREVIEW_PATH
    PREVIEW_DF  = df
    PREVIEW_PATH = preview_path

    return f"Loaded {len(df)} samples, {len(df.columns)} columns from {preview_path}"

test("2. Preview parquet loading", test_data_loading)

# ── Test 3: Audio Extraction ──────────────────────────────────────────────────

def test_audio_extraction():
    import io
    import json
    import librosa
    import numpy as np

    if "PREVIEW_DF" not in globals():
        raise RuntimeError("Skipping — data not loaded (Test 2 failed)")

    row = PREVIEW_DF.iloc[0]

    # Extract audio bytes
    audio_bytes = row["audio"]["bytes"]
    if audio_bytes is None or len(audio_bytes) == 0:
        raise ValueError("Audio bytes are empty")

    # Decode audio
    audio_array, sr = librosa.load(
        io.BytesIO(audio_bytes), sr=16000, mono=True
    )

    if len(audio_array) == 0:
        raise ValueError("Audio array is empty after decoding")

    duration = len(audio_array) / 16000

    # Extract metadata
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {}
    word_alignments = meta.get("word_alignments", [])

    # Store sample for later tests
    global AUDIO_SAMPLE
    AUDIO_SAMPLE = {
        "audio"     : audio_array,
        "sr"        : sr,
        "text"      : row["text_ar_uthmani"],
        "surah_id"  : int(row["surah_id"]),
        "ayah_id"   : int(row["ayah_id"]),
        "reciter_id": int(row["reciter_id"]),
        "duration"  : float(row["ayah_duration_s"]),
    }

    return (f"Audio decoded: {duration:.2f}s, shape={audio_array.shape}, "
            f"sr={sr}Hz, {len(word_alignments)} word alignments")

test("3. Audio extraction", test_audio_extraction)

# ── Test 4: Feature Extraction ────────────────────────────────────────────────

def test_feature_extraction():
    from transformers import WhisperFeatureExtractor

    if "AUDIO_SAMPLE" not in globals():
        raise RuntimeError("Skipping — audio not loaded (Test 3 failed)")

    # Use small model for speed
    print("  (Downloading whisper-small feature extractor — ~1MB)")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    features = feature_extractor(
        AUDIO_SAMPLE["audio"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    shape = features.input_features.shape
    expected = (1, 80, 3000)  # (batch, mel_bins, time_frames)

    if shape != expected:
        raise ValueError(f"Unexpected feature shape: {shape}, expected {expected}")

    global FEATURE_EXTRACTOR
    FEATURE_EXTRACTOR = feature_extractor

    return f"Features shape: {shape} (correct)"

test("4. Feature extraction (Whisper)", test_feature_extraction)

# ── Test 5: Model Loading + Transcription (whisper-small for speed) ───────────

def test_transcription():
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    if "AUDIO_SAMPLE" not in globals():
        raise RuntimeError("Skipping — audio not loaded (Test 3 failed)")

    print("  (Loading whisper-small for local test — ~500MB)")
    print("  (On RunPod we use whisper-large-v3 — this is just a local sanity check)")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="arabic", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()

    inputs = processor(
        AUDIO_SAMPLE["audio"],
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(inputs, language="arabic", task="transcribe")

    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    reference = AUDIO_SAMPLE["text"]

    if not transcription:
        raise ValueError("Transcription is empty")

    # Store for matching test
    global TRANSCRIPTION
    TRANSCRIPTION = transcription

    print(f"  Reference   : {reference[:60]}...")
    print(f"  Transcribed : {transcription[:60]}...")

    return f"Transcription successful ({len(transcription)} chars)"

test("5. Model loading + transcription", test_transcription)

# ── Test 6: Surah/Ayah Matching ───────────────────────────────────────────────

def test_surah_matching():
    from rapidfuzz import fuzz

    if "PREVIEW_DF" not in globals():
        raise RuntimeError("Skipping — data not loaded (Test 2 failed)")

    if "TRANSCRIPTION" not in globals():
        # Use a known ayah text as fallback
        transcription = PREVIEW_DF.iloc[0]["text_ar_simple"]
    else:
        transcription = TRANSCRIPTION

    # Build mini lookup from preview data
    results = []
    for _, row in PREVIEW_DF.iterrows():
        score = fuzz.partial_ratio(transcription, row["text_ar_simple"])
        results.append({
            "surah_id"  : row["surah_id"],
            "ayah_id"   : row["ayah_id"],
            "score"     : score,
            "text"      : row["text_ar_uthmani"][:40],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[0]

    return (f"Top match: Surah {top['surah_id']}, "
            f"Ayah {top['ayah_id']} "
            f"(score={top['score']}%)")

test("6. Surah/Ayah fuzzy matching", test_surah_matching)

# ── Test 7: Data Collator ─────────────────────────────────────────────────────

def test_data_collator():
    import io
    import json
    import torch
    import librosa
    import numpy as np
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from transformers import WhisperProcessor

    if "PREVIEW_DF" not in globals():
        raise RuntimeError("Skipping — data not loaded (Test 2 failed)")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="arabic", task="transcribe"
    )

    # Build 2 fake features to test collation
    features = []
    for i in range(min(2, len(PREVIEW_DF))):
        row = PREVIEW_DF.iloc[i]
        audio_array, _ = librosa.load(
            io.BytesIO(row["audio"]["bytes"]), sr=16000, mono=True
        )
        input_features = processor(
            audio_array, sampling_rate=16000, return_tensors="np"
        ).input_features[0]

        labels = processor.tokenizer(
            row["text_ar_uthmani"], return_tensors="np"
        ).input_ids[0]

        features.append({
            "input_features": input_features,
            "labels"        : labels,
        })

    # Test collation
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    batch = collator(features)

    expected_keys = {"input_features", "labels"}
    if not expected_keys.issubset(batch.keys()):
        raise ValueError(f"Missing keys in batch: {expected_keys - batch.keys()}")

    return (f"Batch keys: {list(batch.keys())}, "
            f"input shape: {batch['input_features'].shape}, "
            f"labels shape: {batch['labels'].shape}")

test("7. Data collator", test_data_collator)

# ── Test 8: WER Metric ────────────────────────────────────────────────────────

def test_wer_metric():
    import evaluate as ev

    wer_metric = ev.load("wer")

    predictions = ["بسم الله الرحمن", "قل هو الله احد"]
    references  = ["بِسْمِ اللَّهِ الرَّحْمَٰنِ", "قُلْ هُوَ اللَّهُ أَحَدٌ"]

    wer = wer_metric.compute(predictions=predictions, references=references)

    if not (0 <= wer <= 1):
        raise ValueError(f"WER out of range: {wer}")

    return f"WER metric works, sample WER = {wer:.4f}"

test("8. WER metric", test_wer_metric)

# ── Test 9: Gradio Interface ──────────────────────────────────────────────────

def test_gradio():
    import gradio as gr
    return f"Gradio version {gr.__version__} available"

test("9. Gradio import", test_gradio)

# ── Test 10: Sheikh Dict ──────────────────────────────────────────────────────

def test_sheikh_dict():
    possible_paths = [
        "sheikh_dict.json",
        "../sheikh_dict.json",
        os.path.join(os.path.dirname(__file__), "sheikh_dict.json"),
        os.path.join(os.path.dirname(__file__), "..", "sheikh_dict.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"Loaded {len(data)} reciters from {path}"

    raise FileNotFoundError(
        "sheikh_dict.json not found. Download from:\n"
        "https://huggingface.co/datasets/FaisaI/tadabur/resolve/main/sheikh_dict.json"
    )

import json
test("10. Sheikh dict (reciter names)", test_sheikh_dict)

# ── Final Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

for name, ok, msg in results:
    status = PASS if ok else FAIL
    print(f"{status} {name}")

print("=" * 60)
print(f"Passed: {passed}/{len(results)}")

if failed == 0:
    print("\n🎉 All tests passed! Safe to run on RunPod.")
else:
    print(f"\n⚠️  {failed} test(s) failed. Fix errors before running on RunPod.")
    print("\nFailed tests:")
    for name, ok, msg in results:
        if not ok:
            print(f"  ❌ {name}: {msg}")