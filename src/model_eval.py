"""
model_eval.py — Evaluates and compares ASR models on the Tadabur dataset.
Generates wer_comparison.csv for the project report.

Models evaluated:
    1. openai/whisper-medium           (vanilla, no fine-tuning)
    2. /workspace/whisper-medium-tadabur   (our fine-tuned model)
    3. FaisaI/tadabur-Whisper-Small    (dataset author's baseline)

Dataset I/O uses the same pyarrow lazy-read approach as train.py to avoid
ArrowNotImplementedError on nested audio columns.

Usage:
    python model_eval.py
"""

import os
import io
import glob
import re
import unicodedata

import numpy as np
import torch
import librosa
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate as evaluate_module

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = "/workspace/tadabur/data"
SAMPLE_RATE  = 16_000
NUM_SAMPLES  = 500
RESULTS_DIR  = "/workspace/results"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Whisper-Medium (Vanilla)",
        "path": "openai/whisper-medium",
        "local": False,
    },
    {
        "name": "Whisper-Medium (Fine-tuned)",
        "path": "/workspace/whisper-medium-tadabur",
        "local": True,
    },
    {
        "name": "Tadabur-Whisper-Small (Author)",
        "path": "FaisaI/tadabur-Whisper-Small",
        "local": False,
    },
]

# ── Arabic normaliser (same as train.py) ───────────────────────────────────────

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]"
)
_TATWEEL = re.compile(r"\u0640")


def normalise_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Load test samples via pyarrow (no pd.read_parquet on audio) ────────────────

def load_test_samples(dataset_path: str, num_samples: int) -> list:
    """Read test samples using pyarrow iter_batches, extracting audio as raw
    scalars to bypass the nested-chunked Arrow error."""

    parquet_files = sorted(glob.glob(f"{dataset_path}/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {dataset_path}")

    # Use the last shard as held-out test data (matches train.py split).
    test_shard = parquet_files[-1]
    print(f"Loading test samples from {test_shard} …")

    samples = []
    pf = pq.ParquetFile(test_shard)
    for batch in pf.iter_batches(
        batch_size=64,
        columns=["audio", "text_ar_uthmani", "text_ar_simple",
                 "ayah_duration_s", "reciter_id", "surah_id", "ayah_id"],
    ):
        durations  = batch.column("ayah_duration_s")
        audio_col  = batch.column("audio")
        text_col   = batch.column("text_ar_uthmani")
        simple_col = batch.column("text_ar_simple")
        reciter_col = batch.column("reciter_id")
        surah_col  = batch.column("surah_id")
        ayah_col   = batch.column("ayah_id")

        for i in range(len(batch)):
            dur = durations[i].as_py()
            if not (0 < dur <= 30):
                continue
            try:
                audio_bytes = audio_col[i]["bytes"].as_py()
                if not audio_bytes:
                    continue
                audio_array, _ = librosa.load(
                    io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True,
                )
                samples.append({
                    "audio": audio_array,
                    "reference": text_col[i].as_py(),
                    "reference_simple": simple_col[i].as_py(),
                    "reciter_id": reciter_col[i].as_py(),
                    "surah_id": surah_col[i].as_py(),
                    "ayah_id": ayah_col[i].as_py(),
                    "duration": dur,
                })
            except Exception:
                continue

            if len(samples) >= num_samples:
                break

        if len(samples) >= num_samples:
            break

    print(f"Loaded {len(samples)} test samples")
    return samples


# ── Evaluate a single model ────────────────────────────────────────────────────

def evaluate_model(model_cfg: dict, test_samples: list) -> dict | None:
    name = model_cfg["name"]
    path = model_cfg["path"]

    if model_cfg["local"] and not os.path.exists(path):
        print(f"Skipping {name} — not found at {path}")
        return None

    print(f"\nEvaluating {name} …")
    try:
        proc = WhisperProcessor.from_pretrained(path)
        mdl = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto",
        )
        mdl.eval()
    except Exception as exc:
        print(f"  Failed to load: {exc}")
        return None

    wer_metric = evaluate_module.load("wer")
    cer_metric = evaluate_module.load("cer")

    predictions, references, rows = [], [], []

    for sample in tqdm(test_samples, desc=name):
        try:
            inputs = proc(
                sample["audio"], sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features.to(DEVICE).half()

            with torch.no_grad():
                    pred_ids = mdl.generate(
                    inputs,
                    language="arabic",
                    task="transcribe",
                    max_new_tokens=225,
                    suppress_tokens=[],
                    forced_decoder_ids=None,
                    )
            transcript = proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            predictions.append(normalise_arabic(transcript))
            references.append(normalise_arabic(sample["reference"]))
            rows.append({
                "surah_id": sample["surah_id"],
                "ayah_id": sample["ayah_id"],
                "reciter_id": sample["reciter_id"],
                "duration": sample["duration"],
                "reference": sample["reference"],
                "prediction": transcript,
            })
        except Exception as exc:
            print(f"  Error on {sample['surah_id']}:{sample['ayah_id']}: {exc}")
            continue

    if not predictions:
        print("  No predictions generated.")
        del mdl
        torch.cuda.empty_cache()
        return None

    overall_wer = wer_metric.compute(predictions=predictions, references=references)
    overall_cer = cer_metric.compute(predictions=predictions, references=references)
    print(f"  Overall WER: {overall_wer*100:.2f}%  CER: {overall_cer*100:.2f}%")

    df = pd.DataFrame(rows)

    # Per-surah WER
    wer_by_surah = {}
    for sid, grp in df.groupby("surah_id"):
        preds_g = [normalise_arabic(p) for p in grp["prediction"]]
        refs_g = [normalise_arabic(r) for r in grp["reference"]]
        wer_by_surah[sid] = wer_metric.compute(predictions=preds_g, references=refs_g)

    # Per-reciter WER (≥5 samples)
    wer_by_reciter = {}
    for rid, grp in df.groupby("reciter_id"):
        if len(grp) < 5:
            continue
        preds_g = [normalise_arabic(p) for p in grp["prediction"]]
        refs_g = [normalise_arabic(r) for r in grp["reference"]]
        wer_by_reciter[rid] = wer_metric.compute(predictions=preds_g, references=refs_g)

    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    df.to_csv(os.path.join(RESULTS_DIR, f"{safe_name}_predictions.csv"),
              index=False, encoding="utf-8")

    del mdl
    torch.cuda.empty_cache()

    return {
        "model_name": name,
        "model_path": path,
        "wer": round(overall_wer, 4),
        "wer_pct": round(overall_wer * 100, 2),
        "cer": round(overall_cer, 4),
        "cer_pct": round(overall_cer * 100, 2),
        "num_samples": len(predictions),
        "wer_by_surah": wer_by_surah,
        "wer_by_reciter": wer_by_reciter,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Tadabur ASR — Model Evaluation")
print(f"Device      : {DEVICE}")
print(f"Test samples: {NUM_SAMPLES}")
print("=" * 60)

test_samples = load_test_samples(DATASET_PATH, NUM_SAMPLES)

results = []
for cfg in MODELS:
    result = evaluate_model(cfg, test_samples)
    if result is not None:
        results.append(result)

if not results:
    print("No models could be evaluated.")
    raise SystemExit(1)

# ── Save comparison table ──────────────────────────────────────────────────────

comparison = pd.DataFrame([{
    "Model": r["model_name"],
    "WER": r["wer"],
    "WER (%)": r["wer_pct"],
    "CER": r["cer"],
    "CER (%)": r["cer_pct"],
    "Samples": r["num_samples"],
} for r in results])

csv_path = os.path.join(RESULTS_DIR, "wer_comparison.csv")
comparison.to_csv(csv_path, index=False, encoding="utf-8")

# Per-surah breakdown
surah_rows = []
for r in results:
    for sid, wer_val in r["wer_by_surah"].items():
        surah_rows.append({"Model": r["model_name"], "surah_id": sid,
                           "WER": round(wer_val, 4)})
if surah_rows:
    pd.DataFrame(surah_rows).to_csv(
        os.path.join(RESULTS_DIR, "wer_by_surah.csv"),
        index=False, encoding="utf-8",
    )

# Per-reciter breakdown
reciter_rows = []
for r in results:
    for rid, wer_val in r["wer_by_reciter"].items():
        reciter_rows.append({"Model": r["model_name"], "reciter_id": rid,
                             "WER": round(wer_val, 4)})
if reciter_rows:
    pd.DataFrame(reciter_rows).to_csv(
        os.path.join(RESULTS_DIR, "wer_by_reciter.csv"),
        index=False, encoding="utf-8",
    )

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(comparison.to_string(index=False))
print("=" * 60)

best = comparison.loc[comparison["WER"].idxmin()]
print(f"\nBest model: {best['Model']}  WER = {best['WER (%)']}%")
print(f"Results saved to {RESULTS_DIR}/")
