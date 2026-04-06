# 🕌 Tadabur — Quran ASR System

> Fine-tuned Whisper Medium for Quran speech recognition, Surah/Ayah identification, and reciter recognition.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-rakansuliman%2Ftadabur--whisper--medium-blue)](https://huggingface.co/rakansuliman/tadabur-whisper-medium)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![CS465 ML Project](https://img.shields.io/badge/CS465-Machine%20Learning-green)]()

## Overview

Tadabur is an end-to-end Quran speech recognition pipeline that:

- 🎙️ **Transcribes** Quran recitations to Arabic text (6.26% WER)
- 📖 **Identifies** the Surah and Ayah being recited (up to 91.2% confidence)
- 👤 **Recognizes** the reciter from 335 supported reciters (98.47% accuracy)

Built on the [Tadabur dataset](https://huggingface.co/datasets/FaisaI/tadabur) — 1,400+ hours, 600+ reciters.

---

## Results

| Model | WER (%) | CER (%) |
|---|---|---|
| Whisper Medium (Vanilla) | 41.10% | 11.47% |
| Tadabur-Whisper-Small (Author) | 47.06% | 12.28% |
| **Ours: Whisper Medium Fine-tuned** | **6.26%** | **4.41%** |

| Metric | Value |
|---|---|
| Reciter classifier accuracy | 98.47% |
| Supported reciters | 335 |
| Surah/Ayah identification confidence | up to 91.2% |

---

## Architecture

```
Audio Input
    ↓
Whisper Encoder (frozen after ASR training)
    ├── Whisper Decoder    → Arabic transcription
    └── MLP Classifier     → Reciter name (335 classes)
    ↓
Fuzzy Matching (RapidFuzz)
    ↓
Surah + Ayah identification (6,236 ayahs)
```

---

## Repository Structure

```
tadabur-asr/
├── src/
│   ├── train.py              # Whisper fine-tuning
│   ├── train_reciter.py      # Reciter classifier training
│   ├── model_eval.py         # WER/CER evaluation
│   └── inference.py          # Gradio inference app
├── tadabur_eda.ipynb          # Exploratory data analysis
├── supported_reciters.txt     # All 335 supported reciters
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/rakansuliman/tadabur-asr
cd tadabur-asr
pip install -r requirements.txt
apt-get install -y ffmpeg
```

---

## Usage

### Run inference app locally

```bash
python src/inference.py
```

### Quick transcription

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, torch

processor = WhisperProcessor.from_pretrained("rakansuliman/tadabur-whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("rakansuliman/tadabur-whisper-medium")

audio, _ = librosa.load("recitation.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
with torch.no_grad():
    ids = model.generate(inputs, language="arabic", task="transcribe",
                         max_new_tokens=225, suppress_tokens=[], forced_decoder_ids=None)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

---

## Training

### 1. ASR Fine-tuning (Whisper)

```bash
# On RunPod RTX 4090, ~17 hours
python src/train.py
```

### 2. Reciter Classifier

```bash
# Two-phase: extract embeddings → train MLP
python src/train_reciter.py
```

### 3. Evaluation

```bash
python src/model_eval.py
```

---

## Requirements

```
torch
transformers
librosa
gradio
rapidfuzz
pyarrow
numpy
scipy
pandas
tqdm
evaluate
jiwer
soundfile
tensorboard
```

---

## Model Weights

Available at [huggingface.co/rakansuliman/tadabur-whisper-medium](https://huggingface.co/rakansuliman/tadabur-whisper-medium):

- `model.safetensors` — Fine-tuned Whisper Medium (3.06GB)
- `reciter_classifier.pt` — MLP reciter classifier (2.76MB)
- `reciter_idx_to_id.json` — Classifier index → reciter ID mapping
- `reciter_id_to_idx.json` — Reciter ID → classifier index mapping
- `sheikh_dict.json` — Reciter ID → Arabic name
- `surah_dict.json` — Surah index → Arabic name
- `supported_reciters.txt` — Full list of 335 supported reciters

---

## Citation

```bibtex
@misc{suliman2026tadabur,
  author = {Suliman, Rakan},
  title  = {Tadabur: Quran ASR with Surah/Ayah Identification and Reciter Recognition},
  year   = {2026},
  url    = {https://github.com/rakansuliman/tadabur-asr}
}
```

---

## License

CC BY-NC 4.0 — Research and educational use only.
Please engage with Quran content respectfully.

---

*CS465 Machine Learning Project — Spring 2026*