"""
train_reciter.py
Trains a reciter identification classifier on top of frozen Whisper encoder.

Two-phase approach:
    Phase 1 — Extract embeddings (shard by shard, low RAM):
        Read one shard at a time → extract Whisper encoder embeddings
        → save to /workspace/reciter-embeddings/*.npz
        RAM usage: ~2GB max at any point

    Phase 2 — Train classifier (from saved embeddings):
        Load embeddings from disk (800MB total)
        Train linear classifier: 1024 → 512 → num_reciters
        Fast, no audio in RAM during training

Usage:
    python train_reciter.py
"""

import os
import io
import glob
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

WHISPER_MODEL_PATH = "/workspace/whisper-medium-tadabur"
DATASET_PATH       = "/workspace/tadabur/data"
OUTPUT_DIR         = "/workspace/reciter-classifier"
EMBEDDINGS_DIR     = "/workspace/reciter-embeddings"

SAMPLE_RATE        = 16000
MAX_AUDIO_SECS     = 30
MIN_SAMPLES        = 50
MAX_SHARDS         = 500
BATCH_SIZE         = 256   # large batch — embeddings fit in RAM easily
EPOCHS             = 20
LEARNING_RATE      = 1e-3
HIDDEN_DIM         = 1024  # whisper-medium encoder hidden size
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── Step 1: Count samples per reciter ─────────────────────────────────────────

print("=" * 60)
print("Step 1: Counting samples per reciter...")
print("=" * 60)

parquet_files = sorted(glob.glob(f"{DATASET_PATH}/*.parquet"))[:MAX_SHARDS]
reciter_counts = Counter()

for pf_path in parquet_files:
    pf = pq.ParquetFile(pf_path)
    for batch in pf.iter_batches(batch_size=512, columns=["reciter_id"]):
        for i in range(len(batch)):
            rid = batch.column("reciter_id")[i].as_py()
            reciter_counts[rid] += 1

valid_reciters    = sorted([rid for rid, c in reciter_counts.items() if c >= MIN_SAMPLES])
reciter_id_to_idx = {rid: idx for idx, rid in enumerate(valid_reciters)}
reciter_idx_to_id = {idx: rid  for idx, rid in enumerate(valid_reciters)}
num_reciters      = len(valid_reciters)

print(f"Total reciters       : {len(reciter_counts)}")
print(f"Valid reciters (100+): {num_reciters}")

with open(os.path.join(OUTPUT_DIR, "reciter_id_to_idx.json"), "w") as f:
    json.dump({str(k): v for k, v in reciter_id_to_idx.items()}, f)
with open(os.path.join(OUTPUT_DIR, "reciter_idx_to_id.json"), "w") as f:
    json.dump({str(k): v for k, v in reciter_idx_to_id.items()}, f)

print("Mappings saved.")

# ── Step 2: Load Whisper encoder ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2: Loading Whisper encoder (frozen)...")
print("=" * 60)

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
whisper   = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
whisper.eval()
for param in whisper.parameters():
    param.requires_grad = False

print(f"Whisper loaded and frozen on {DEVICE}")

# ── Step 3: Extract embeddings shard by shard ─────────────────────────────────

print("\n" + "=" * 60)
print("Step 3: Extracting embeddings (shard by shard, low RAM)...")
print("=" * 60)

total_extracted = 0

for shard_idx, pf_path in enumerate(parquet_files):
    shard_name    = os.path.basename(pf_path).replace(".parquet", "")
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"{shard_name}.npz")

    # Skip if already extracted
    if os.path.exists(embedding_path):
        data = np.load(embedding_path)
        total_extracted += len(data["embeddings"])
        print(f"  Shard {shard_idx+1}/{len(parquet_files)}: "
              f"{shard_name} already extracted ({len(data['embeddings'])} samples)")
        continue

    shard_embeddings = []
    shard_labels     = []

    try:
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=32):
            reciter_col  = batch.column("reciter_id")
            duration_col = batch.column("ayah_duration_s")
            audio_col    = batch.column("audio")

            for i in range(len(batch)):
                rid      = reciter_col[i].as_py()
                duration = duration_col[i].as_py()

                if rid not in reciter_id_to_idx:
                    continue
                if not (0 < duration <= MAX_AUDIO_SECS):
                    continue

                try:
                    audio_bytes = audio_col[i]["bytes"].as_py()
                    if not audio_bytes:
                        continue

                    # Decode audio
                    audio_array, _ = librosa.load(
                        io.BytesIO(audio_bytes),
                        sr=SAMPLE_RATE, mono=True
                    )
                    if len(audio_array) < SAMPLE_RATE * 0.5:
                        continue

                    # Extract features
                    input_features = processor(
                        audio_array,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt"
                    ).input_features.to(DEVICE).half()

                    # Extract embedding from frozen Whisper encoder
                    with torch.no_grad():
                        encoder_out = whisper.model.encoder(input_features)
                        embedding   = encoder_out.last_hidden_state.mean(dim=1)
                        embedding   = embedding.float().cpu().numpy()[0]  # (1024,)

                    shard_embeddings.append(embedding)
                    shard_labels.append(reciter_id_to_idx[rid])

                except Exception:
                    continue

    except Exception as e:
        print(f"  Warning: could not read {pf_path}: {e}")
        continue

    # Save shard embeddings to disk
    if shard_embeddings:
        np.savez(
            embedding_path,
            embeddings=np.array(shard_embeddings, dtype=np.float32),
            labels=np.array(shard_labels, dtype=np.int64),
        )
        total_extracted += len(shard_embeddings)
        print(f"  Shard {shard_idx+1}/{len(parquet_files)}: "
              f"{shard_name} → {len(shard_embeddings)} embeddings saved "
              f"(total: {total_extracted})")
    else:
        print(f"  Shard {shard_idx+1}/{len(parquet_files)}: {shard_name} — no valid samples")

print(f"\nTotal embeddings extracted: {total_extracted}")

# ── Step 4: Load all embeddings into RAM ──────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4: Loading embeddings into RAM for training...")
print("=" * 60)

all_embeddings = []
all_labels     = []

for npz_path in sorted(glob.glob(f"{EMBEDDINGS_DIR}/*.npz")):
    data = np.load(npz_path)
    all_embeddings.append(data["embeddings"])
    all_labels.append(data["labels"])

all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 1024)
all_labels     = np.concatenate(all_labels,     axis=0)  # (N,)

print(f"Embeddings shape : {all_embeddings.shape}")
print(f"Labels shape     : {all_labels.shape}")
print(f"RAM used         : ~{all_embeddings.nbytes / 1e6:.0f} MB")
print(f"Unique reciters  : {len(np.unique(all_labels))}")

# Convert to tensors
X = torch.tensor(all_embeddings, dtype=torch.float32)
y = torch.tensor(all_labels,     dtype=torch.long)

# Free numpy arrays
del all_embeddings, all_labels

# Train/val split
N         = len(X)
val_size  = int(0.1 * N)
idx       = torch.randperm(N)
train_idx = idx[val_size:]
val_idx   = idx[:val_size]

train_ds = TensorDataset(X[train_idx], y[train_idx])
val_ds   = TensorDataset(X[val_idx],   y[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Step 5: Train classifier ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 5: Training reciter classifier...")
print("=" * 60)

class ReciterClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


classifier = ReciterClassifier(HIDDEN_DIM, num_reciters).to(DEVICE)
optimizer  = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion  = nn.CrossEntropyLoss()

print(f"Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")
print(f"Classes          : {num_reciters}")
print(f"Epochs           : {EPOCHS}")
print(f"Batch size       : {BATCH_SIZE}")
print("=" * 60)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    classifier.train()
    train_loss = train_correct = train_total = 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = classifier(X_batch)
        loss   = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        train_total   += y_batch.size(0)

    # Validate
    classifier.eval()
    val_correct = val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = classifier(X_batch)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
            val_total   += y_batch.size(0)

    train_acc = train_correct / train_total * 100
    val_acc   = val_correct   / val_total   * 100
    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train acc: {train_acc:.2f}% | "
          f"Val acc: {val_acc:.2f}% | "
          f"Loss: {train_loss/len(train_loader):.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            classifier.state_dict(),
            os.path.join(OUTPUT_DIR, "reciter_classifier.pt")
        )
        print(f"  ✅ Best model saved ({val_acc:.2f}%)")

print("\n" + "=" * 60)
print(f"Training complete!")
print(f"Best validation accuracy : {best_val_acc:.2f}%")
print(f"Classifier saved to      : {OUTPUT_DIR}/reciter_classifier.pt")
print(f"Mappings saved to        : {OUTPUT_DIR}/")
print("=" * 60)