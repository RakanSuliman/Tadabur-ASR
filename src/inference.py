import torch
import librosa
import io
import pyarrow.parquet as pq
import glob
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load your fine-tuned model
processor = WhisperProcessor.from_pretrained("/workspace/whisper-medium-tadabur")
model = WhisperForConditionalGeneration.from_pretrained(
    "/workspace/whisper-medium-tadabur",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load 5 samples from shard 770 (unseen)
pf = pq.ParquetFile(sorted(glob.glob("/workspace/tadabur/data/*.parquet"))[-1])
samples = []
for batch in pf.iter_batches(batch_size=64, columns=["audio", "text_ar_uthmani", "ayah_duration_s"]):
    for i in range(len(batch)):
        if 0 < batch.column("ayah_duration_s")[i].as_py() <= 30:
            audio_bytes = batch.column("audio")[i]["bytes"].as_py()
            audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            samples.append({
                "audio": audio_array,
                "reference": batch.column("text_ar_uthmani")[i].as_py()
            })
        if len(samples) >= 5:
            break
    if len(samples) >= 5:
        break

# Transcribe each sample
for i, s in enumerate(samples):
    inputs = processor(s["audio"], sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
    with torch.no_grad():
        pred_ids = model.generate(inputs, language="arabic", task="transcribe", max_new_tokens=225)
    transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    print(f"Sample {i+1}")
    print(f"  Reference   : {s['reference']}")
    print(f"  Transcribed : {transcription}")
    print()