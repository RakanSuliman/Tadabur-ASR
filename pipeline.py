"""
pipeline.py
Full Tadabur ASR Pipeline — Training → Evaluation → Inference
Runs on RunPod with RTX 4090 + Secure Cloud Network Volume

Usage:
    python pipeline.py                # run full pipeline
    python pipeline.py --skip-train   # skip training, run eval + inference
    python pipeline.py --train-only   # training only
    python pipeline.py --eval-only    # evaluation only
    python pipeline.py --infer-only   # inference server only
    python pipeline.py --test-only    # run test pipeline only
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta

# ── Colors ────────────────────────────────────────────────────────────────────

class C:
    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE   = "\033[0;34m"
    CYAN   = "\033[0;36m"
    BOLD   = "\033[1m"
    NC     = "\033[0m"

def log(msg):     print(f"{C.BLUE}[{timestamp()}]{C.NC} {msg}")
def success(msg): print(f"{C.GREEN}[{timestamp()}] ✅ {msg}{C.NC}")
def warn(msg):    print(f"{C.YELLOW}[{timestamp()}] ⚠️  {msg}{C.NC}")
def error(msg):   print(f"{C.RED}[{timestamp()}] ❌ {msg}{C.NC}"); sys.exit(1)
def section(msg): print(f"\n{C.CYAN}{'═'*50}{C.NC}\n{C.CYAN}  {msg}{C.NC}\n{C.CYAN}{'═'*50}{C.NC}\n")
def timestamp():  return datetime.now().strftime("%H:%M:%S")

# ── Configuration ─────────────────────────────────────────────────────────────

WORKSPACE    = "/workspace"
CODE_DIR     = f"{WORKSPACE}/tadabur-asr"
DATASET_DIR  = f"{WORKSPACE}/tadabur/data"
MODEL_OUTPUT = f"{WORKSPACE}/whisper-medium-tadabur"
RESULTS_DIR  = f"{WORKSPACE}/results"
LOG_DIR      = f"{WORKSPACE}/logs"

# ── Argument Parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Tadabur ASR Pipeline")
parser.add_argument("--skip-train", action="store_true", help="Skip training step")
parser.add_argument("--train-only", action="store_true", help="Run training only")
parser.add_argument("--eval-only",  action="store_true", help="Run evaluation only")
parser.add_argument("--infer-only", action="store_true", help="Run inference server only")
parser.add_argument("--test-only",  action="store_true", help="Run test pipeline only")
args = parser.parse_args()

# ── Helper: Run a Python script with live output ──────────────────────────────

def run_script(script_path, log_file):
    """
    Runs a Python script, streams output live to terminal
    and also saves it to a log file simultaneously.
    Returns exit code.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log(f"Running: {script_path}")
    log(f"Log file: {log_file}")
    print()

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")   # live terminal output
            f.write(line)         # save to log file
            f.flush()

        process.wait()
        return process.returncode

# ── Helper: Format elapsed time ───────────────────────────────────────────────

def format_elapsed(seconds):
    return str(timedelta(seconds=int(seconds)))

# ── Step 0: Pre-flight Checks ─────────────────────────────────────────────────

def preflight():
    section("🔍 Pre-flight Checks")

    # Python version
    py_ver = sys.version.split()[0]
    success(f"Python {py_ver}")

    # Check CUDA + GPU
    try:
        import torch
        if not torch.cuda.is_available():
            error("CUDA not available — is the GPU pod running?")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        success(f"GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
    except ImportError:
        error("torch not installed — run: pip install torch")

    # Check required packages
    required = ["transformers", "accelerate", "evaluate",
                "jiwer", "librosa", "gradio", "rapidfuzz", "pandas",
                "pyarrow"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        error(f"Missing packages: {', '.join(missing)}\nRun: pip install {' '.join(missing)}")
    success(f"All {len(required)} required packages found")

    # Check dataset
    if not os.path.exists(DATASET_DIR):
        warn(f"Dataset not found at {DATASET_DIR}")
        log("Attempting to download from HuggingFace...")
        result = subprocess.run([
            "hf", "download", "FaisaI/tadabur",
            "--repo-type", "dataset",
            "--local-dir", f"{WORKSPACE}/tadabur",
            "--include", "data/*",
            "--include", "sheikh_dict.json",
            "--include", "surah_dict.json"
        ])
        if result.returncode != 0:
            error("Dataset download failed. Run: hf login")
        success("Dataset downloaded successfully")
    else:
        shards = [f for f in os.listdir(DATASET_DIR) if f.endswith(".parquet")]
        success(f"Dataset found: {len(shards)} parquet shards at {DATASET_DIR}")

    # Check code files
    for fname in ["train.py", "model_eval.py", "inference.py"]:
        fpath = os.path.join(CODE_DIR, "src", fname)
        if not os.path.exists(fpath):
            error(f"{fname} not found at {fpath} — upload your code files first")
    success("All code files found")

    # Create output directories
    for d in [MODEL_OUTPUT, RESULTS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    success("Output directories ready")

    # Summary
    print()
    log("Pipeline configuration:")
    log(f"  Dataset    : {DATASET_DIR}")
    log(f"  Model out  : {MODEL_OUTPUT}")
    log(f"  Results    : {RESULTS_DIR}")
    log(f"  Logs       : {LOG_DIR}")
    print()

# ── Step 1: Test Pipeline ─────────────────────────────────────────────────────

def run_tests():
    section("🧪 Step 1: Test Pipeline")

    test_script = os.path.join(CODE_DIR, "test_pipeline.py")
    if not os.path.exists(test_script):
        warn("test_pipeline.py not found — skipping tests")
        return

    log_file = os.path.join(LOG_DIR, "test_pipeline.log")
    exit_code = run_script(test_script, log_file)

    # Check results
    with open(log_file) as f:
        content = f.read()

    if "All tests passed" in content:
        success("All tests passed — safe to proceed")
    else:
        failed = content.count("FAIL")
        warn(f"{failed} test(s) failed — check {log_file}")
        answer = input("Continue anyway? (y/n): ").strip().lower()
        if answer != "y":
            error("Aborted by user")

# ── Step 2: Training ──────────────────────────────────────────────────────────

def run_training():
    section("🏋️  Step 2: Training whisper-medium")

    log("Starting training — whisper-medium on RTX 4090")
    log(f"Checkpoints saved every 1000 steps to {MODEL_OUTPUT}")
    log(f"Training log: {LOG_DIR}/train.log")
    log("To monitor in another terminal: tail -f /workspace/logs/train.log")
    print()

    start = time.time()
    log_file = os.path.join(LOG_DIR, "train.log")
    exit_code = run_script(os.path.join(CODE_DIR, "src", "train.py"), log_file)
    elapsed = format_elapsed(time.time() - start)

    if exit_code == 0:
        success(f"Training completed in {elapsed}")
        success(f"Model saved to {MODEL_OUTPUT}")
    else:
        error(f"Training failed after {elapsed} — check {log_file}")

# ── Step 3: Evaluation ────────────────────────────────────────────────────────

def run_evaluation():
    section("📊 Step 3: Evaluation")

    # Warn if fine-tuned model missing
    if not os.path.exists(MODEL_OUTPUT) or not os.listdir(MODEL_OUTPUT):
        warn(f"Fine-tuned model not found at {MODEL_OUTPUT}")
        warn("Will evaluate vanilla Whisper + author's model only")

    log("Running evaluation on 500 test samples...")
    log(f"Evaluation log: {LOG_DIR}/evaluate.log")
    print()

    start = time.time()
    log_file = os.path.join(LOG_DIR, "evaluate.log")
    exit_code = run_script(os.path.join(CODE_DIR, "src", "model_eval.py"), log_file)
    elapsed = format_elapsed(time.time() - start)

    if exit_code == 0:
        success(f"Evaluation completed in {elapsed}")

        # Print results table
        csv_path = os.path.join(RESULTS_DIR, "wer_comparison.csv")
        if os.path.exists(csv_path):
            print()
            print(f"{C.CYAN}{'━'*50}{C.NC}")
            print(f"{C.CYAN}  WER COMPARISON TABLE (for IEEE paper){C.NC}")
            print(f"{C.CYAN}{'━'*50}{C.NC}")
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                print(df[["Model", "WER (%)", "CER (%)"]].to_string(index=False))
            except Exception as e:
                warn(f"Could not display table: {e}")
            print(f"{C.CYAN}{'━'*50}{C.NC}")
            success(f"Results saved to {csv_path}")
    else:
        error(f"Evaluation failed — check {log_file}")

# ── Step 4: Inference Server ──────────────────────────────────────────────────

def run_inference():
    section("🎙️  Step 4: Inference Server")

    log("Starting Gradio server on port 7860")
    log("Access via: RunPod dashboard → Connect → HTTP Service → Port 7860")
    log("Press Ctrl+C to stop")
    print()

    log_file = os.path.join(LOG_DIR, "inference.log")
    exit_code = run_script(os.path.join(CODE_DIR, "src", "inference.py"), log_file)

    if exit_code != 0:
        error(f"Inference server stopped — check {log_file}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ████████╗ █████╗ ██████╗  █████╗ ██████╗ ██╗   ██╗██████╗ ")
    print("     ██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔══██╗")
    print("     ██║   ███████║██║  ██║███████║██████╔╝██║   ██║██████╔╝")
    print("     ██║   ██╔══██║██║  ██║██╔══██║██╔══██╗██║   ██║██╔══██╗")
    print("     ██║   ██║  ██║██████╔╝██║  ██║██████╔╝╚██████╔╝██║  ██║")
    print("     ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝")
    print(f"  Quranic ASR Pipeline — CS465 ML Project{C.NC}\n")

    # Pre-flight always runs
    preflight()

    # Route based on arguments
    if args.test_only:
        run_tests()

    elif args.train_only:
        run_tests()
        run_training()

    elif args.eval_only:
        run_evaluation()

    elif args.infer_only:
        run_inference()

    elif args.skip_train:
        run_evaluation()
        run_inference()

    else:
        # Full pipeline
        run_tests()
        run_training()
        run_evaluation()
        run_inference()

    # Final summary
    section("🎉 Pipeline Complete")
    success("All steps finished successfully")
    print()
    log("Output summary:")
    log(f"  Trained model : {MODEL_OUTPUT}")
    log(f"  Results CSV   : {RESULTS_DIR}/wer_comparison.csv")
    log(f"  All logs      : {LOG_DIR}/")
    print()


if __name__ == "__main__":
    main()