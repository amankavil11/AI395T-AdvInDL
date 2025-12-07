import argparse
import zipfile
from pathlib import Path

BLACKLIST = ["__pycache__", ".pyc", ".ipynb"]
# Exclude checkpoint-related files and training artifacts
CHECKPOINT_BLACKLIST = ["optimizer.pt", "scheduler.pt", "rng_state.pth", 
                        "trainer_state.json", "training_args.bin", 
                        "events.out.tfevents"]
MAXSIZE_MB = 40


def bundle(homework_dir: str, utid: str):
    """
    Usage: python3 bundle.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        # Skip if matches base blacklist
        if any(b in str(f) for b in BLACKLIST):
            continue
        
        # Skip checkpoint directories entirely
        if "checkpoint-" in f.name or "tensorboard" in f.name:
            continue
        
        # Skip checkpoint-related files (optimizer, scheduler, etc.)
        if any(b in f.name for b in CHECKPOINT_BLACKLIST):
            continue
        
        # Only include files, not directories
        if f.is_file():
            files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))

    # Zip all files, keeping the directory structure
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, homework_dir.stem / f.relative_to(homework_dir))

    output_size_mb = output_path.stat().st_size / 1024 / 1024

    if output_size_mb > MAXSIZE_MB:
        print("Warning: The created zip file is larger than expected!")

    print(f"Submission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("homework")
    parser.add_argument("utid")

    args = parser.parse_args()

    bundle(args.homework, args.utid)
