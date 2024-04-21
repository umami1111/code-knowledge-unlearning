import json
from pathlib import Path
import sys

assert len(sys.argv) >= 5

prompt_dir = Path(sys.argv[1])
src_dir = Path(sys.argv[2])
ext = sys.argv[3]
trg_path = Path(sys.argv[4])
num = int(sys.argv[5]) if len(sys.argv) >= 6 else None

assert prompt_dir.exists()
assert src_dir.exists()

trg_path.parents[0].mkdir(parents=True, exist_ok=True)
outputs = []

if num is not None:
    import random
    idxs = random.sample(range(len([f for f in src_dir.glob(f"**/*{ext}")])), k=num)

for i, p in enumerate(prompt_dir.glob(f"**/*{ext}")):
    if num is None or i in idxs:
        src_path = src_dir / p.name
        assert src_path.exists()
        prompt_lines = None
        with p.open("r") as f:
            prompt_lines = f.read().rstrip("\n").split("\n")
        prompt_len = len(prompt_lines)
        assert prompt_len > 0
        prompt_idx = 0
        matched = False
        with src_path.open("r") as f:
            src_lines = f.read().rstrip("\n").split("\n")
            for i in range(prompt_len, len(src_lines)):
                if src_lines[i - prompt_len:i] == prompt_lines:
                    matched = True
                    break
            assert matched
            outputs.append({"filename": src_path.as_posix(), "question": "\n".join(prompt_lines), "answer": "\n".join(src_lines[i:])})
        
with trg_path.open("w") as f:
    json.dump(outputs, f)