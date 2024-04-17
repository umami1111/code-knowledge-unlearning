import json
from pathlib import Path
import sys

assert len(sys.argv) >= 4

src_dir = Path(sys.argv[1])
ext = sys.argv[2]
trg_path = Path(sys.argv[3])
num = int(sys.argv[4]) if len(sys.argv) >= 5 else None

assert src_dir.exists()

trg_path.parents[0].mkdir(parents=True, exist_ok=True)
outputs = []

if num is not None:
    import random
    idxs = random.sample(range(len([f for f in src_dir.glob(f"**/*{ext}")])), k=num)

for i, p in enumerate(src_dir.glob(f"**/*{ext}")):
    if num is None or i in idxs:
        with p.open("r") as f:
            outputs.append({"filename": p.as_posix(), "content": f.read()})
        
with trg_path.open("w") as f:
    json.dump(outputs, f)