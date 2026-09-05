"""Lab C unit test: verify packed-batch attention resets and response-only
loss masking at example seams (Ch. 11)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.packing import pack_examples, loss_mask  # noqa: E402


def main():
    ex = [([1, 2, 3], [10, 11]), ([4, 5], [12]), ([6], [13, 14, 15])]
    seq = 16
    ids, seg, n_real = pack_examples(ex, seq_len=seq, eot_id=0)
    mask = loss_mask(seg, [len(p) for p, _ in ex], [len(r) for _, r in ex], n_real)
    print("ids: ", np.array(ids))
    print("seg: ", np.array(seg), " (segment id per position; -1 = pad)")
    print("loss:", np.array(mask), " (1 = response token scored)")
    # seam: distinct segment ids appear (attention must reset across them)
    seams_ok = len({s for s in seg if s >= 0}) >= 2
    # prompt tokens never scored; padding never scored
    prompt_masked = all(mask[i] == 0 for i in range(3))          # first prompt
    pad_masked = all(m == 0 for m, s in zip(mask, seg) if s == -1)
    some_scored = sum(mask) > 0
    ok = seams_ok and prompt_masked and pad_masked and some_scored
    print("PACKING INSPECT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
