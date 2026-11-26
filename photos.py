import hashlib
from pathlib import Path

from PIL import Image

PHOTOS = Path(__file__).parent / "data" / "photos"

def dhash(path):
    # perceptual hash: shrink to 9x8 grey, 1 bit per pixel = "brighter than its right neighbour?"
    # survives resizing, recompression and small colour changes; a byte hash (md5) does not
    img = Image.open(path).convert("L").resize((9, 8))
    px = img.tobytes()
    bits = [px[row * 9 + col] > px[row * 9 + col + 1] for row in range(8) for col in range(8)]
    return "".join("1" if b else "0" for b in bits)

def distance(a, b):
    # how many of the 64 bits differ (0 = same picture)
    return sum(x != y for x, y in zip(a, b))

# ---- check: a resaved copy stays close, a different photo is far ----
if __name__ == "__main__":
    old, reused = PHOTOS / "old_k0_laptop.jpg", PHOTOS / "k6.jpg"
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    print(f"md5 equal (k6 vs old claim photo): {md5(old) == md5(reused)}")
    for p in sorted(PHOTOS.glob("*.jpg")):
        print(f"dHash distance {old.name} vs {p.name}: {distance(dhash(old), dhash(p))}")
