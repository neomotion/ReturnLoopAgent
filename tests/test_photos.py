import hashlib

from photos import PHOTOS, dhash, distance

def test_resaved_photo_is_caught_but_different_photos_are_not():
    old = dhash(PHOTOS / "old_k0_laptop.jpg")
    md5 = lambda name: hashlib.md5((PHOTOS / name).read_bytes()).hexdigest()
    assert md5("k6.jpg") != md5("old_k0_laptop.jpg")          # an exact hash misses the reuse
    assert distance(old, dhash(PHOTOS / "k6.jpg")) <= 8        # the perceptual hash catches it
    assert distance(old, dhash(PHOTOS / "k1.jpg")) > 8
    assert distance(old, dhash(PHOTOS / "k9.jpg")) > 8
