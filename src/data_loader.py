import urllib.request
import numpy as np
from src.config import RAW_DIR, SMD_URL, MACHINE, N_TRAIN, N_TEST, N_FEATURES_RAW


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for part in ("train", "test", "test_label"):
        f = RAW_DIR / f"{part}.txt"
        if not f.exists() or f.stat().st_size == 0:
            urllib.request.urlretrieve(f"{SMD_URL}/{part}/{MACHINE}.txt", f)
    return RAW_DIR


def load_raw():
    """Returns train (N,38), test (N,38), test_labels (N,) as float32/float32/uint8."""
    download()
    train = np.loadtxt(RAW_DIR / "train.txt", delimiter=",", dtype=np.float32)
    test = np.loadtxt(RAW_DIR / "test.txt", delimiter=",", dtype=np.float32)
    y = np.loadtxt(RAW_DIR / "test_label.txt").astype(np.uint8)
    assert train.shape == (N_TRAIN, N_FEATURES_RAW) and test.shape == (N_TEST, N_FEATURES_RAW) and y.shape == (N_TEST,)
    return train, test, y
