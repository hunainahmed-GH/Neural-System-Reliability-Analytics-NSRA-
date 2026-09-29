from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACHINE = "machine-1-1"
RAW_DIR = ROOT / "data" / "raw" / MACHINE
PROC_DIR = ROOT / "data" / "processed" / MACHINE
EDA_DIR = ROOT / "experiments" / "results" / "eda"
SMD_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"

N_FEATURES_RAW, N_FEATURES = 38, 30
# feature ids are 0-based column indices (f00..f37)
CONSTANT_FEATURES = [4, 7, 16, 17, 26, 28, 36, 37]
N_TRAIN = N_TEST = 28479

# chronological splits: (source, start, end)
SPLITS = {
    "train_fit":  ("train", 0, 22783),
    "train_val":  ("train", 22783, 28479),
    "sup_train":  ("test", 0, 17794),
    "sup_val":    ("test", 17794, 20437),
    "final_test": ("test", 20437, 28479),
}
EXPECTED_ROWS = {"train_fit": 22783, "train_val": 5696, "sup_train": 17794, "sup_val": 2643, "final_test": 8042}
EXPECTED_ANOMALIES = {"sup_train": 1100, "sup_val": 1178, "final_test": 416}

WINDOW_SIZE = 30
LABEL_MODE = "last"
SCALING = "none"  # source data already min-max scaled to [0,1]
SEED = 42
