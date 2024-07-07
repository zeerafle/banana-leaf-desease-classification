import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
NUM_TREES = 100
SEED = 9
BINS = 8
FIXED_SIZE = (500, 500)
TRAIN_PATH = os.path.join(ROOT, 'dataset', 'train')
TEST_PATH = os.path.join(ROOT, 'dataset', 'test')
H5_TRAIN_DATA = os.path.join(ROOT, 'output', 'train_data.h5')
H5_TRAIN_LABELS = os.path.join(ROOT, 'output', 'train_labels.h5')
