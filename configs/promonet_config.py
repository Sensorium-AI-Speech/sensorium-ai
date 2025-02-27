from pathlib import Path

# Root location for saving outputs
ROOT_DIR = Path('/project/mc232/ds867/sensorium_ai')

# Location to save assets to be bundled with pip release
ASSETS_DIR = ROOT_DIR / 'assets'

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'

# Location to save results
RESULTS_DIR = ROOT_DIR / 'results'

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

MODULE = 'promonet'

# Configuration name
CONFIG = 'promonet-orig'