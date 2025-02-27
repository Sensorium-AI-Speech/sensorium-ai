from pathlib import Path

MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-zeroshot-shuffle-warmup50-weight1en1-fdisc-fm-pitch(50,1050)'

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

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 300000

# Weight applied to the discriminator loss
ADVERSARIAL_LOSS_WEIGHT = .1

# Training batch size
BATCH_SIZE = 256

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = False

# Step to start training discriminator
DISCRIMINATOR_START_STEP = 250000

# Whether to use the same discriminator as FARGAN
FARGAN_DISCRIMINATOR = True

# Weight applied to the feature matching loss
FEATURE_MATCHING_LOSS_WEIGHT = .1

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use the multi-period waveform discriminator from HiFi-GAN
MULTI_PERIOD_DISCRIMINATOR = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True

# Whether to use WavLM x-vectors for zero-shot speaker conditioning
ZERO_SHOT = True

# Whether to shuffle speaker embeddings during training
ZERO_SHOT_SHUFFLE = True

# Minimum and maximum frequency
FMIN = 50.  # Hz
FMAX = 1050.  # Hz

# Number of previous frames used for lookback in FARGAN 
FARGAN_PREVIOUS_FRAMES = 2  # frames    (note: before, I increased to 3 because (SAMPLE_RATE / FMIN) > (2 * promonet.HOPSIZE) = 512)