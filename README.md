# Sensorium-AI Dev

## Setting up with this repo
1. Create and activate a Python 3.10 environment (using `conda` or `venv` is fine) (i.e. `conda create --name sensorium_1 python=3.10`)
2. Run `pip install --upgrade "pip<24.1" && pip install "setuptools>=61.0,<70"`
    - This is to mitigate `FileNotFoundError: [Errno 2] No such file or directory: 'fairseq/version.txt` when installing promonet.
2. Clone this repo.
3. Run `git submodule update --init --recursive`
4. Within `./torbi/pyproject.toml`, change the lines with torch dependencies to have no version specifier (i.e. change `torch==2.3.1` to just `torch`)
5. Install `torbi` (`cd ./torbi && pip install . && cd ..`)
6. Install dev branch of `penn` (`pip install ./penn`)
7. Install dev branch of `promonet` (`pip install ./promonet`)
    - You can sanity check by running `python -c "import promonet; help(promonet)"` and seeing if under `PACKAGE CONTENTS` you see `adapt`,`baseline`, ..., `train`.
8. Download `*.pt` and `*.ts` for a model.
    - i.e. `fargan-zeroshot-causal-weighted-incpitch-singing-600k.pt` and `fargan-zeroshot-causal-weighted-incpitch-singing-600k.ts`
9. Extracting speech features. Run the following (check below for config file and weight file pairs):

```
python -m promonet.data.pack --config <config_file.py> --audio_file <audio_file> --output_file <output_csv> --checkpoint <weight_file.pt> [--gpu 0]
```

For example:

```
python -m promonet.data.pack --config configs/fargan-zeroshot-shuffle-warmup50-weight1en1-fdisc-fm-pitch\(50\,1050\)-incpitch.py --audio_file ../sensorium_max_objects/test_files/wav/test.wav --output_file ./test.csv --checkpoint /mnt/d/_doctorate/model/fargan-zeroshot-causal-weighted-incpitch-singing-600k.pt --gpu 0
```

## Config-Weight File Pairs
Under `./configs` are some of the configs we used for training/inference that you'll need to supply for feature extraction (i.e. `python -m promonet.data.pack`)

| Weight File | Config File |
|-------|-------|
|`fargan-zeroshot-causal-weighted-incpitch-singing-600k.pt` | `configs/fargan-zeroshot-shuffle-warmup50-weight1en1-fdisc-fm-pitch\(50\,1050\)-incpitch.py` |



## Setting up in general



## Windows Setup
I personally use Windows Subspace for Linux (WSL-2) for development. Assuming that you've setup WSL and CUDA for WSL correctly (https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2), I setup my developement environment by doing the following:

1. Setup a python environment (personally I use venv: `python3 -m venv .venv`)
2. Activate the environment if not activated already.
3. `pip install --upgrade "pip<24.1" && pip install "setuptools>=61.0,<70"`
4. Clone torbi (`git clone https://github.com/maxrmorrison/torbi.git`)
5. Edit the torch dependencies within `torbi/pyproject.toml` to have no version specifier.
6. Install `torbi` (`cd torbi/ && pip install -e . && cd ..`)
7. Install dev branch of `penn` (`git clone -b dev https://github.com/interactiveaudiolab/penn.git && cd penn && pip install -e . && cd ..`)
8. Install dev branch of `promonet` (`git clone -b dev https://github.com/maxrmorrison/promonet.git && cd promonet && pip install -e . && cd ..`)
9. Install `torchutil` so we can have a progress bar for downloading datasets:
    * `git clone https://github.com/maxrmorrison/torchutil.git`
    * Edit stuff.
    * `cd torchutil && pip install -e . && cd ..`
10. Check if your promonet installation is good:
    * Run `python`
    * `import promonet`
    * `help(promonet)`
    * If there are no package contents (i.e. adapt, baseline, etc.), then `quit()` and do `pip install -e .` again.
    * Repeat the check to see if that worked. (If not, then there needs to be more troubleshooting)

### WSL + CUDA steps
1. Remove old GPG Key
    * `sudo apt-key del 7fa2af80`
2. Follow the instructions to intstall the CUDA Toolkit for WSL-Ubuntu (https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
    1. wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    2. sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    3. wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.0-1_amd64.deb
    4. sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.0-1_amd64.deb
    5. sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
    6. sudo apt-get update
    7. sudo apt-get -y install cuda-toolkit-12-6