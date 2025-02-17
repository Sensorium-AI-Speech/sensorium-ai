# Sensorium-AI Dev

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

### 
