## Installation

### Requirements:
- PyTorch >= 1.0. Installation instructions can be found in https://pytorch.org/get-started/locally/.
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9,< 6.0
- (optional) OpenCV for the webcam demo

### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name FCOS python=3.6
conda activate FCOS


# FCOS and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.2
# zzh: pytorch>1.10.1 may cause compiling error in the last step. So use pytorch=1.10.1
#      for Linux-CUDA11: conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/tianzhi0549/FCOS.git
cd FCOS

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop --no-deps

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

## some tips for error solutions in installation
# 1. THC/THC.h: No such file or directory: use python 3.6 + pytorch 1.10.1
# 2. AttributeError: module 'torch._six' has no attribute 'PY3': change PY3 to PY37 in python/FCOS/FCOS/fcos_core/utils/imports.py

```

### Option 2: Docker Image (Requires CUDA, Linux only)
*The following steps are for original maskrcnn-benchmark. Please change the repository name if needed.* 

Build image with defaults (`CUDA=9.0`, `CUDNN=7`, `FORCE_CUDA=1`):

    nvidia-docker build -t maskrcnn-benchmark docker/
    
Build image with other CUDA and CUDNN versions:

    nvidia-docker build -t maskrcnn-benchmark --build-arg CUDA=9.2 --build-arg CUDNN=7 docker/
    
Build image with FORCE_CUDA disabled:

    nvidia-docker build -t maskrcnn-benchmark --build-arg FORCE_CUDA=0 docker/
    
Build and run image with built-in jupyter notebook(note that the password is used to log in jupyter notebook):

    nvidia-docker build -t maskrcnn-benchmark-jupyter docker/docker-jupyter/
    nvidia-docker run -td -p 8888:8888 -e PASSWORD=<password> -v <host-dir>:<container-dir> maskrcnn-benchmark-jupyter
