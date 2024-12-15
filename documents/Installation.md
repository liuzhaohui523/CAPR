### Installation

Our codebase is developed based on Ubuntu 18.04 and Pytorch framework.

```bash
# We suggest to create a new conda environment with python version 3.8
conda create --name cvpr python=3.8

# Activate conda environment
conda activate PHMR

# Install Pytorch that is compatible with your CUDA version
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install OpenDR
pip install matplotlib
Install opendr refer to this (https://github.com/microsoft/MeshTransformer/issues/35#issuecomment-1784231076)
If you find that two files are missing, please refer to the link (https://github.com/microsoft/MeshTransformer/issues/35#issuecomment-1781222050)


# Install CVPR
git clone https://github.com/liuzhaohui523/CAPR
cd CAPR
python setup.py build develop

# Install requirements
pip install -r requirements.txt
```