## How to Install
We organize the most needed packages in `requirements.txt`. However, for some error-prone packages, we describe step-by-step instructions below.
```bash
git clone https://github.com/Tessantess/ARCH
cd ARCH
```
### 1. Create Conda Environment
```bash
conda create -n arch python=3.7
conda activate arch
```

### 2. Install torch
You can install with CUDA support at [here](https://pytorch.org/). If the CUDA version of your machine is not listed here, please  refer [here](https://pytorch.org/get-started/previous-versions/).
For example, for CUDA 11.1, torch 1.9.1, your can use this command:
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111  -f https://download.pytorch.org/whl/torch_stable.html
```
## 3. Install Pytorch3d
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
If errors occurs, please follow detailed installation instructions at [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### 4. Install requirements
```bash
pip install -r requirements.txt
```

### 5. Install vis-pytorch
the default branch of vis-pytorch need torch>=1.10, which is too strict for most situations. Here we checkout to a early version and install by source:
```bash
git clone https://github.com/lucidrains/vit-pytorch.git
cd vis-pytorch
git checkout 64aae4680b67ce0169c1d3a153502425d25456d7
python setup.py install
```

### 6. Install smpl 
In this project, we use a `smpl` package from SCANimate, please follow [these lines](https://github.com/shunsukesaito/SCANimate/blob/main/install.sh#L18-L20) to install it in this conda enviroment.  
