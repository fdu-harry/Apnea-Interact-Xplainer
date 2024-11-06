# Apnea Interact Xplainer (AIX)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Stars](https://img.shields.io/github/stars/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Forks](https://img.shields.io/github/forks/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ffdu-harry%2FApnea-Interact-Xplainer&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)
[![Watching](https://img.shields.io/github/watchers/fdu-harry/Apnea-Interact-Xplainer?style=social)](https://github.com/你的用户名/仓库名/watchers)

## Overview
AIX (Apnea Interact Xplainer) is a transparent AI system that enables sleep apnea diagnosis through efficient variable-modal analysis across clinical and home settings. It is a PyTorch-based framework containing both interpretable deep learning models and interactive visualization tools, as introduced in our paper [Transparent Artificial Intelligence-enabled Interpretable and Interactive Sleep Apnea Assessment across Flexible Monitoring Scenarios].

<div align="center">
    <img src="https://raw.githubusercontent.com/fdu-harry/Apnea-Interact-Xplainer/main/figures/AIX.jpg" width="800px">
    <p>Framework of AIX system for transparent sleep apnea assessment.</p>
</div>


## Contents
- [Key Features](#key-features)
- [AIX One-Click Installation Guide](#aix-one-click-installation-guide)
- [Steps to Run Pred_single_modal.py](#steps-to-run-pred_single_modalpy)
- [Data Preparation](#downloading-data)
- [Installation](#installation)
- [Model Training](#model-training)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)
- [Citation](#citation)


## Key Features
- 🔍 Multi-level interpretable visualization
- 🏥 Flexible monitoring scenarios support
- 🤝 Human-AI collaborative diagnosis
- 📊 Interactive result exploration
- 📱 Clinical deployment ready


## AIX One-Click Installation Guide
You can download the pre-built executable from:

- [Baidu Netdisk](https://pan.baidu.com/s/1VnALWGZ3c44CqNQFaTSlNg?pwd=aixx) (Extraction Code: aixx)
- [Google Drive](https://drive.google.com/file/d/1NiiBx4XNWH8ubIBfoLP4I6Kw5A9vHBPO/view?usp=drive_link)

### System Requirements
- Windows 10/11
- 8GB RAM minimum
- 10GB free disk space

### Quick Start
1. Download the AIX.zip file
2. Extract to your preferred location
3. Run AIX.exe
4. For detailed step-by-step instructions, please watch our tutorial video:
- [AIX Operation Tutorial](file name: AIX video; size: 26M):
- [Baidu Netdisk](https://pan.baidu.com/s/1q1M8KSGgmn8_E7ovnLssSw?pwd=aixx) (Extraction Code: aixx)
- [Google Drive](https://drive.google.com/file/d/1l-UolCbew5eL7EsSvH7HctGgKL_zdUen/view?usp=drive_link)


## Steps to Run Pred_single_modal.py

Since the model weight file is large (39M), it is stored using Git LFS and requires specific steps to download:

### Option 1: Download ZIP
1. Click the green "Code" button and select "Download ZIP"
2. Extract the downloaded ZIP file
3. Download model weights:
   - Check `model_weights_download.txt` for download links
   - Place the model files in the `./weights` directory
  

### Option 2: Clone Repository
#### Prerequisites
> Note: If Git or Git LFS is already installed, you can uninstall them from Control Panel first to avoid conflicts.

1. Download and install Git from: https://git-scm.com/downloads
2. Download and install Git LFS from: https://git-lfs.com/

#### Steps

1. Enter the drive 
```bash
H:    # Choose any drive with sufficient storage space
```

2. Create a new directory
```bash
mkdir test_lfs
```
3. Enter the directory
```bash
cd test_lfs
```

4. Install Git LFS
```bash
git lfs install
```

5. Clone repository (using SSH)
```bash
git clone git@github.com:fdu-harry/Apnea-Interact-Xplainer.git
```

6. Enter project directory
```bash
cd Apnea-Interact-Xplainer
```

7. Pull LFS files
```bash
git lfs pull
```
8. Run file Pred_single_modal.py


## Downloading Data
Our experiments use both public and private datasets. Here are the instructions for accessing each dataset:

### 1. Public Datasets
#### NSRR Dataset (https://sleepdata.org/)
Install NSRR gem and download SHHS/MESA/MROS/CFS dataset:
```bash
gem install nsrr
nsrr download shhs
nsrr download mesa
nsrr download mros
nsrr download cfs
```

## Installation

### Environment Setup
```bash
# Create and activate conda environment
conda create -n aix python=3.9
conda activate aix

# Install PyTorch (GPU version)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

## License
This project is licensed under MIT License with additional terms:

### Permissions
- Academic and research use with proper citation
- Access to source code for research purposes
- Clinical deployment ONLY through explicit authorization from the authors

### Usage Restrictions
- Commercial use requires explicit permission from the authors
- Clinical deployment requires prior written agreement
- Large-scale usage or modifications need author authorization

### For Clinical Partners
For clinical deployment or collaboration inquiries, please contact:
- Author Name: Shuaicong Hu
- Email: schu22@m.fudan.edu.cn
- Institution: Fudan university

### For Researchers
If you use this code in your research, please cite our paper:
```bibtex
[Waiting for updates......]
