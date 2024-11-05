# Apnea Interact Xplainer (AIX)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Stars](https://img.shields.io/github/stars/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Forks](https://img.shields.io/github/forks/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ffdu-harry%2FApnea-Interact-Xplainer&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)

## Overview
AIX (Apnea Interact Xplainer) is a transparent AI system that enables sleep apnea diagnosis through efficient variable-modal analysis across clinical and home settings. It is a PyTorch-based framework containing both interpretable deep learning models and interactive visualization tools, as introduced in our paper [Transparent Artificial Intelligence-enabled Interpretable and Interactive Sleep Apnea Assessment across Flexible Monitoring Scenarios].

<div align="center">
    <img src="figures/framework.png" width="800px">
    <p>Framework of AIX system for transparent sleep apnea assessment.</p>
</div>

## Key Features
- 🔍 Multi-level interpretable visualization
- 🏥 Flexible monitoring scenarios support
- 🤝 Human-AI collaborative diagnosis
- 📊 Interactive result exploration
- 📱 Clinical deployment ready

## Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#downloading-data)
- [Model Training](#model-training)
- [Visualization](#visualization)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

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
[您的论文引用格式]
