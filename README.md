# Transparent Artificial Intelligence-enabled Interpretable and Interactive Sleep Apnea Assessment across Flexible Monitoring Scenarios

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Stars](https://img.shields.io/github/stars/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Forks](https://img.shields.io/github/forks/fdu-harry/Apnea-Interact-Xplainer?style=social)
![Visitors](https://visitor-badge.glitch.me/badge?page_id=fdu-harry.Apnea-Interact-Xplainer)

[Paper](论文链接)

## Summary
Early detection of potentially widespread undiagnosed sleep apnea (SA) is crucial for preventing its severe health complications, yet large-scale diagnosis faces barriers of limited trust in automated analysis and monitoring inaccessibility, particularly due to the absence of transparent artificial intelligence (AI) frameworks capable of monitoring adaptation. Here, we develop Apnea Interact Xplainer (AIX), a transparent AI system that enables SA diagnosis through efficient variable-modal analysis across clinical and home settings. Analyzing 15,510 polysomnography records from six independent multi-ethnic databases, AIX achieves 99.8% accuracy within one severity grade and R-squared of 0.93-0.96 for apnea-hypopnea index prediction on external test sets. AIX provides multi-level expert-logic interpretable visualization of respiratory patterns enabling human-AI collaboration. Notably, AIX achieves sensitivity of 0.949 for early SA detection using only oximetry signals, while providing nightly risk assessment and intelligent monitoring reports. This study establishes a paradigm shift in advancing early and cost-effective SA diagnosis through transparent AI.

## Downloading Data
Our experiments use both public and private datasets. Here are the instructions for accessing each dataset:

### 1. Public Datasets
#### SHHS Dataset
Install NSRR gem and download SHHS dataset:
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
