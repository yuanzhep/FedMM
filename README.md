# FedMM: A PyTorch Implementation for Federated Multi-Modal Learning in Computational Pathology

## Overview
The FedMM framework aims to address the challenges of privacy and modality heterogeneity in computational pathology. This implementation is based on the paper "FedMM: Federated Multi-Modal Learning with Modality Heterogeneity in Computational Pathology."

## Key Features
- **Privacy-Preserving**: No need to access users' raw data, mitigating privacy risks.
- **Heterogeneous Modalities**: Designed to handle Heterogeneous (and possibly overlapping) data modalities across clients.
- **Local Inference**: Clients can locally extract features and perform classifications, reducing the need for centralized computation.
- **Improved Generalization**: Trains federated feature extractors on distributed data, enhancing generalizability.

## Performance
Comprehensive evaluations on two publicly available datasets show that FedMM significantly outperforms existing baselines in terms of accuracy and AUC metrics.
![Example Image](./img/FedMM.png)
## Installation Steps

### Prerequisites
- Install Anaconda/Miniconda

### Required Packages
\`\`\`bash
$ conda env create --name FedMM --file env.yml
$ conda activate FedMM
\`\`\`

### Additional Libraries
- Install PyTorch
- Install OpenSlide and openslide-python

### Tutorials
- [Tutorial 1](#)
- [Tutorial 2 for Windows](#)

## Data Processing

### Processing Raw WSI Data

#### Downloading WSI Data
- From GDC data portal
  - Use GDC data portal with a manifest file and a configuration file. Note: downloading raw WSIs may take several days and ~5TB of disk space.
  - Refer to [TCGA data portal documentation](#) for more details.

#### Preparing Patches
- OpenSlide, a C library with Python API, is used for reading WSI data.
- Consult the [OpenSlide Python API documentation](#) for details.

## Folder Structure
\`\`\`
tcga_wsi_data (distributed to three hospitals)
|-- wsi_nsclc 
|   |-- luad
|   |-- lusc
|-- wsi_rcc 
|   |-- kirc
|   |-- kirp
|-- features_0725_2023 (Each csv represents a WSI, rows in csv represent different instances or patches)
|   |-- TCGA-22-5480-01Z-00-DX1.csv
|   |-- ...
|-- label.csv (labels)
\`\`\`

\`\`\`
tcga_cnv_data (distributed to three hospitals)
|-- wsi_nsclc 
|   |-- luad
|   |-- lusc
|-- wsi_rcc 
|   |-- kirc
|   |-- kirp
|-- features_0802_2023
|   |-- TCGA-22-5480-01Z-00-DX1.csv
|   |-- ...
|-- label.csv (labels)
\`\`\`
