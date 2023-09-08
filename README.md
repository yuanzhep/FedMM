# Federated Multi-Modal Learning (FedMM) for Computational Pathology

## Overview

The fusion of complementary multimodal information is increasingly vital in computational pathology for more accurate diagnostics. The traditional multimodal learning methods require access to users' raw data, raising significant privacy concerns. Federated Learning (FL) offers a privacy-preserving alternative but does not adequately address the challenges posed by heterogeneous modalities across different clients.

To bridge this gap, we introduce the Federated Multi-Modal Learning (FedMM) framework. FedMM trains multiple federated single-modal feature extractors, in contrast to conventional FL methods that focus on training a multimodal fusion model.

## Key Features

- **Privacy-Preserving**: No need to access users' raw data, mitigating privacy risks.
- **Heterogeneous Modalities**: Tailored to handle different (and possibly overlapping) data modalities across clients.
- **Local Inference**: Clients can locally extract features and perform classifications, reducing the need for centralized computation.
- **Improved Generalization**: Federated feature extractors are trained on distributed data, enhancing their ability to generalize.

## Performance

Comprehensive evaluations on two publicly available datasets show that FedMM significantly outperforms existing baselines in terms of accuracy and AUC metrics.

## Getting Started

To get started with FedMM, follow these installation steps:

```bash
git clone https://github.com/yourusername/FedMM.git
cd FedMM
pip install -r requirements.txt
