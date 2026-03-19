# MMKNet: Neural Network-Assisted Multi-Model Kalman Filter for Target Tracking

This repository contains the official PyTorch implementation of MMKNet, proposed in the paper:

MMKNet: A neural network-assisted multi-model Kalman filter for real-time target tracking
Shanli Chen, Peng Cai, Xiao Liu, Yunfei Zheng, Shiyuan Wang
Signal Processing, 2026
Paper Link: https://www.sciencedirect.com/science/article/abs/pii/S0165168426001088

## Overview

MMKNet is a neural network-assisted Kalman filter designed for tracking highly maneuverable targets. The key idea is to integrate an encoder-only Transformer into the Extended Kalman Filter (EKF) framework to predict the target's turn rate in real time, enabling adaptive state estimation with dynamically updated motion models.

## Main Contributions

- Hybrid Approach: Combines model-based interpretability (EKF) with data-driven flexibility (Transformer) for maneuvering target tracking
- Turn Rate Prediction: Uses self-attention to estimate angular velocity from radar measurements, adapting between CV and CT models
- No Real Training Data Needed: Leverages prior knowledge of motion parameters to generate pseudo-labeled training data (AAL method)
- Robust Design: Difference-based input features and sliding windows capture motion patterns effectively
- Efficient: Outperforms traditional IMM and existing NNA methods with fewer parameters

## Code Structure

├── main_linear.py          # Main training and evaluation script

├── filter.py               # MMKNet filtering implementation

├── gen_dataset.py          # Artificial data generation (AAL)

├── state_dict_learner.py   # Neural network training utilities

├── Data/linear/            # Generated training/test datasets

├── Model/linear/           # Saved model checkpoints

├── Simulations/linear/     # Simulation outputs and logs

└── README.md

## Citation

If you find this code useful in your research, please cite:

@article{chen2026mmknet,
  title={MMKNet: A neural network-assisted multi-model Kalman filter for real-time target tracking},
  author={Chen, Shanli and Cai, Peng and Liu, Xiao and Zheng, Yunfei and Wang, Shiyuan},
  journal={Signal Processing},
  pages={110594},
  year={2026},
  publisher={Elsevier}
}
