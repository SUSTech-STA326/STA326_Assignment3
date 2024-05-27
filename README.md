# TO USE THE CODE
## First unzip data.zip to where it locates.!!!!!!!!

Then run run_models.py. All configs can be modified at run_models too.

Use plot.ipynb to make plots.

# Some codes are cited from 
Yihong Cheng, Qrh "neural-collaborative-filtering." https://github.com/yihong-chen/neural-collaborative-filtering

---

# STA326 - Assignment 3

## Overview

This assignment focuses on implementing and comparing three recommender system models—GMF, MLP, and NeuMF—based on the Neural Collaborative Filtering (NCF) framework as discussed in the referenced paper by He et al. (2017). The implementation will utilize the MovieLens dataset and will be evaluated using HR@10 and NDCG@10 metrics.

## Requirements

- Python 3.8 or later
- PyTorch 1.8 or later
- NumPy
- Pandas
- Matplotlib


## Models

- **GMF (Generalized Matrix Factorization)**: A matrix factorization model enhanced with neural architecture.
- **MLP (Multi-Layer Perceptron)**: A deep learning model for learning user-item interaction functions.
- **NeuMF (Neural Matrix Factorization)**: Combines GMF and MLP under a shared framework for enhanced performance.

## Usage

To run the models, execute the command like:
```
python run_models.py
```

This script will train the GMF, MLP, and NeuMF models on the MovieLens dataset and output the evaluation metrics HR@10 and NDCG@10 for each model.

## Evaluation

After training, the models' performance can be evaluated by comparing HR@10 and NDCG@10 metrics. Additionally, you can reproduce the ablation study for the MLP model with different layers to observe the impact on performance.

## Submission

Submit your assignment by creating a pull request to the specified GitHub repository. Ensure that your final submission includes:
- All source code files
- A one-page report detailing your results, submitted on Blackboard

## References

- He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
- Code ref: https://github.com/yihong-chen/neural-collaborative-filtering

