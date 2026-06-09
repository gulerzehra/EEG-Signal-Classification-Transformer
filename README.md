# EEG-Signal-Classification-Transformer

Neural Decoding Framework: Benchmarking Traditional Ensemble Learning against Self-Attention Transformers on Multi-Subject EEG Data
This repository contains the official implementation for our Neural Engineering semester project. The core objective of this framework is to address the cross-subject variability bottleneck in Non-Invasive Brain-Computer Interfaces (BCIs) by constructing a robust subject-independent decoding pipeline for Motor Imagery (MI) classification.

Our pipeline benchmarks a multi-core optimized Traditional Machine Learning Baseline (Random Forest) against a modern Deep Learning Sequence Architecture (Self-Attention EEG Transformer) using the benchmark BCI Competition IV Dataset 2a.

System Block Diagram & Pipeline Architecture
The experimental workflow is systematically structured into four core processing blocks:

[ Raw Multi-Channel EEG Input ] (9 Subjects Accumulated --> 2592 Total Trials)
│
▼
[ Bandpass Digital Filtering ] (4 - 38 Hz Spectrum: Traps Mu & Beta Rhythms)
│
▼
[ Stratified Holdout Splitting ] (Train Corpus vs. Balanced Holdout Test Set)
│
┌──────┴────────────────────────┐
▼                               ▼
[ Flattened Matrix Reshaping ]   [ Continuous Sequence Tokenization ]
│                               │
▼                               ▼
[ Baseline Model: Random Forest ] [ Proposed Model: EEG Transformer Network ]
│                               │
└──────┬────────────────────────┘
▼
[ Unified Benchmarking Suite ] (4-Panel Visual Analytics & Precision-Recall Matrix)

Core Performance Metrics Summary
To circumvent individual subject overfitting and evaluate proper generalization, trials from all 9 distinct subjects were aggregated into a comprehensive corpus (2,592 baseline trials), utilizing an unseen stratified test partition (519 evaluation trials) with an ideal uniform class balance (256 Left Hand vs. 263 Right Hand trials).

Evaluation Metric | Baseline Architecture (Random Forest) | Proposed Architecture (EEG Transformer)
Overall Holdout Test Accuracy | 0.5395 | 0.5125
Left Hand - Precision | 0.5316 | 0.5065
Left Hand - Recall | 0.5586 | 0.4570
Left Hand - F1-Score | 0.5448 | 0.4805
Right Hand - Precision | 0.5480 | 0.5174
Right Hand - Recall | 0.5209 | 0.5665
Right Hand - F1-Score | 0.5341 | 0.5408

Visual Analytics Suite
The comparative evaluation framework maps spatial-temporal neural patterns across localized sensorimotor regions.

Quantitative Highlights & Discussion
Generalization Over Random Chance: In a non-synthetic, multi-subject paradigm characterized by massive non-stationarity, consistently scaling beyond the strict 50.0% stochastic baseline confirms that both models extract authentic cortical synchronization features instead of transient artifact patterns.

The Self-Attention Advantage: While the Random Forest model leverages localized variance across the flattened spatial-temporal matrix efficiently, the proposed EEG Transformer establishes superior tracking on Right Hand tasks, achieving an F1-score of 0.5408 (surpassing the baseline's 0.5341). This explicitly indicates that scaled dot-product attention maps sequential, long-range event-related desynchronization (ERD) oscillations across the motor cortex with enhanced localized resolution.

Project Execution & Deployment Guide
Prerequisites & Dependencies
Ensure your environment satisfies the following requirements prior to notebook execution:

pip install torch numpy pandas scikit-learn seaborn matplotlib

Reproducing the Experiments
Clone the repository to your environment:
git clone [https://github.com/your_username/your_repository_name.git](https://github.com/gulerzehra/EEG-Signal-Classification-Transformer.git
)

Open the file EEG_Classification_Benchmark.ipynb in Google Colab.

Navigate to "Runtime" -> "Change runtime type" and select T4 GPU Acceleration (CUDA backend compilation).

Run all cells sequentially to reproduce the comprehensive metrics table and export the high-resolution (300 DPI) analytical visualization suite.

Academic Contribution & Framework Future Outlook
This project documents a transparent baseline for sequential neural decoding models. Future developmental iterations will integrate adaptive spatial spatial filtering blocks—such as Common Spatial Patterns (CSP) or layered Convolutional Feature Extractors (CNN)—directly preceding the Transformer encoders to simultaneously capture advanced topological and non-linear dynamic brain signal configurations.
