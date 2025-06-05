# AI-ML-Internship-Task7

Support Vector Machines (SVM) Task
Overview
This project implements a Support Vector Machine (SVM) classifier using the Breast Cancer dataset from Scikit-learn. The task demonstrates linear and RBF kernel SVMs, visualizes decision boundaries, tunes hyperparameters, and evaluates performance using cross-validation.
Project Structure
svm_task/
├── data/                      # Directory for dataset (not included in repo)
├── src/
│   └── svm_classifier.py      # Main script for SVM implementation
├── plots/
│   └── decision_boundary_*.png # Decision boundary visualizations
├── README.md                  # Project description


Setup Instructions

Clone the repository:git clone <your-repo-link>

Run the script:python src/svm_classifier.py



Implementation Details

Dataset: Breast Cancer dataset from Scikit-learn.
Features: All features for training, first two features (mean radius, mean texture) for visualization.
Models: Linear SVM and RBF SVM.
Visualization: Decision boundaries plotted for 2D data.
Hyperparameter Tuning: Grid search over C and gamma for RBF SVM.
Evaluation: Accuracy and cross-validation scores reported.
