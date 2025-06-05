import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and prepare the dataset
def load_and_prepare_data():
    # Load Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Select two features for visualization (mean radius and mean texture)
    X_2d = X[:, [0, 1]]  # First two features for 2D plotting
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_2d_scaled = scaler.fit_transform(X_2d)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(X_2d_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X_2d_train, X_2d_test, y_2d_train, y_2d_test, data.feature_names

# 2. Train SVM models
def train_svm_models(X_train, y_train, X_2d_train, y_2d_train):
    # Linear kernel SVM for full dataset
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train, y_train)
    
    # RBF kernel SVM for full dataset
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    # Linear kernel SVM for 2D dataset (for visualization)
    svm_linear_2d = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear_2d.fit(X_2d_train, y_2d_train)
    
    # RBF kernel SVM for 2D dataset (for visualization)
    svm_rbf_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf_2d.fit(X_2d_train, y_2d_train)
    
    return svm_linear, svm_rbf, svm_linear_2d, svm_rbf_2d

# 3. Visualize decision boundary (for 2D data)
def plot_decision_boundary(X, y, model, title, filename):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Mean Radius (scaled)')
    plt.ylabel('Mean Texture (scaled)')
    plt.title(title)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', filename))
    plt.close()

# 4. Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf']
    }
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# 5. Evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Training and test accuracy
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\n{model_name} Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))

# Main function
def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, X_2d_train, X_2d_test, y_2d_train, y_2d_test, feature_names = load_and_prepare_data()
    
    # Train models
    svm_linear, svm_rbf, svm_linear_2d, svm_rbf_2d = train_svm_models(X_train, y_train, X_2d_train, y_2d_train)
    
    # Visualize decision boundaries (using 2D models)
    plot_decision_boundary(X_2d_train, y_2d_train, svm_linear_2d, 'Linear SVM Decision Boundary', 'decision_boundary_linear.png')
    plot_decision_boundary(X_2d_train, y_2d_train, svm_rbf_2d, 'RBF SVM Decision Boundary', 'decision_boundary_rbf.png')
    
    # Evaluate models (using full dataset)
    evaluate_model(svm_linear, X_train, X_test, y_train, y_test, "Linear SVM")
    evaluate_model(svm_rbf, X_train, X_test, y_train, y_test, "RBF SVM")
    
    # Hyperparameter tuning
    best_model, best_params = tune_hyperparameters(X_train, y_train)
    print("\nBest Parameters from Grid Search:", best_params)
    evaluate_model(best_model, X_train, X_test, y_train, y_test, "Tuned RBF SVM")

if __name__ == "__main__":
    main()