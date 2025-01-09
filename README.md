
# Complex Quantitative Change Based on Big Data

## Author: Tan Wei Ming

This project explores advanced AI and Big Data analysis techniques to predict quantitative changes in large datasets. The focus is on multiple stages of data processing, feature engineering, model training, hyperparameter tuning, and advanced model evaluation. The project demonstrates the integration of various AI/ML models and Big Data processing workflows to solve real-world challenges.

## Features

- **Data Preprocessing**: Extensive data cleaning, missing values imputation, outlier detection, and normalization.
- **Feature Engineering**: Advanced techniques like feature selection, transformation, dimensionality reduction (e.g., PCA, LDA).
- **Modeling**: Regression, classification, and time-series models for predictions and anomaly detection.
- **Pipelines**: Full automation of training, tuning, and prediction pipelines.
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV for optimizing model parameters.
- **Evaluation**: In-depth performance metrics like accuracy, precision, recall, F1-score, MSE, and R2.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- jupyter
- xgboost
- lightgbm
- catboost

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Structure

This project follows a modular structure:

- **data/**: Contains raw, processed, and external data files.
- **notebooks/**: Jupyter notebooks for data preprocessing, model training, feature engineering, and analysis.
- **src/**: Source code for data processing, model definitions, pipelines, and utilities.
- **tests/**: Unit tests for data processing, feature engineering, models, and pipelines.
- **requirements.txt**: Python packages required for the project.
- **setup.py**: Installation script for the project.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/TanWeiMing/Complex-Quantitative-Change-Based-on-Big-Data.git
   ```

2. Install the required dependencies:
   ```bash
   cd Complex-Quantitative-Change-Based-on-Big-Data
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks in the `notebooks/` directory for data analysis and modeling:
   ```bash
   jupyter notebook
   ```

4. Train models and tune hyperparameters by running:
   ```bash
   python src/pipelines/training_pipeline.py
   ```

5. View results and evaluations in the `notebooks/result_analysis.ipynb`.
