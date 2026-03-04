# Chicago Crime Type Prediction — Machine Learning and Big Data (7006SCN)

## Project Overview

This project applies distributed machine learning using PySpark MLlib to predict 
crime categories from the Chicago Crime Dataset. The dataset contains over 8 million 
records collected from 2001 to 2024. Four classification algorithms were implemented 
and compared against scikit-learn single-node baselines.

## Dataset

- Source: Hugging Face: https://huggingface.co/datasets/gymprathap/Chicago-Crime-Dataset
- Original Source: Chicago Data Portal (https://data.cityofchicago.org)
- Size: 8,104,658 rows, 22 columns, approximately 1.91GB
- Target Classes: PROPERTY, VIOLENT, DRUG

To load the dataset, run the Data ingestion notebook. 
The dataset will be downloaded automatically from Hugging Face.

## Project Structure
```
Machine-Learning-and-Big-Data/
│
├── Data ingestion.ipynb                        # Data loading, EDA, Parquet conversion
├── feature engineering.ipynb                  # Feature engineering and preprocessing pipeline
├── Model Building and Tuning.ipynb            # Model training for LR, DT, RF, GBT
├── Hyperparameter tuning and Sklearn baseline.ipynb   # Tuning and sklearn comparison
├── Scalability Analysis and Model Stability.ipynb     # Scaling experiments and bootstrap analysis
│
├── lr_model/                                  # Saved Logistic Regression model
├── dt_model/                                  # Saved Decision Tree model
├── rf_model/                                  # Saved Random Forest model
├── gbt_model/                                 # Saved GBT model
│
└── README.md
```

## How to Run

1. Open Google Colab and mount your Google Drive
2. Install required libraries by running the first cell in any notebook
3. Run the notebooks in the following order:
   - Data ingestion.ipynb
   - feature engineering.ipynb
   - Model Building and Tuning.ipynb
   - Hyperparameter tuning and Sklearn baseline.ipynb
   - Scalability Analysis and Model Stability.ipynb

## Requirements

The following libraries are required and can be installed via pip:

- pyspark
- datasets
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install all at once by running this in your notebook:

pip install pyspark datasets pandas numpy matplotlib seaborn scikit-learn

## Models Implemented

- Logistic Regression — PySpark MLlib (multiclass baseline)
- Decision Tree — PySpark MLlib (nonlinear geographic boundaries)
- Random Forest — PySpark MLlib (ensemble, best multiclass model)
- Gradient Boosted Trees — PySpark MLlib (binary, VIOLENT vs Other)
- Scikit-learn baselines — LR, DT, RF on 5 percent subsample for comparison

## Key Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.5992 | 0.4527 |
| Decision Tree | 0.5984 | 0.5316 |
| Random Forest | 0.6108 | 0.4869 |
| GBT Binary | 0.6931 | 0.5902 |

Random Forest bootstrap mean accuracy: 0.6028 (95% CI: 0.6011 to 0.6047)

## Tableau Dashboards

All four interactive dashboards are published on Tableau Public.

Link: https://public.tableau.com/app/profile/shashi.vardhan1951/vizzes

The dashboards cover:
- Dashboard 1: Data quality and pipeline monitoring
- Dashboard 2: Model performance and feature importance
- Dashboard 3: Business insights and crime predictions
- Dashboard 4: Scalability and cost analysis

## Environment

- Platform: Google Colab
- Spark Version: 4.0.2
- Python Version: 3.12
- Executor Memory: 4GB
- Shuffle Partitions: 200
