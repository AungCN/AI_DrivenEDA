# AI-Driven Exploratory Data Analysis (EDA) & Time-Series Modeling

An interactive **Streamlit** application suite for automated Exploratory Data Analysis (EDA), feature engineering, supervised classification/regression benchmarking, and time-series forecasting.

## Key Features

- **Automated EDA & Visualization**: Summary statistics, missing value handling, correlation heatmaps, and distribution plots.
- **Supervised Machine Learning (`aimodel_eda_v1.py` – `aimodel_eda_v7_3.py`)**:
  - Classification & Regression with Logistic/Linear Regression, Decision Trees, Random Forest, KNN, SVM, Naive Bayes, XGBoost, LightGBM, and Keras Neural Networks.
  - Automated `SelectKBest` feature selection, cross-validation (`StratifiedKFold` / `KFold`), and model persistence (`*.pkl` tracked via Git LFS).
- **Time-Series Analysis & Forecasting (`aimodel_eda_ts_v8.py`, `aimodel_eda_ts_v9.py`)**:
  - Augmented Dickey-Fuller (ADF) stationarity test and seasonal decomposition.
  - ARIMA, SARIMAX, ARCH/GARCH volatility modeling, and LSTM deep learning forecasting.

## Installation & Usage

1. Clone the repository (with [Git LFS](https://git-lfs.com/) installed to pull pre-trained `.pkl` models):
   ```bash
   git lfs install
   git clone https://github.com/AungCN/AI_DrivenEDA.git
   cd AI_DrivenEDA
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit web app:
   ```bash
   # For general ML & Automated EDA:
   streamlit run aimodel_eda_v7_3.py

   # For Time-Series EDA & Forecasting:
   streamlit run aimodel_eda_ts_v9.py
   ```

## License

This project is licensed under the [MIT License](LICENSE).
