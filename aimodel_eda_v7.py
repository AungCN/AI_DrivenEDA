# aimodel_eda_v7.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# Helper Functions
# ---------------------------------
# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_installed = True
except:
    xgb_installed = False
    
def safe_mode_impute(series, fill_value='Unknown'):
    mode_values = series.mode()
    if not mode_values.empty:
        return series.fillna(mode_values.iloc[0])
    else:
        return series.fillna(fill_value)

def auto_eda(df):
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Basic Information")
    st.write(df.describe(include='all'))
    st.write("Shape:", df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap (Numerical Features Only)")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numerical columns to plot correlation heatmap.")

def preprocess(df, target_column):
    df = df.copy()

    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # Ensure all values are string
        df[col] = le.fit_transform(df[col])

    # Separate X and y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    X = X.fillna(X.median())
    y = safe_mode_impute(y)

    return X, y

def train_models(X, y, task_type):
    models = {}
    scores = {}

    if task_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        scoring = 'accuracy'

    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(),
            'SVR': SVR()
        }
        scoring = 'r2'

    leaderboard = []

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            avg_score = np.mean(cv_scores)
            leaderboard.append((name, avg_score))
        except Exception as e:
            st.warning(f"⚠️ Model {name} could not be trained. Reason: {str(e)}")

    leaderboard = sorted(leaderboard, key=lambda x: x[1], reverse=True)

    return leaderboard

# ---------------------------------
# Streamlit App
# ---------------------------------

def main():
    st.title("🤖 AI-Driven EDA and AutoML Leaderboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        auto_eda(df)

        target_column = st.selectbox("Select the Target Column", df.columns)

        task_type = st.radio("Select Task Type", ('classification', 'regression'))

        if st.button("Train Models 🚀"):
            X, y = preprocess(df, target_column)

            leaderboard = train_models(X, y, task_type)

            if leaderboard:
                st.subheader("Leaderboard")
                leaderboard_df = pd.DataFrame(leaderboard, columns=["Model", "Score"])
                st.write(leaderboard_df)
            else:
                st.error("No models could be trained. Please check your data!")

if __name__ == "__main__":
    main()
