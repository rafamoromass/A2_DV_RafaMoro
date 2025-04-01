import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, 
                             confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle

# --- Utility function to load a Seaborn dataset (cached) ---
@st.cache_data
def load_seaborn_dataset(name):
    return sns.load_dataset(name)

# --- Sidebar: Dataset Selection ---
st.sidebar.title("Dataset Selection")
dataset_option = st.sidebar.radio("Select dataset type", ("Predefined Dataset", "Upload CSV"))

if dataset_option == "Predefined Dataset":
    dataset_name = st.sidebar.selectbox("Select a dataset", ["iris", "tips", "titanic"])
    data = load_seaborn_dataset(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# --- Main page: Data preview ---
st.write("## Data Preview")
st.dataframe(data.head())

# --- Begin configuration for model training inside a form ---
st.write("## Model Configuration and Training")
with st.form("train_form"):
    st.subheader("Data & Feature Setup")
    all_columns = data.columns.tolist()
    target_column = st.selectbox("Select Target Variable", options=all_columns)
    # Default: all columns except target are selected as features.
    feature_columns = st.multiselect("Select Feature Variables", 
                                     options=[col for col in all_columns if col != target_column],
                                     default=[col for col in all_columns if col != target_column])
    
    st.subheader("Task and Model Selection")
    task_type = st.radio("Select Task Type", options=["Regression", "Classification"])
    if task_type == "Regression":
        model_option = st.selectbox("Select Model", options=["Linear Regression", "Random Forest Regressor"])
    else:
        model_option = st.selectbox("Select Model", options=["Logistic Regression", "Random Forest Classifier"])
    
    st.subheader("Parameter Configuration")
    test_size = st.slider("Test Size (fraction)", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    
    # Additional model parameters if using Random Forest
    if "Random Forest" in model_option:
        n_estimators = st.number_input("Number of Trees", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.number_input("Max Depth (0 for None)", min_value=0, max_value=50, value=0, step=1)
    
    # Parameter for Logistic Regression
    if model_option == "Logistic Regression":
        C = st.number_input("Inverse Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    
    # Submit button to trigger model training
    submitted = st.form_submit_button("Fit Model")

# --- Process configuration and train model if form submitted ---
if submitted:
    # Check that at least one feature is selected
    if len(feature_columns) == 0:
        st.error("Please select at least one feature variable.")
        st.stop()
    
    # Copy data and select features/target
    df = data.copy()
    X = df[feature_columns]
    y = df[target_column]
    
    # Preprocessing: one-hot encode categorical features
    X_processed = pd.get_dummies(X, drop_first=True)
    
    # For classification, if target is non-numeric, encode it
    if task_type == "Classification":
        if y.dtype == 'object' or str(y.dtype).startswith("category"):
            y_processed = pd.factorize(y)[0]
        else:
            y_processed = y
    else:
        y_processed = y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=42
    )
    
    # --- Model selection and training ---
    if task_type == "Regression":
        if model_option == "Linear Regression":
            model = LinearRegression()
        elif model_option == "Random Forest Regressor":
            max_depth_val = None if max_depth == 0 else int(max_depth)
            model = RandomForestRegressor(n_estimators=int(n_estimators),
                                          max_depth=max_depth_val,
                                          random_state=42)
    else:  # Classification
        if model_option == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=1000)
        elif model_option == "Random Forest Classifier":
            max_depth_val = None if max_depth == 0 else int(max_depth)
            model = RandomForestClassifier(n_estimators=int(n_estimators),
                                           max_depth=max_depth_val,
                                           random_state=42)
    
    model.fit(X_train, y_train)
    st.session_state["trained_model"] = model  # Save model in session state
    
    # --- Model Evaluation & Visualizations ---
    st.write("## Model Performance")
    y_pred = model.predict(X_test)
    
    if task_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error:** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")
        
        # Residual distribution plot
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots()
        ax_res.hist(residuals, bins=20)
        ax_res.set_title("Residual Distribution")
        ax_res.set_xlabel("Residuals")
        ax_res.set_ylabel("Frequency")
        st.pyplot(fig_res)
        
        # Feature importance (coefficients or RF importance)
        st.write("### Feature Importance")
        if model_option == "Linear Regression":
            coef = model.coef_
            importance = pd.Series(coef, index=X_processed.columns)
        elif model_option == "Random Forest Regressor":
            importance = pd.Series(model.feature_importances_, index=X_processed.columns)
        importance = importance.sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots()
        importance.plot(kind="bar", ax=ax_imp)
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)
    
    else:  # Classification
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        cax = ax_cm.matshow(cm, cmap=plt.cm.Blues)
        fig_cm.colorbar(cax)
        for (i, j), z in np.ndenumerate(cm):
            ax_cm.text(j, i, str(z), ha='center', va='center')
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
        
        # ROC Curve for binary classification
        if len(np.unique(y_test)) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            st.write(f"**ROC AUC:** {roc_auc:.3f}")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle='--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        else:
            st.write("ROC Curve not available for multiclass classification.")
        
        # Feature importance for classification
        st.write("### Feature Importance")
        if model_option == "Logistic Regression":
            # Use absolute values of coefficients for importance
            coef = model.coef_[0]
            importance = pd.Series(np.abs(coef), index=X_processed.columns)
        elif model_option == "Random Forest Classifier":
            importance = pd.Series(model.feature_importances_, index=X_processed.columns)
        importance = importance.sort_values(ascending=False)
        fig_imp_cls, ax_imp_cls = plt.subplots()
        importance.plot(kind="bar", ax=ax_imp_cls)
        ax_imp_cls.set_title("Feature Importance")
        st.pyplot(fig_imp_cls)
    
    # --- Model Export ---
    model_data = pickle.dumps(model)
    st.download_button("Download Trained Model",
                       data=model_data,
                       file_name="trained_model.pkl",
                       mime="application/octet-stream")
