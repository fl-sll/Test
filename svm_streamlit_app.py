# svm_streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the breast cancer dataset
@st.cache_data
def load_data():
    breast_cancer = datasets.load_breast_cancer()
    return breast_cancer

# Preprocess the data (split and scale)
@st.cache_data
def preprocess_data(breast_cancer):
    X = breast_cancer.data
    y = breast_cancer.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Train the SVM model
@st.cache_data
def train_svm(X_train, y_train, kernel='linear', C=1.0, gamma='scale'):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# Main Streamlit app function
def main():
    st.title("Breast Cancer Classification using SVM")
    st.write("This app allows you to explore SVM classification using the breast cancer dataset from scikit-learn.")
    
    # Load the dataset
    breast_cancer = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(breast_cancer)
    
    # Sidebar for user input
    st.sidebar.header("Model Hyperparameters")
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.sidebar.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    
    # Train the model with user-selected parameters
    svm_model = train_svm(X_train, y_train, kernel=kernel, C=C, gamma=gamma)
    
    # Make predictions and evaluate
    y_pred = svm_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    # Display results
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    
    # Display confusion matrix and classification report
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, target_names=breast_cancer.target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Visualize the confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
