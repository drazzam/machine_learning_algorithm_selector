import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Function to perform machine learning prediction
def perform_prediction(algorithm, X_train, y_train, X_test):
    if algorithm == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Gradient Boosting":
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Support Vector Machine":
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "Multi-Layer Perceptron":
        model = MLPClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == "XGBoost":
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    return y_pred

# Function to display prediction results and performance metrics
def display_results(y_test, y_pred):
    st.write("#### Prediction Results:")
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    st.write("#### Performance Metrics:")
    # Add your own performance metrics evaluation here

# Main function to run the app
def main():
    st.set_page_config(page_title="Machine Learning App", layout="wide")
    st.title("Machine Learning App")
    st.write("### Upload CSV Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        st.write("### Select Dependent and Independent Variables")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selected_columns = st.multiselect("Select Independent Variables", X.columns)
        X = X[selected_columns]
        y_column = st.selectbox("Select Dependent Variable", y)
        y = df[y_column]
        st.write("### Select Machine Learning Algorithm")
        algorithm = st.selectbox("Select Algorithm", ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine", "K-Nearest Neighbors", "Multi-Layer Perceptron", "XGBoost"])
        test_size = st.slider("Select Test Size (%)", 10, 50, 20)
        random_state = st.slider("Select Random State", 1, 100, 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        st.write("### Machine Learning Prediction")
        st.write(f"Selected Algorithm: {algorithm}")
        st.write(f"Test Size: {test_size}%")
        st.write(f"Random State: {random_state}")
        st.write("#### Selected Independent Variables:")
        st.write(selected_columns)
        st.write("#### Selected Dependent Variable:")
        st.write(y_column)
        st.write("#### Training Dataset Size:")
        st.write(len(X_train))
        st.write("#### Test Dataset Size:")
        st.write(len(X_test))
        st.write("#### Selected Machine Learning Algorithm:")
        st.write(algorithm)
        y_pred = perform_prediction(algorithm, X_train, y_train, X_test)
        display_results(y_test, y_pred)

def display_results(y_test, y_pred):
    st.write("#### Prediction Results:")
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    st.write("#### Performance Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.write(f"Precision: {precision_score(y_test, y_pred)}")
    st.write(f"Recall: {recall_score(y_test, y_pred)}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred)}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

