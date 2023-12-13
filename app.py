import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
    return np.where(x == 1, 1, 0)

toy_data = {'Column1': [1, 0, 1], 'Column2': [0, 1, 1]}
toy_df = pd.DataFrame(toy_data)

cleaned_df = toy_df.applymap(clean_sm)

ss = s.copy()

ss['sm_li'] = clean_sm(ss['web1h'])

selected_features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']

ss = ss[selected_features + ['sm_li']]

education_mapping = {
    1: 'Less than high school',
    2: 'High school incomplete',
    3: 'High school graduate',
    4: 'Some college, no degree',
    5: 'Two-year associate degree',
    6: 'Four-year college or university degree',
    7: 'Some postgraduate or professional schooling',
    8: 'Postgraduate or professional degree',
    98: 'Don’t know',
    99: 'Refused'
}

marital_mapping = {
    1: 'Married',
    2: 'Living with a partner',
    3: 'Divorced',
    4: 'Separated',
    5: 'Widowed',
    6: 'Never been married',
    8: 'Don’t know',
    9: 'Refused'
}

gender_mapping = {
    1: 'Male',
    2: 'Female',
    3: 'Other',
    98: 'Don’t know',
    99: 'Refused'
}

ss['educ2'] = ss['educ2'].map(education_mapping)
ss['marital'] = ss['marital'].map(marital_mapping)
ss['gender'] = ss['gender'].map(gender_mapping)

ss = ss.dropna()

exploratory_analysis = ss.groupby('sm_li').mean(numeric_only=True)

y = ss['sm_li']

X = ss.drop('sm_li', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(include=['number']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='error'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=10))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

conf_matrix_df = pd.DataFrame(conf_matrix, columns=["Predicted Not Using S.M.", "Predicted Using S.M."], index=["Actually Not Using S.M.", "Actually Using S.M."])

classification_rep = classification_report(y_test, y_pred)

example_data_1 = {
    'income': 8,
    'educ2': 'Postgraduate or professional degree',
    'par': 2,
    'marital': 'Married',
    'gender': 'Female',
    'age': 42
}

example_data_2 = {
    'income': 8,
    'educ2': 'Postgraduate or professional degree',
    'par': 2, 
    'marital': 'Married',
    'gender': 'Female',
    'age': 82
}

example_df_1 = pd.DataFrame([example_data_1])
example_df_2 = pd.DataFrame([example_data_2])

X_example_1 = pipeline.named_steps['preprocessor'].transform(example_df_1)
X_example_2 = pipeline.named_steps['preprocessor'].transform(example_df_2)

probabilities_1 = pipeline.named_steps['classifier'].predict_proba(X_example_1)
probabilities_2 = pipeline.named_steps['classifier'].predict_proba(X_example_2)

def main():
    st.title("LinkedIn Tool Prediction")
    st.sidebar.header("Please answer a few questions:")
    income = st.sidebar.slider("What is your income?", 1, 8, 5)
    educ2 = st.sidebar.selectbox("Highest level of education?", ['Less than high school', 'High school incomplete', 'High school graduate', 'Some college, no degree', 'Two-year associate degree', 'Four-year college or university degree', 'Some postgraduate or professional schooling', 'Postgraduate or professional degree'])
    par = st.sidebar.radio("Are you a parent?", [1, 2])
    marital = st.sidebar.selectbox("What is your marital status?", ['Married', 'Living with a partner', 'Divorced', 'Separated', 'Widowed', 'Never been married'])
    gender = st.sidebar.radio("Please state your gender", ['Male', 'Female', 'Other'])
    age = st.sidebar.slider("Age", 18, 100, 30)

    input_data = pd.DataFrame({
        'income': [income],
        'educ2': [educ2],
        'par': [par],
        'marital': [marital],
        'gender': [gender],
        'age': [age]
    })

    X_input = preprocessor.transform(input_data)

    prediction = pipeline.predict(X_input)
    probabilities = pipeline.predict_proba(X_input)[:, 1]

    st.subheader("Prediction Results")
    st.write("LinkedIn User: Yes" if prediction[0] == 1 else "LinkedIn User: No")
    st.write(f"Probability of LinkedIn Usage: {probabilities[0]:.4f}")

if __name__ == "__main__":
    main()
