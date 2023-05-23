import streamlit as st
import pickle
import numpy as np
import locale

@st.cache_resource
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


@st.cache_data(experimental_allow_widgets=True)
def show_predict_page():
    st.title("Software Developer Salary Predictor")

    st.write("""### Enter the following details""")

    countries = (
        "India",
        "United States of America",
        "United Kingdom of Great Britain and Northern Ireland",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)

        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        formatted_salary = locale.format_string("%0.2f", salary[0], grouping=True)
        formatted_salary_with_commas = f"${formatted_salary}"

        st.subheader(f"The estimated salary is {formatted_salary_with_commas}")
       
