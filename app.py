import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# função que retorna a previsão (e sua probabilidade)


def predict_quality(model, df):
    predictions_data = predict_model(estimator=model, data=df)

    if predictions_data['Label'][0] == 0:
        return 'False', predictions_data['Score'][0]

    return 'True', predictions_data['Score'][0]


# carrega o modelo
model = load_model('lr_v1')


st.title("Diabetes prediction")

# seção da sidebar
st.sidebar.title("Input data")
pregnant = st.sidebar.number_input(
    "Number of times pregnant", value=0, step=1, min_value=0)
plasma_glucose = st.sidebar.number_input(
    "Plasma glucose concentration a 2 hours in an oral glucose tolerance test", value=0, step=1, min_value=0)
blood_pressure = st.sidebar.number_input(
    "Diastolic blood pressure (mm Hg)", value=0, step=1, min_value=0)
triceps = st.sidebar.number_input(
    "Triceps skin fold thickness (mm)", value=0, step=1, min_value=0)
serum_insulin = st.sidebar.number_input(
    "2-Hour serum insulin (mu U/ml)", value=0, step=1, min_value=0)
imc = st.sidebar.number_input(
    "Body mass index (weight in kg/(height in m)^2)", value=0.0, step=0.1, format="%0.1f", min_value=0.0)
pedigree = st.sidebar.number_input("Diabetes pedigree function",
                                   value=0.000, step=0.001, format="%0.3f", min_value=0.000)
age = st.sidebar.number_input("Age (years)", value=0, step=1, min_value=0)

# alinhando o botão da sidebar
_, c2, _ = st.sidebar.columns(3)
predict = c2.button("Predict")

# os atributos do modelo lidos a partir da sidebar
features = {'Number of times pregnant': pregnant,
            'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': plasma_glucose,
            'Diastolic blood pressure (mm Hg)': blood_pressure,
            'Triceps skin fold thickness (mm)': triceps,
            '2-Hour serum insulin (mu U/ml)': serum_insulin,
            'Body mass index (weight in kg/(height in m)^2)': imc,
            'Diabetes pedigree function': pedigree,
            'Age (years)': age
            }

# transforma o dicionário em um dataframe e mostra na página
features_df = pd.DataFrame([features])
st.table(features_df)

# se o botão de previsão foi clicado
if predict:
    prediction, score = predict_quality(model, features_df)
    st.write(f'Diabetes prediction: {prediction}')
    st.write(f'Prediction score: {score}')
