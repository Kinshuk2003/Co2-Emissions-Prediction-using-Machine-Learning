import streamlit as st
import pandas as pd

import numpy as np
import base64
from joblib import dump,load

from sklearn.decomposition import PCA
import base64





st.write("""
# CO2 Emission Prediction App
This app predicts the **Ammount of CO2 Emission** 
""")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\fg.png')    
    

st.sidebar.header('Input Parameters')

def user_input_features():
    country=st.sidebar.slider('country', 0.0, 1.0,0.1)
    year=st.sidebar.slider('year', 0.0, 1.0,0.1)
    co2_per_capita = st.sidebar.slider('co2_per_capita', 0.0, 1.0,0.1)
    trade_co2 = st.sidebar.slider('trade_co2', 0.0, 1.0,0.1)
    cement_co2 = st.sidebar.slider('cement_co2', 0.0, 1.0,0.1)
    cement_co2_per_capita = st.sidebar.slider('cement_co2_per_capita', 0.0, 1.0,0.1)
    coal_co2 = st.sidebar.slider('coal_co2', 0.0, 1.0,0.1)
    coal_co2_per_capita	= st.sidebar.slider('coal_co2_per_capita', 0.0, 1.0,0.1)
    flaring_co2	= st.sidebar.slider('flaring_co2', 0.0, 1.0,0.1)
    flaring_co2_per_capita	= st.sidebar.slider('flaring_co2_per_capita', 0.0, 1.0,0.1)
    gas_co2	= st.sidebar.slider('gas_co2', 0.0, 1.0,0.1)
    gas_co2_per_capita	= st.sidebar.slider('gas_co2_per_capita', 0.0, 1.0,0.1)
    oil_co2	= st.sidebar.slider('oil_co2', 0.0, 1.0,0.1)
    oil_co2_per_capita	= st.sidebar.slider('oil_co2_per_capita', 0.0, 1.0,0.1)
    other_industry_co2	= st.sidebar.slider('other_industry_co2', 0.0, 1.0,0.1)
    other_co2_per_capita	= st.sidebar.slider('other_co2_per_capita', 0.0, 1.0,0.1)
    co2_growth_prct	= st.sidebar.slider('co2_growth_prct', 0.0, 1.0,0.1)
    co2_growth_abs	= st.sidebar.slider('co2_growth_abs', 0.0, 1.0,0.1)
    co2_per_gdp	= st.sidebar.slider('co2_per_gdp', 0.0, 1.0,0.1)
    co2_per_unit_energy	= st.sidebar.slider('co2_per_unit_energy', 0.0, 1.0,0.1)
    consumption_co2	= st.sidebar.slider('consumption_co2', 0.0, 1.0,0.1)
    consumption_co2_per_capita	= st.sidebar.slider('consumption_co2_per_capita', 0.0, 1.0,0.1)
    consumption_co2_per_gdp	= st.sidebar.slider('consumption_co2_per_gdp', 0.0, 1.0,0.1)
    cumulative_co2	= st.sidebar.slider('cumulative_co2', 0.0, 1.0,0.1)
    cumulative_cement_co2	= st.sidebar.slider('cumulative_cement_co2', 0.0, 1.0,0.1)
    cumulative_coal_co2	= st.sidebar.slider('cumulative_coal_co2', 0.0, 1.0,0.1)
    cumulative_flaring_co2	= st.sidebar.slider('cumulative_flaring_co2', 0.0, 1.0,0.1)
    cumulative_gas_co2	= st.sidebar.slider('cumulative_gas_co2', 0.0, 1.0,0.1)
    cumulative_oil_co2	= st.sidebar.slider('cumulative_oil_co2', 0.0, 1.0,0.1)
    cumulative_other_co2	= st.sidebar.slider('cumulative_other_co2', 0.0, 1.0,0.1)
    trade_co2_share	= st.sidebar.slider('trade_co2_share', 0.0, 1.0,0.1)
    share_global_co2	= st.sidebar.slider('share_global_co2', 0.0, 1.0,0.1)
    share_global_cement_co2	= st.sidebar.slider('share_global_cement_co2', 0.0, 1.0,0.1)
    share_global_coal_co2	= st.sidebar.slider('share_global_coal_co2', 0.0, 1.0,0.1)
    share_global_flaring_co2	= st.sidebar.slider('share_global_flaring_co2', 0.0, 1.0,0.1)
    share_global_gas_co2	= st.sidebar.slider('share_global_gas_co2', 0.0, 1.0,0.1)
    share_global_oil_co2	= st.sidebar.slider('share_global_oil_co2', 0.0, 1.0,0.1)
    share_global_other_co2	= st.sidebar.slider('share_global_other_co2', 0.0, 1.0,0.1)
    share_global_cumulative_co2	= st.sidebar.slider(' share_global_cumulative_co2', 0.0,1.0,0.1)
    share_global_cumulative_cement_co2	= st.sidebar.slider('share_global_cumulative_cement_co2', 0.0, 1.0,0.1)
    share_global_cumulative_coal_co2	= st.sidebar.slider('share_global_cumulative_coal_co2', 0.0, 1.0,0.1)
    share_global_cumulative_flaring_co2	= st.sidebar.slider('share_global_cumulative_flaring_co2', 0.0, 1.0,0.1)
    share_global_cumulative_gas_co2	= st.sidebar.slider(' share_global_cumulative_gas_co2', 0.0, 1.0,0.1)
    share_global_cumulative_oil_co2	= st.sidebar.slider(' share_global_cumulative_oil_co2', 0.0, 1.0,0.1)
    share_global_cumulative_other_co2	= st.sidebar.slider('share_global_cumulative_other_co2', 0.0, 1.0,0.1)
    total_ghg	= st.sidebar.slider(' total_ghg', 0.0, 1.0,0.1)
    ghg_per_capita	= st.sidebar.slider(' ghg_per_capita', 0.0, 1.0,0.1)
    total_ghg_excluding_lucf	= st.sidebar.slider(' total_ghg_excluding_lucf', 0.0, 1.0,0.1)
    ghg_excluding_lucf_per_capita	= st.sidebar.slider('ghg_excluding_lucf_per_capita', 0.0, 1.0,0.1)
    methane	= st.sidebar.slider('methane', 0.0, 1.0,0.1)
    methane_per_capita	= st.sidebar.slider('methane_per_capita', 0.0, 1.0,0.1)
    nitrous_oxide	= st.sidebar.slider(' nitrous_oxide', 0.0, 1.0,0.1)
    nitrous_oxide_per_capita	= st.sidebar.slider('nitrous_oxide_per_capita', 0.0, 1.0,0.1)
    population	= st.sidebar.slider('population', 0.0, 1.0,0.1)
    gdp	= st.sidebar.slider('gdp', 0.0, 1.0,0.1)
    primary_energy_consumption	= st.sidebar.slider('  primary_energy_consumption', 0.0, 1.0,0.1)
    energy_per_capita	= st.sidebar.slider('energy_per_capita', 0.0, 1.0,0.1)
    energy_per_gdp	= st.sidebar.slider(' energy_per_gdp', 0.0, 1.0,0.1)

    data = {'country':country,
            'year':year,
            'co2_per_capita':co2_per_capita,
            'trade_co2':trade_co2,
            'cement_co2':cement_co2,
            'cement_co2_per_capita':cement_co2_per_capita,
            'coal_co2':coal_co2,
            'coal_co2_per_capita':coal_co2_per_capita,
            'flaring_co2':flaring_co2,
            'flaring_co2_per_capita':flaring_co2_per_capita,
            'gas_co2':gas_co2,
            'gas_co2_per_capita':gas_co2_per_capita,
            'oil_co2':oil_co2, 
            'oil_co2_per_capita':oil_co2_per_capita,
            'other_industry_co2':other_industry_co2,
            'other_co2_per_capita':other_co2_per_capita,
            'co2_growth_prct':co2_growth_prct,
            'co2_growth_abs':co2_growth_abs,
            'co2_per_gdp':co2_per_gdp,
            'co2_per_unit_energy':co2_per_unit_energy,
            'consumption_co2':consumption_co2,
            'consumption_co2_per_capita':consumption_co2_per_capita,
            'consumption_co2_per_gdp':consumption_co2_per_gdp,
            'cumulative_co2':cumulative_co2,
            'cumulative_cement_co2':cumulative_cement_co2,
            'cumulative_coal_co2':cumulative_coal_co2,
            'cumulative_flaring_co2':cumulative_flaring_co2,
            'cumulative_gas_co2':cumulative_gas_co2,
            'cumulative_oil_co2':cumulative_oil_co2,
            'cumulative_other_co2':cumulative_other_co2,
            'trade_co2_share':trade_co2_share,
            'share_global_co2':share_global_co2,
            'share_global_cement_co2':share_global_cement_co2,
            'share_global_coal_co2':share_global_coal_co2,
            'share_global_flaring_co2':share_global_flaring_co2,
            'share_global_gas_co2':share_global_gas_co2,
            'share_global_oil_co2':share_global_oil_co2,
            'share_global_other_co2':share_global_other_co2,
            'share_global_cumulative_co2':share_global_cumulative_co2,
            'share_global_cumulative_cement_co2':share_global_cumulative_cement_co2,
            'share_global_cumulative_coal_co2':share_global_cumulative_coal_co2,
            'share_global_cumulative_flaring_co2':share_global_cumulative_flaring_co2,
            'share_global_cumulative_gas_co2':share_global_cumulative_gas_co2,
            'share_global_cumulative_oil_co2':share_global_cumulative_oil_co2,
            'share_global_cumulative_other_co2':share_global_cumulative_other_co2,
            'total_ghg':total_ghg,
            'ghg_per_capita':ghg_per_capita,
            'total_ghg_excluding_lucf':total_ghg_excluding_lucf,
            'ghg_excluding_lucf_per_capita':ghg_excluding_lucf_per_capita,
            'methane':methane,
            'methane_per_capita':methane_per_capita,
            'nitrous_oxide':nitrous_oxide,
            'nitrous_oxide_per_capita':nitrous_oxide_per_capita,
            'population':population,
            'gdp':gdp,
            'primary_energy_consumption':primary_energy_consumption,
            'energy_per_capita':energy_per_capita,
            'energy_per_gdp':energy_per_gdp,
            
            }
    
    features = pd.DataFrame(data, index=[0.0])
    return features

df = user_input_features()

st.subheader('Input parameters')
st.write(df)



pca= load(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\PCA.joblib')

P = pca.transform(df)

regr = load(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\LinearRegression.joblib')
regr1 = load(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\KnnRegression.joblib')
regr2 = load(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\DTRegression.joblib')
regr3 = load(r'C:\Users\Leeladhar Royal\Desktop\ml app\Machine_Learning_WebApp\RidgeRegression.joblib')

prediction = regr.predict(P)
prediction1 = regr1.predict(P)
prediction2 = regr2.predict(P)
prediction3 = regr3.predict(P)

st.subheader('Predicted co2 emission (in PPM)')

regression= {'Linear Regression': prediction,
            'Knn Regression': prediction1,
            'Decision Tree' : prediction2,
            'Ridge Regression': prediction3
        }


fea = pd.DataFrame(regression, index=[0.0])

st.subheader('Predictions Using Different Regressors')
st.write(fea)








