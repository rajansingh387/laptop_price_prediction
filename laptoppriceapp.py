import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
from sklearn import *
import pickle
import streamlit as st


# Load the model and dataset
model = pickle.load(open('pipe_rf.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.title('Laptop Price Prediction')
st.header('Fill the details to predict laptop Price')



company = st.selectbox('Company',df['company'].unique())
type = st.selectbox('Type',df['typename'].unique())
ram = st.selectbox('Ram in (GB)',[8, 16, 4,2, 12,6, 32,24,64])
weight = st.number_input('Weight(in kg)')
touchscreen = st.selectbox('Touchscreen',['No','Yes'])  # actual value is [0,1]
ips = st.selectbox('IPS',['No','Yes'])              # actual value is [0,1]
cpu = st.selectbox('CPU',df['cpu brand'].unique())
hdd = st.selectbox('HDD(GB)', [0,  500, 1000, 2000,   32,  128])
ssd = st.selectbox('SSD(GB)',[128, 0,256,512,32,64,1000,1024,16,768,180,240,8])
gpu = st.selectbox('GPU',df['gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())

