import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Halo ka Iqbal
# Aplikasi SNMPTN

Ini adalah aplikasi untuk memprediksi **kelulusan** di snmptn!

Data diambil dari SMAN 30 Jakarta TA 2021-2022
""")

st.sidebar.header('Input Disini')

st.sidebar.markdown("""
[Contoh file CSV](https://drive.google.com/file/d/1ekPsv3JnFRNZikDU2ktDcFJ3jA2apWMa/view?usp=sharing)
""")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        jenjang = st.sidebar.selectbox('Jenjang',('ipa','ips'))
        kode_ptn = st.sidebar.text_input('kode_ptn','351')
        kode_jurusan = st.sidebar.text_input('kode_jurusan','1026')
        rank = st.sidebar.text_input('rank','32')
        nr = st.sidebar.text_input('nr','85.473')
        mf_go = st.sidebar.text_input('mf_go','167.8')
        mk_gl = st.sidebar.text_input('mk_gl','166')
        mb_mg = st.sidebar.text_input('mb_mg','165.4')
        fk_ol = st.sidebar.text_input('fk_ol','166.2')
        fb_mo = st.sidebar.text_input('fb_mo','165.6')
        kb_ml = st.sidebar.text_input('kb_ml','163.8')
        data = {'jenjang' : jenjang,
                'kode_ptn' : kode_ptn,
                'kode_jurusan' : kode_jurusan,
                'rank' : rank,
                'nr' : nr,
                'mf_go' : mf_go,
                'mk_gl' : mk_gl,
                'mb_mg' : mb_mg,
                'fk_ol' : fk_ol,
                'fb_mo' : fb_mo,
                'kb_ml' : kb_ml}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

snmptn_raw = pd.read_csv('test streamlit sman 30.csv')
snmptn = snmptn_raw.drop(columns=['lulus'])
df = pd.concat([input_df, snmptn], axis=0)

encode = ['jenjang']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)

load_clf = pickle.load(open('snmptn_clf.pkl','rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
snmptn_lulus = np.array (['ya', 'tidak'])
st.write(snmptn_lulus[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)