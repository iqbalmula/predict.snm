import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
	st.title('Assalamualaikum Iqbal Muhammad :)')
	st.text('Projek ini untuk observasi dataset dari lokasi 104')


with dataset:
	st.header('104 dataset')
	st.text('Dataset dari tahun 2010 sampai 2015')

	kenari_data = pd.read_csv('streamlit 104.csv')
	st.write(kenari_data.head())

	st.subheader('Distribusi churn di lokasi kenari')
	churn_dist = pd.DataFrame(kenari_data['churn'].value_counts())
	st.bar_chart(churn_dist)

with features:
	st.header('Fitur-fitur yang dibuat')

	st.markdown('* **fitur churn:** adalah fitur yang menunjukkan pelanggan melakukan churn atau tidak')
	st.markdown('* **fitur pekerjaan:** adalah fitur yang menunjukkan pekerjaan orang tua')

with model_training:
	st.header('Waktu untuk training model')
	st.text('Pilih hyperparameter dari model dan lihat perubahan performance')

	sel_col, disp_col = st.columns(2)

	max_depth = sel_col.slider('Berapa max_depth yang diinginkan dari model?', min_value=10, max_value=100, value=20, step=10)

	n_estimators = sel_col.selectbox('Berapa banyak pohon yang diinginkan?', options=[100,200,300,'No Limit'], index=0)

	input_feature = sel_col.text_input('Fitur apa yang ingin diobservasi?','pekerjaan')

	regr = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

	X = kenari_data[[input_feature]]
	y = kenari_data[['pekerjaan']]

	regr.fit(X,y)
	prediction = (regr.predict(y))

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y, predicition))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y, predicition))

	disp_col.subheader('R squared error of the model is:')	
	disp_col.write(R_squared_error(y, predicition))