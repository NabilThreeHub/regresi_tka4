import pandas as pd 
import streamlit as st
import joblib

model = joblib.load("model_joblib")

st.title("Prediksi Nilai TKA")
st.markdown("Aplikasi machine learning regression untuk memprediksi nilai TKA berdasarkan fitur jumlah jam belajar, presentase kehadiran dan keikutsertaan bimbel")

jam_belajar_per_hari = st.slider("Jam Belajar Per Hari", 0, 24, 12) 
persen_kehadiran = st.slider("Presentase Kehadiran", 0, 100, 50)
bimbel = st.pills("Ikut Bimbel", ["ya", "tidak"], default="ya")


if st.button("prediksi", type="primary"):
	data_baru=pd.DataFrame([[jam_belajar_per_hari, persen_kehadiran, bimbel]],
                       columns=["jam_belajar_per_hari", "persen_kehadiran", "bimbel"])
	prediksi = model.predict(data_baru)[0]
	prediksi=prediksi.clip(0,100)
	st.success(f"Model memprediksi nilai TKA : {prediksi:.0f}")
	st.balloons()
st.divider()
st.caption("Dibuat oleh Nabil Albara")