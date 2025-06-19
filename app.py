import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model_phs.pkl")  # Pastikan file ini di-upload di root folder

model = load_model()

st.title("📊 Prediksi Risiko Otomatis dengan Machine Learning")
st.write("Upload file CSV berisi data baru, dan sistem akan memprediksi risiko secara otomatis.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📋 Data yang diupload:")
        st.dataframe(data.head())

        prediksi = model.predict(data)
        data['prediksi'] = prediksi

        st.success("✅ Prediksi selesai. Berikut hasilnya:")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download hasil prediksi", data=csv, file_name="hasil_prediksi.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
