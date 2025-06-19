import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_model_phs.pkl")

model = load_model()

# Daftar fitur yang digunakan saat pelatihan model
used_features = [
    'kota_Jakarta', 'kota_Makassar', 'kota_Medan', 'kota_Semarang', 'kota_Surabaya'
    # Tambahkan fitur lain yang digunakan saat training kalau ada
]

st.title("📊 Prediksi Risiko Otomatis dengan Machine Learning")
st.write("Upload file CSV berisi data baru. Sistem akan otomatis melakukan preprocessing dan prediksi.")

uploaded_file = st.file_uploader("📂 Upload data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data mentah
        data = pd.read_csv(uploaded_file)
        st.write("📋 Data asli:")
        st.dataframe(data.head())

        # --- Preprocessing ---
        # One-hot encoding kolom kota
        kota_encoded = pd.get_dummies(data['kota'], prefix='kota')

        # Inisialisasi dataframe kosong dengan semua fitur yang dibutuhkan model
        for col in used_features:
            if col not in kota_encoded.columns:
                kota_encoded[col] = 0  # fitur tidak ada di data baru, isi 0

        # Pastikan urutan kolom sesuai model
        data_final = kota_encoded[used_features]

        # --- Prediksi ---
        pred = model.predict(data_final)
        data['prediksi'] = pred

        st.success("✅ Prediksi selesai.")
        st.write(data)

        # Download hasil
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download hasil prediksi", data=csv, file_name="hasil_prediksi.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses: {e}")

