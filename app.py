import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("best_model_phs.pkl")

model = load_model()

# Fitur yang digunakan saat pelatihan model
used_features = [
    'ikut_webinar_terakhir',
    'jumlah_kunjungan_3_bulan_terakhir',
    'nilai_sponsor_juta_rp',
    'kota_Yogyakarta', 'kota_Jakarta', 'kota_Makassar',
    'spesialisasi_Jantung', 'spesialisasi_Anak', 'spesialisasi_Paru'
]

st.title("📊 Prediksi Risiko Dokter")
uploaded_file = st.file_uploader("📂 Upload CSV data dokter", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Data asli:")
        st.dataframe(df.head())

        # Preprocessing:
        df_encoded = df.copy()

        # One-hot encoding kota
        kota_encoded = pd.get_dummies(df_encoded['kota'], prefix='kota')
        # One-hot encoding spesialisasi
        spek_encoded = pd.get_dummies(df_encoded['spesialisasi'], prefix='spesialisasi')

        # Gabungkan semuanya
        df_final = pd.concat([df_encoded, kota_encoded, spek_encoded], axis=1)

        # Drop kolom kategorikal asli (jika tidak digunakan langsung)
        df_final = df_final.drop(columns=['kota', 'spesialisasi', 'nama_dokter', 'dokter_id'], errors='ignore')

        # Tambahkan fitur yang hilang dengan default 0
        for col in used_features:
            if col not in df_final.columns:
                df_final[col] = 0

        # Urutkan sesuai fitur pelatihan
        df_final = df_final[used_features]

        # Prediksi
        pred = model.predict(df_final)
        df['prediksi'] = pred

        st.success("✅ Prediksi berhasil.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download hasil prediksi", data=csv, file_name="hasil_prediksi.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error saat memproses: {e}")


