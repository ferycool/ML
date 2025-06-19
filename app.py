import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("retrained_gradient_boosting_model.pkl")

model = load_model()

st.title("🧠 Prediksi Dokter yang Akan Membeli")
st.write("Upload file CSV atau Excel berisi data dokter baru untuk memprediksi siapa yang berpotensi membeli.")

uploaded_file = st.file_uploader("📂 Upload file CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Baca file
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Rename kolom agar sesuai jika perlu
        df.rename(columns={
            '_3_bulan_terakhir': 'jumlah_kunjungan_3_bulan_terakhir'
        }, inplace=True)

        st.subheader("📋 Data asli:")
        st.dataframe(df.head())

        # --- Preprocessing ---
        df_clean = df.copy()

        # Ubah koma jadi titik (format angka)
        df_clean['nilai_sponsor_juta_rp'] = df_clean['nilai_sponsor_juta_rp'].astype(str).str.replace(",", ".").astype(float)
        df_clean['total_pembelian_tahun_lalu_juta_rp'] = df_clean['total_pembelian_tahun_lalu_juta_rp'].astype(str).str.replace(",", ".").astype(float)

        # One-hot encoding untuk kolom kategorikal
        kota_encoded = pd.get_dummies(df_clean['kota'], prefix='kota')
        spes_encoded = pd.get_dummies(df_clean['spesialisasi'], prefix='spesialisasi')
        rs_encoded = pd.get_dummies(df_clean['tipe_rumah_sakit'], prefix='tipe_rumah_sakit')

        df_features = pd.concat([
            df_clean[['tahun_praktik', 'jumlah_kunjungan_3_bulan_terakhir', 'ikut_webinar_terakhir',
                      'nilai_sponsor_juta_rp', 'total_pembelian_tahun_lalu_juta_rp']],
            kota_encoded, spes_encoded, rs_encoded
        ], axis=1)

        # Ambil daftar fitur dari model
        expected_features = model.feature_names_in_

        # Tambahkan kolom yang hilang agar sesuai
        for col in expected_features:
            if col not in df_features.columns:
                df_features[col] = 0

        # Susun sesuai urutan saat model dilatih
        df_features = df_features[expected_features]

        # --- Prediksi ---
        df['prediksi'] = model.predict(df_features)

        # 🔍 Distribusi hasil prediksi
        st.subheader("🔎 Distribusi Prediksi")
        st.write(df['prediksi'].value_counts())
        st.bar_chart(df['prediksi'].value_counts())

        # ✅ Tampilkan semua data + prediksi
        st.success("✅ Prediksi selesai.")
        st.dataframe(df[['nama_dokter', 'kota', 'spesialisasi', 'nilai_sponsor_juta_rp', 'prediksi']])

        # 🧾 Tampilkan dokter yang diprediksi akan beli
        df_beli = df[df['prediksi'] == 1]
        st.subheader("🧾 Dokter yang Diprediksi Akan Membeli")
        if df_beli.empty:
            st.warning("Tidak ada dokter yang diprediksi akan membeli.")
        else:
            st.dataframe(df_beli[['nama_dokter', 'kota', 'spesialisasi', 'nilai_sponsor_juta_rp', 'prediksi']])
            csv_beli = df_beli.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download hanya dokter yang beli", data=csv_beli,
                               file_na_
