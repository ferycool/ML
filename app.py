import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("best_model_phs.pkl")

model = load_model()

# Fitur yang digunakan saat training
used_features = [
    'tahun_praktik',
    'jumlah_kunjungan_3_bulan_terakhir',
    'ikut_webinar_terakhir',
    'nilai_sponsor_juta_rp',
    'total_pembelian_tahun_lalu_juta_rp',  # ✅ Tambahan wajib
    'kota_Bandung', 'kota_Jakarta', 'kota_Medan', 'kota_Semarang', 'kota_Surabaya', 'kota_Yogyakarta',
    'spesialisasi_Anak', 'spesialisasi_Jantung', 'spesialisasi_Kandungan', 'spesialisasi_Kulit',
    'spesialisasi_Paru', 'spesialisasi_THT', 'spesialisasi_Umum',
    'tipe_rumah_sakit_Klinik Mandiri', 'tipe_rumah_sakit_Pemerintah', 'tipe_rumah_sakit_Swasta Tipe A'
]

st.title("🧠 Prediksi Dokter yang Akan Membeli")
st.write("Upload data CSV atau Excel (.csv / .xlsx) dari dokter baru untuk diprediksi siapa yang berpotensi membeli.")

uploaded_file = st.file_uploader("📂 Upload file CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Baca file
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Ganti nama kolom jika perlu (mapping untuk cocok dengan model)
        df.rename(columns={
            '_3_bulan_terakhir': 'jumlah_kunjungan_3_bulan_terakhir'
        }, inplace=True)

        st.subheader("📋 Data asli:")
        st.dataframe(df.head())

        # --- Preprocessing ---
        df_clean = df.copy()

        # Format koma ke float
        df_clean['nilai_sponsor_juta_rp'] = df_clean['nilai_sponsor_juta_rp'].astype(str).str.replace(",", ".").astype(float)
        df_clean['total_pembelian_tahun_lalu_juta_rp'] = df_clean['total_pembelian_tahun_lalu_juta_rp'].astype(str).str.replace(",", ".").astype(float)

        # One-hot encoding kategori
        kota_encoded = pd.get_dummies(df_clean['kota'], prefix='kota')
        spes_encoded = pd.get_dummies(df_clean['spesialisasi'], prefix='spesialisasi')
        rs_encoded = pd.get_dummies(df_clean['tipe_rumah_sakit'], prefix='tipe_rumah_sakit')

        df_features = pd.concat([
            df_clean[['tahun_praktik', 'jumlah_kunjungan_3_bulan_terakhir',
                      'ikut_webinar_terakhir', 'nilai_sponsor_juta_rp',
                      'total_pembelian_tahun_lalu_juta_rp']],
            kota_encoded, spes_encoded, rs_encoded
        ], axis=1)

        # Tambah kolom yang hilang
        for col in used_features:
            if col not in df_features.columns:
                df_features[col] = 0

        # Urutkan kolom sesuai training
        df_features = df_features[used_features]

        # --- Prediksi ---
        df['prediksi'] = model.predict(df_features)

        # 🔍 Distribusi prediksi
        st.subheader("🔎 Distribusi Prediksi")
        st.write(df['prediksi'].value_counts())
        st.bar_chart(df['prediksi'].value_counts())

        # ✅ Tampilkan semua hasil
        st.success("✅ Prediksi selesai.")
        st.dataframe(df[['nama_dokter', 'kota', 'spesialisasi', 'nilai_sponsor_juta_rp', 'prediksi']])

        # 🧾 Filter: dokter yang beli
        df_beli = df[df['prediksi'] == 1]
        st.subheader("🧾 Dokter yang Diprediksi Akan Membeli")
        if df_beli.empty:
            st.warning("Tidak ada dokter yang diprediksi akan membeli.")
        else:
            st.dataframe(df_beli[['nama_dokter', 'kota', 'spesialisasi', 'nilai_sponsor_juta_rp', 'prediksi']])
            csv_beli = df_beli.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download hanya dokter yang beli", data=csv_beli,
                               file_name="dokter_yang_beli.csv", mime='text/csv')

        # ⬇️ Download semua hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download semua hasil", data=csv,
                           file_name="prediksi_dokter.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses: {e}")
