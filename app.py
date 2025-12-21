import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Pangan Nasional", layout="wide")

# --- JUDUL & INTRO ---
st.title("📊 Dashboard Monitoring & Prediksi Harga Pangan")
st.markdown("""
Aplikasi ini mengintegrasikan **Big Data Pemerintah (PIHPS)** dan **Berita Online** untuk:
1.  Memantau tren harga pangan.
2.  Memprediksi harga 30 hari ke depan (Machine Learning).
3.  Menganalisis isu/sentimen publik (Social Network Analysis).
""")
st.markdown("---")

# --- SIDEBAR MENU ---
menu = st.sidebar.selectbox("Pilih Menu Analisis:", 
                            ["🏠 Beranda", "📈 Prediksi Harga (ML)", "🕸️ Analisis Isu (SNA)"])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Load Data Harga
    try:
        df_harga = pd.read_csv('data/processed/data_bersih.csv')
        df_harga['Tanggal'] = pd.to_datetime(df_harga['Tanggal'])
    except:
        df_harga = pd.DataFrame()
    
    # Load Data Berita
    try:
        df_berita = pd.read_csv('data/processed/data_berita_eksternal.csv')
    except:
        df_berita = pd.DataFrame()
        
    return df_harga, df_berita

df_harga, df_berita = load_data()

# ==============================================================================
# HALAMAN 1: BERANDA
# ==============================================================================
if menu == "🏠 Beranda":
    st.subheader("Sekilas Data Pangan Nasional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📂 Data Internal (Pemerintah)")
        if not df_harga.empty:
            st.write(f"Jumlah Data: {len(df_harga)} baris")
            st.write("Komoditas:", df_harga['Item'].unique())
            st.dataframe(df_harga.head(), height=200)
        else:
            st.warning("Data harga belum ditemukan. Jalankan notebook cleaning dulu.")

    with col2:
        st.success("🌍 Data Eksternal (Berita)")
        if not df_berita.empty:
            st.write(f"Jumlah Berita: {len(df_berita)} artikel")
            st.dataframe(df_berita[['Title', 'Media', 'Date']].head(), height=200)
        else:
            st.warning("Data berita belum ditemukan. Jalankan notebook scraping dulu.")

# ==============================================================================
# HALAMAN 2: PREDIKSI HARGA (MACHINE LEARNING)
# ==============================================================================
elif menu == "📈 Prediksi Harga (ML)":
    st.subheader("Prediksi Tren Harga Masa Depan")
    st.write("Menggunakan algoritma **Linear Regression** untuk meramal harga 30 hari ke depan.")
    
    if not df_harga.empty:
        # Pilih Komoditas
        pilihan_item = st.selectbox("Pilih Komoditas:", df_harga['Item'].unique())
        
        # Filter Data
        df_item = df_harga[df_harga['Item'] == pilihan_item].copy()
        
        # --- PROSES ML DI BALIK LAYAR ---
        # 1. Feature Engineering
        df_item['Date_Ordinal'] = df_item['Tanggal'].map(dt.datetime.toordinal)
        X = df_item[['Date_Ordinal']]
        y = df_item['Harga']
        
        # 2. Training Model
        model = LinearRegression()
        model.fit(X, y)
        
        # 3. Forecasting (30 Hari)
        last_date_ord = df_item['Date_Ordinal'].max()
        future_days = 30
        future_dates_ord = np.array([last_date_ord + i for i in range(1, future_days+1)]).reshape(-1, 1)
        prediksi_ml = model.predict(future_dates_ord)
        future_dates = [dt.date.fromordinal(int(val)) for val in future_dates_ord.flatten()]
        
        # --- VISUALISASI ---
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Garis Data Asli
        ax.plot(df_item['Tanggal'], df_item['Harga'], label='Data Historis (Aktual)', color='blue', linewidth=2)
        
        # Garis Prediksi
        ax.plot(future_dates, prediksi_ml, label='Prediksi AI (30 Hari)', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f"Forecast Harga {pilihan_item}", fontsize=14)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (Rp)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Tampilkan Tabel Prediksi
        with st.expander("Lihat Angka Detail Prediksi"):
            df_pred = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Harga (Rp)': prediksi_ml})
            st.dataframe(df_pred)
            
    else:
        st.error("Data harga kosong!")

# ==============================================================================
# HALAMAN 3: SOCIAL NETWORK ANALYSIS (SNA)
# ==============================================================================
elif menu == "🕸️ Analisis Isu (SNA)":
    st.subheader("Analisis Jejaring Isu Pangan")
    st.write("Memetakan hubungan kata (Co-occurrence Network) dari judul berita yang discraping.")
    
    if not df_berita.empty:
        # --- PROSES SNA ---
        # 1. Gabung teks
        text_data = " ".join(df_berita['Title'].tolist()).lower()
        for char in "-.,|?":
            text_data = text_data.replace(char, "")
        words = text_data.split()
        
        # 2. Buat Graph
        G = nx.Graph()
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if len(w1) > 3 and len(w2) > 3: # Filter kata pendek
                if G.has_edge(w1, w2):
                    G[w1][w2]['weight'] += 1
                else:
                    G.add_edge(w1, w2, weight=1)
        
        # 3. Filter Node agar tidak terlalu ruwet (Hanya tampilkan yg sering muncul)
        # Ambil node yang punya koneksi lebih dari 1
        core_nodes = [node for node, degree in dict(G.degree()).items() if degree > 1]
        G_sub = G.subgraph(core_nodes)
        
        col_sna1, col_sna2 = st.columns([3, 1])
        
        with col_sna1:
            # 4. Gambar Graph
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G_sub, k=0.6, seed=42)
            nx.draw(G_sub, pos, with_labels=True, 
                    node_size=[v * 100 for v in dict(G_sub.degree()).values()], 
                    node_color='skyblue', 
                    font_size=10, 
                    edge_color="gray", 
                    alpha=0.7)
            st.pyplot(fig)
            
        with col_sna2:
            st.write("**Top Kata Kunci:**")
            degrees = sorted(G_sub.degree, key=lambda x: x[1], reverse=True)[:10]
            for word, count in degrees:
                st.write(f"- {word.title()} ({count} koneksi)")
                
    else:
        st.error("Data berita kosong!")