import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt

# Try importing WordCloud (Optional dependency handling)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# --- IMPORT OPTIONAL MODULES ---
try:
    from GoogleNews import GoogleNews
    GOOGLE_NEWS_AVAILABLE = True
except ImportError:
    GOOGLE_NEWS_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dashboard Pangan Nasional", 
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# --- COORDINATE DICTIONARY (USER PROVIDED) ---
PROVINCE_COORDS = {
    "Aceh": [4.6951, 96.7494], "Sumatera Utara": [2.1154, 99.5451], "Sumatera Barat": [-0.7399, 100.8000],
    "Riau": [0.5104, 101.4478], "Kepulauan Riau": [3.9167, 108.0000], "Jambi": [-1.4852, 102.4381],
    "Bengkulu": [-3.5778, 102.3464], "Sumatera Selatan": [-3.3194, 104.9144], "Kepulauan Bangka Belitung": [-2.7411, 106.4406],
    "Lampung": [-4.5586, 105.4068], "Banten": [-6.4058, 106.0640], "DKI Jakarta": [-6.2088, 106.8456],
    "Jawa Barat": [-7.0909, 107.6689], "Jawa Tengah": [-7.1510, 110.1403], "DI Yogyakarta": [-7.8754, 110.4262],
    "Jawa Timur": [-7.5361, 112.2384], "Bali": [-8.3405, 115.0920], "Nusa Tenggara Barat": [-8.6529, 117.3616],
    "Nusa Tenggara Timur": [-8.6574, 121.0794], "Kalimantan Barat": [-0.2787, 111.4753], "Kalimantan Tengah": [-1.6815, 113.3824],
    "Kalimantan Selatan": [-3.0926, 115.2838], "Kalimantan Timur": [0.5387, 116.4194], "Kalimantan Utara": [3.0731, 116.0414],
    "Sulawesi Utara": [0.6247, 123.9750], "Gorontalo": [0.6999, 122.4467], "Sulawesi Tengah": [-1.4300, 121.4456],
    "Sulawesi Barat": [-2.8441, 119.2321], "Sulawesi Selatan": [-3.6687, 119.9740], "Sulawesi Tenggara": [-4.1449, 122.1746],
    "Maluku": [-3.2385, 130.1453], "Maluku Utara": [1.5709, 127.8087], "Papua Barat": [-1.3361, 133.1747],
    "Papua": [-4.2699, 138.0804]
}

# --- CUSTOM HEADER ---
def render_header():
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header p {
            color: #d1d8e0;
            margin-top: 5px;
            font-size: 1.1rem;
        }
        </style>
        <div class="main-header">
            <h1>Dashboard Monitoring & Prediksi Harga Pangan</h1>
            <p>Integrasi Big Data PIHPS & Analisis Sentimen Berita Online</p>
        </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910795.png", width=100)
    st.markdown("### Navigasi")
    menu = st.radio("", 
        ["🏠 Beranda", "📈 Prediksi Harga (ML)", "🗺️ Peta Geografis", "🕸️ Analisis Isu (SNA)"],
        index=0
    )
    st.markdown("---")

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=3600)
def load_main_data():
    """Load internal processed data and live news."""
    # Internal Data
    try:
        df_harga = pd.read_csv('data/processed/data_bersih.csv')
        df_harga['Tanggal'] = pd.to_datetime(df_harga['Tanggal'])
    except:
        df_harga = pd.DataFrame()
    
    # Newspaper Logic (Simulated/Scraped)
    try:
        # Check if saved file exists first
        df_berita = pd.read_csv('data/processed/data_berita_eksternal.csv')
    except:
        df_berita = pd.DataFrame()
        
    return df_harga, df_berita

df_harga_internal, df_berita_internal = load_main_data()

# ==============================================================================
# MENU 1: BERANDA
# ==============================================================================
if menu == "🏠 Beranda":
    render_header()
    
    st.subheader("📊 Ringkasan Eksekutif")
    m1, m2, m3, m4 = st.columns(4)
    
    # Calculate simple metrics
    if not df_harga_internal.empty:
        curr_price = df_harga_internal['Harga'].iloc[-1]
        prev_price = df_harga_internal['Harga'].iloc[-2] if len(df_harga_internal) > 1 else curr_price
        delta = curr_price - prev_price
        delta_pct = (delta / prev_price) * 100 if prev_price != 0 else 0
        commodities = df_harga_internal['Item'].nunique()
    else:
        curr_price, delta, delta_pct, commodities = 0, 0, 0, 0
        
    with m1: st.metric("Harga Nasional (Avg)", f"Rp {curr_price:,.0f}", f"{delta_pct:.2f}%")
    with m2: st.metric("Jumlah Komoditas", f"{commodities}", "Item")
    with m3: st.metric("Total Data Isu", f"{len(df_berita_internal)}", "Artikel")
    with m4: st.metric("Status Sistem", "Online", "Normal")
    
    st.markdown("---")
    t1, t2 = st.tabs(["📂 Data PIHPS (Internal)", "🌍 Berita (Eksternal)"])
    with t1: st.dataframe(df_harga_internal, use_container_width=True, height=300)
    with t2: st.dataframe(df_berita_internal, use_container_width=True, height=300)

# ==============================================================================
# MENU 2: PREDIKSI HARGA (ML - UPGRADED)
# ==============================================================================
elif menu == "📈 Prediksi Harga (ML)":
    render_header()
    st.subheader("📈 Forecasting Harga Komoditas")
    
    if not df_harga_internal.empty:
        # Input
        col_item, _ = st.columns([1,3])
        with col_item:
            item_ml = st.selectbox("Pilih Komoditas:", df_harga_internal['Item'].unique())
            
        df_ml = df_harga_internal[df_harga_internal['Item'] == item_ml].copy()
        
        # Linear Regression Logic
        df_ml['Date_Ordinal'] = df_ml['Tanggal'].map(dt.datetime.toordinal)
        X = df_ml[['Date_Ordinal']]
        y = df_ml['Harga']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast 30 Days
        last_date = df_ml['Date_Ordinal'].max()
        future_dates_ord = np.array([last_date + i for i in range(1, 31)]).reshape(-1, 1)
        pred_price = model.predict(future_dates_ord)
        future_dates_real = [dt.date.fromordinal(int(x)) for x in future_dates_ord.flatten()]
        
        # --- 1. SMART INSIGHT (TOP) ---
        last_actual_price = df_ml['Harga'].iloc[-1]
        last_predicted_price = pred_price[-1]
        change_val = last_predicted_price - last_actual_price
        change_pct = (change_val / last_actual_price) * 100
        
        if change_val > 0:
            st.error(f"⚠️ **Insight:** Harga diprediksi **NAIK** sebesar **{change_pct:.2f}%** (Rp {change_val:,.0f}) dalam 30 hari ke depan.")
        else:
            st.success(f"✅ **Insight:** Harga diprediksi **TURUN** sebesar **{abs(change_pct):.2f}%** (Rp {abs(change_val):,.0f}) dalam 30 hari ke depan.")
            
        # --- 2. INTERACTIVE CHART (MIDDLE) ---
        fig = go.Figure()
        
        # Historical Data (Blue)
        fig.add_trace(go.Scatter(
            x=df_ml['Tanggal'], 
            y=df_ml['Harga'], 
            mode='lines+markers', 
            name='Data Asli', 
            line=dict(color='#1e3c72', width=2),
            hovertemplate="<b>Tanggal:</b> %{x|%d %b %Y}<br><b>Harga:</b> Rp %{y:,.0f}<extra></extra>"
        ))
        
        # Forecast Data (Red Dashed)
        fig.add_trace(go.Scatter(
            x=future_dates_real, 
            y=pred_price, 
            mode='lines+markers', 
            name='Prediksi (30 Hari)', 
            line=dict(color='#ff4b4b', dash='dash', width=2),
            hovertemplate="<b>Tanggal:</b> %{x|%d %b %Y}<br><b>Prediksi:</b> Rp %{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Trend & Prediksi Harga: {item_ml}",
            xaxis_title="Periode Waktu",
            yaxis_title="Harga (Rp)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 3. DETAIL DATA IN EXPANDER (BOTTOM) ---
        with st.expander("Lihat Angka Detail Prediksi"):
            # Create readable dataframe
            df_forecast = pd.DataFrame({
                'Tanggal': future_dates_real,
                'Prediksi Harga (Rp)': pred_price.astype(int)
            })
            
            # Show Dataframe
            st.dataframe(df_forecast, use_container_width=True)
            
            # Download Button
            csv = df_forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Prediksi (CSV)",
                data=csv,
                file_name='prediksi_harga_30_hari.csv',
                mime='text/csv'
            )
            
    else:
        st.warning("Data internal tidak tersedia untuk prediksi.")

# ==============================================================================
# MENU 3: PETA GEOGRAFIS (KEEP - NO CHANGES)
# ==============================================================================
elif menu == "🗺️ Peta Geografis":
    render_header()
    st.subheader("🗺️ Peta Pantauan Harga (Dot Map)")
    
    # 1. LOAD DATA FILES
    COMMODITY_FILES = {
        "Beras": "data/raw/harga_beras_prov.csv",
        "Cabai": "data/raw/harga_cabai_prov.csv",
        "Minyak Goreng": "data/raw/harga_minyak_prov.csv",
        "Bawang Merah": "data/raw/harga_bawang_prov.csv",
    }
    
    col_sel, _ = st.columns([1,3])
    with col_sel:
        komoditas_pilihan = st.selectbox("Pilih Komoditas:", list(COMMODITY_FILES.keys()))
        
    @st.cache_data
    def load_clean_data(file_path):
        try:
            df = pd.read_csv(file_path)
            # STRICT REQUIREMENT: Column 1 is Province, Column 2 is Price. Column 0 is No.
            clean_df = pd.DataFrame()
            clean_df['Provinsi'] = df.iloc[:, 1] # Index 1
            clean_df['Harga'] = df.iloc[:, 2]    # Index 2
            
            # Cleaning
            clean_df = clean_df[clean_df['Provinsi'] != 'Semua Provinsi'].copy()
            clean_df['Harga'] = clean_df['Harga'].astype(str).str.replace(',', '', regex=False).str.replace('.', '', regex=False)
            clean_df['Harga'] = pd.to_numeric(clean_df['Harga'], errors='coerce')
            clean_df.dropna(subset=['Provinsi', 'Harga'], inplace=True)
            return clean_df
        except Exception as e:
            return pd.DataFrame()

    df_map = load_clean_data(COMMODITY_FILES[komoditas_pilihan])
    
    if not df_map.empty:
        # 2. COORDINATE MAPPING
        def get_lat_lon(prov_name):
            safe_name = str(prov_name).strip()
            if safe_name in PROVINCE_COORDS:
                return PROVINCE_COORDS[safe_name]
            return None

        # Apply coordinates
        coords = df_map['Provinsi'].apply(get_lat_lon)
        df_map['lat'] = coords.apply(lambda x: x[0] if x else None)
        df_map['lon'] = coords.apply(lambda x: x[1] if x else None)
        df_map = df_map.dropna(subset=['lat', 'lon'])
        
        if not df_map.empty:
            # 3. VISUALIZATION LOGIC
            avg_price = df_map['Harga'].mean()
            
            def set_color(price):
                if price > avg_price:
                    return [255, 0, 0, 200] # RED
                else:
                    return [0, 255, 0, 200] # GREEN
            
            df_map['color'] = df_map['Harga'].apply(set_color)
            df_map['size'] = 50000 
            
            st.map(df_map, latitude='lat', longitude='lon', color='color', size='size')
            
            c1, c2, c3 = st.columns(3)
            expensive = df_map.loc[df_map['Harga'].idxmax()]
            cheaper = df_map.loc[df_map['Harga'].idxmin()]
            
            with c1: st.metric("Rata-rata Nasional", f"Rp {avg_price:,.0f}")
            with c2: st.metric("Provinsi Termahal", expensive['Provinsi'], f"Rp {expensive['Harga']:,}")
            with c3: st.metric("Provinsi Termurah", cheaper['Provinsi'], f"Rp {cheaper['Harga']:,}")
            
            st.caption(f"Visualisasi Dot Map: Merah = Diatas Rata-rata ({avg_price:,.0f}), Hijau = Dibawah Rata-rata.")
            
            with st.expander("Lihat Data Tabel"):
                st.dataframe(df_map[['Provinsi', 'Harga']].sort_values(by='Harga', ascending=False), use_container_width=True)
        else:
            st.error("Gagal mapping koordinat. Cek nama provinsi di CSV.")
    else:
        st.error("Data kosong. Pastikan file uploaded.")

# ==============================================================================
# MENU 4: SNA (UPGRADED WITH TABS & WORDCLOUD)
# ==============================================================================
elif menu == "🕸️ Analisis Isu (SNA)":
    render_header()
    st.subheader("🕸️ Analisis Isu & Topik Berita")
    
    if not df_berita_internal.empty:
        # Preprocessing common for both stats
        text_str = " ".join(df_berita_internal['Title'].tolist()).lower()
        for char in "-.,|?": text_str = text_str.replace(char, "")
        words = text_str.split()
        
        # Determine stopwords and clean list
        STOPWORDS = ["dan", "yang", "di", "ke", "untuk", "ini", "itu", "dari", "dengan", "pada", "dalam", "baca", "juga"]
        clean_words = [w for w in words if w not in STOPWORDS and len(w) >= 4]
        
        # --- TABS LAYOUT ---
        tab_net, tab_wc = st.tabs(["🕸️ Jaringan Koneksi (Network)", "☁️ Word Cloud"])
        
        # --- TAB 1: NETWORK GRAPH ---
        with tab_net:
            # Build Graph
            G = nx.Graph()
            for i in range(len(words)-1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:
                    if G.has_edge(words[i], words[i+1]): 
                        G[words[i]][words[i+1]]['weight'] += 1
                    else:
                        G.add_edge(words[i], words[i+1], weight=1)
            
            if len(G.nodes) > 0:
                core_nodes = [n for n, d in G.degree() if d > 1]
                G_sub = G.subgraph(core_nodes) if core_nodes else G
                
                pos = nx.spring_layout(G_sub, seed=42)
                edge_x, edge_y = [], []
                for edge in G_sub.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                
                node_x, node_y, node_text, node_size = [], [], [], []
                for node in G_sub.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node} ({G_sub.degree(node)})")
                    node_size.append(G_sub.degree(node) * 6)
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y, 
                    text=node_text, 
                    mode='markers+text', 
                    hoverinfo='text',
                    textposition='top center',
                    marker=dict(size=node_size, color=node_size, colorscale='Viridis', showscale=True)
                )
                
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    showlegend=False, 
                    margin=dict(b=0,l=0,r=0,t=0), 
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Data belum cukup untuk Network Graph.")
                
        # --- TAB 2: WORD CLOUD ---
        with tab_wc:
            if WORDCLOUD_AVAILABLE and len(clean_words) > 0:
                text_clean = " ".join(clean_words)
                wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_clean)
                
                # Display using Matplotlib
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            elif not WORDCLOUD_AVAILABLE:
                st.error("Visualisasi WordCloud memerlukan library 'wordcloud'.")
            else:
                st.info("Tidak ada kata kunci yang cukup.")
        
        # --- KEYWORD STATS (Common for both) ---
        st.markdown("---")
        st.subheader("📊 Statistik Detail Kata Kunci")
        
        word_counts = Counter(clean_words)
        most_common = word_counts.most_common(15)
        df_freq = pd.DataFrame(most_common, columns=['Kata Kunci', 'Frekuensi']).sort_values(by='Frekuensi', ascending=True)
        
        c_chart, c_piew = st.columns([2, 1])
        with c_chart:
            fig_bar = px.bar(df_freq, x='Frekuensi', y='Kata Kunci', orientation='h', title='Top 15 Keywords')
            fig_bar.update_traces(marker_color='#2a5298')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_piew:
            st.write("Tabel Frekuensi:")
            df_table = df_freq.sort_values(by='Frekuensi', ascending=False).reset_index(drop=True)
            df_table.index += 1
            st.dataframe(df_table, height=300, use_container_width=True)
            
    else:
        st.warning("Data berita kosong.")