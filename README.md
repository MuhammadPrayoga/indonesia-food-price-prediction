# Dashboard Pangan Nasional 📊

Dashboard ini adalah aplikasi berbasis web yang dibangun menggunakan **Streamlit** untuk memantau dan memprediksi harga pangan nasional, serta menganalisis isu-isu terkait pangan melalui berita online.

## Fitur Utama

Aplikasi ini mengintegrasikan **Big Data Pemerintah (PIHPS)** dan **Berita Online** untuk:

1.  **Beranda (Monitoring Data)**:
    *   Menampilkan ringkasan data harga pangan dari sumber internal (pemerintah).
    *   Menampilkan ringkasan data berita eksternal.
    
2.  **Prediksi Harga (Machine Learning)**:
    *   Menggunakan algoritma **Linear Regression**.
    *   Memprediksi tren harga komoditas pangan untuk 30 hari ke depan.
    *   Visualisasi perbandingan data historis dan hasil prediksi.

3.  **Analisis Isu (Social Network Analysis)**:
    *   Memetakan hubungan antar kata kunci (Co-occurrence Network) dari judul berita.
    *   Visualisasi graf jaringan isu pangan yang sedang tren.

## Teknologi yang Digunakan

*   **Python 3**
*   **Streamlit**: Framework untuk membuat dashboard web interaktif.
*   **Pandas & NumPy**: Manipulasi dan analisis data.
*   **Scikit-Learn**: Pembuatan model Machine Learning (Linear Regression).
*   **NetworkX**: Analisis jaringan (Social Network Analysis).
*   **Matplotlib**: Visualisasi data dan grafik.
*   **GoogleNews**: (Dependencies) Pengambilan data berita.

## Struktur Folder

```
.
├── app.py                # File utama aplikasi Streamlit
├── requirements.txt      # Daftar dependensi library
├── data/                 # Folder penyimpanan data (csv, dll)
│   └── processed/        # Data yang sudah dibersihkan (data_bersih.csv, data_berita_eksternal.csv)
├── models/               # Penyimpanan model ML (jika ada)
└── notebooks/            # Jupyter Notebooks untuk cleaning/scraping/analisis awal
```

## Cara Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Prasyarat
Pastikan Anda sudah menginstall Python di komputer Anda.

### 2. Instalasi Dependensi
Disarankan untuk menggunakan *Virtual Environment*.

```bash
# (Opsional) Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install library yang dibutuhkan
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi

Jalankan perintah berikut di terminal:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

## Catatan
Pastikan file data (`data/processed/data_bersih.csv` dan `data/processed/data_berita_eksternal.csv`) tersedia agar dashboard dapat menampilkan informasi dengan benar. Jika tidak, jalankan notebook yang relevan di folder `notebooks/` terlebih dahulu.
