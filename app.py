import streamlit as st
import pandas as pd
import time
import re
import tensorflow as tf  
import numpy as np       
import pickle            
import io

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. CSS CUSTOM (LABEL DISESUAIKAN) ---
st.markdown("""
    <style>
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; }
    
    /* STYLE LABEL BULAT ORANYE */
    .label-skripsi {
        display: inline-block;
        background-color: #fb923c;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        text-align: center;
        line-height: 24px;
        font-weight: bold;
        margin-right: 8px;
        font-size: 14px;
        vertical-align: middle;
    }

    .metric-card {
        background-color: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f3f4f6;
        display: flex; justify-content: space-between; align-items: center;
    }
    .icon-box { width: 35px; height: 35px; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
    .bg-blue { background-color: #dbeafe; color: #3b82f6; }
    .bg-green { background-color: #d1fae5; color: #10b981; }
    .bg-red { background-color: #fee2e2; color: #ef4444; }
    .bg-gray { background-color: #f3f4f6; color: #6b7280; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MOCK MODEL (BIAR SS BERSIH) ---
@st.cache_resource
def load_mock():
    return True

load_mock()

# --- 4. SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 15662, "positif": 4477, "negatif": 4585, "netral": 6600}
if 'history' not in st.session_state:
    st.session_state.history = [{"Teks": "Aplikasi ini sangat bagus!", "Hasil": "Positif", "Waktu": "10:00:01"}]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### Sentiment🙂")
    st.caption("Project Analisis Sentimen Ruangguru")
    menu = st.radio("MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")

# --- 6. HALAMAN DASHBOARD ---
if menu == "Dashboard":
    st.header("Dashboard")
    st.write("Overview Statistik Real-time")
    
    # Deskripsi a: Panel Metrik Statistik
    st.markdown("#### <span class='label-skripsi'>a</span> Panel Metrik Statistik", unsafe_allow_html=True)
    s = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    # Deskripsi b: Visualisasi Ikonik (Ditaruh di dalam kartu)
    with c1: st.markdown(f'<div class="metric-card"><div><small>Total Data</small><div><b>{s["total"]}</b></div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div><small>Positif</small><div><b>{s["positif"]}</b></div></div><div class="icon-box bg-green"><span class="label-skripsi" style="width:18px;height:18px;line-height:18px;font-size:10px;margin:0;">b</span>😊</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div><small>Negatif</small><div><b>{s["negatif"]}</b></div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div><small>Netral</small><div><b>{s["netral"]}</b></div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.write("")
    # Deskripsi c: Tabel Riwayat Analisis
    st.markdown("#### <span class='label-skripsi'>c</span> Tabel Riwayat Analisis", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# --- 7. HALAMAN DATA MANAGEMENT ---
elif menu == "Data Management":
    st.header("Data Management")
    
    # Deskripsi a: Komponen File Uploader
    st.markdown("#### <span class='label-skripsi'>a</span> Komponen File Uploader", unsafe_allow_html=True)
    st.file_uploader("Upload dataset (.csv/.xlsx)", type=["csv", "xlsx"], label_visibility="collapsed")
    
    # Deskripsi b: Panel Pratinjau Data
    st.markdown("#### <span class='label-skripsi'>b</span> Panel Pratinjau Data", unsafe_allow_html=True)
    mock_df = pd.DataFrame({"publishedAt": ["2024-07-03"], "author": ["User1"], "textDisplay": ["Bagus banget!"], "likeCount": [10]})
    st.dataframe(mock_df, use_container_width=True)

    # Deskripsi c: Tombol Kontrol Operasi
    st.write("")
    st.markdown("<span class='label-skripsi'>c</span> <b>Tombol Kontrol Operasi</b>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    col1.button("🔍 Jalankan Batch Analysis")
    col2.button("🗑️ Hapus File")

# --- 8. HALAMAN SENTIMENT PREDICTION ---
elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    
    with st.container(border=True):
        # Deskripsi a: Input Text Area
        st.markdown("#### <span class='label-skripsi'>a</span> Input Text Area", unsafe_allow_html=True)
        input_text = st.text_area("Masukkan ulasan", placeholder="Ketik di sini...", label_visibility="collapsed")
        
        # Deskripsi b: Tombol Analisis
        st.write("")
        st.markdown("<span class='label-skripsi'>b</span> <b>Tombol Analisis</b>", unsafe_allow_html=True)
        btn = st.button("Analisis Sentimen Sekarang")
        
        if btn or input_text:
            st.divider()
            # Deskripsi c: Panel Output Prediksi
            st.markdown("#### <span class='label-skripsi'>c</span> Panel Output Prediksi", unsafe_allow_html=True)
            res_col, conf_col = st.columns(2)
            res_col.success("Hasil: Positif 😀")
            conf_col.progress(0.98, text="Confidence Score: 98%")
