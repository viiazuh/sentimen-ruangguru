import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf
import numpy as np
import pickle
import io
import os
import plotly.express as px 

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- CUSTOM CSS (PRESISI FIGMA & INTER FONT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp { 
        font-family: 'Inter', sans-serif !important; 
    }
    
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }

    /* SIDEBAR CONTAINER */
    [data-testid="stSidebar"] { 
        background-color: white !important; 
        border-right: 1px solid #e5e7eb !important; 
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 2rem;
    }

    /* HEADER SIDEBAR */
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .sidebar-subtitle {
        font-size: 16px;
        color: #1e293b;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* RADIO MENU STYLING */
    div.row-widget.stRadio > div {
        gap: 15px; 
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }

    [data-testid="stSidebar"] label {
        font-size: 18px !important;
        font-weight: 400 !important;
        color: #000000 !important;
    }

    /* DASHBOARD & DATA MANAGEMENT METRIC CARD */
    .metric-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        border: 1px solid #f3f4f6; 
        margin-bottom: 1rem; 
    }
    .metric-title { color: #6b7280; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937; font-size: 1.75rem; font-weight: 700; }

    /* BUTTONS */
    .stButton>button { 
        background: #f97316 !important; 
        color: white !important; 
        border-radius: 8px !important; 
        font-weight: 600 !important; 
        border: none !important;
        width: 100%;
    }
    
    /* Tombol Download Khusus agar lebih kecil/rapi */
    [data-testid="stDownloadButton"] > button {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None
if 'page' not in st.session_state: st.session_state.page = 0 
if 'page_dashboard' not in st.session_state: st.session_state.page_dashboard = 0 

# State lokal untuk menyimpan data statistik prediksi satuan (sebagai pengganti Firebase)
if 'single_stats' not in st.session_state:
    st.session_state.single_stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

# --- MODEL LOADING ---
@st.cache_resource
def load_sentiment_model():
    try:
        model = tf.keras.models.load_model('models/model_hybrid_coc.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('models/normalization_dicts.pkl', 'rb') as f:
            norm_dict = pickle.load(f)
        return model, tokenizer, norm_dict
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model_ml, tokenizer_ml, norm_dict = load_sentiment_model()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    if norm_dict:
        normalized = [norm_dict.get(word, word) for word in words]
        return " ".join(normalized).strip()
    return text.strip()

def get_prediction(text):
    if model_ml and tokenizer_ml:
        normalized = normalize_text(text)
        seq = tokenizer_ml.texts_to_sequences([normalized])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')
        prediction = model_ml.predict(padded, verbose=0)
        labels, emojis = ["Netral", "Negatif", "Positif"], ["😐", "😞", "😀"]
        idx = np.argmax(prediction)
        return labels[idx], emojis[idx], int(np.max(prediction) * 100)
    return "Error", "⚠️", 0

# --- PENGATURAN WARNA GRAFIK ---
COLOR_MAP = {
    "Positif": "#3b82f6", 
    "Negatif": "#ef4444", 
    "Netral": "#9ca3af"
}

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Sentiment🙂</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Analisis Sentimen Ruangguru</div>', unsafe_allow_html=True)
    menu = st.radio("NAVIGATION", ["Dashboard", "Data Management", "Sentiment Prediction"])

# --- DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
    
    # ==========================================
    # BAGIAN 1: STATISTIK PREDIKSI SATUAN (SESSION STATE)
    # ==========================================
    st.markdown("<h4>📊 Statistik Prediksi Satuan (Real-time Session)</h4>", unsafe_allow_html=True)
    s = st.session_state.single_stats
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s["positif"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s["negatif"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s["netral"]}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<div style='font-size:16px; font-weight:600; margin-bottom:10px; margin-top:10px;'>Grafik Sentimen (Real-time Session)</div>", unsafe_allow_html=True)
    if s["total"] > 0:
        df_rt = pd.DataFrame({
            "Sentimen": ["Positif", "Negatif", "Netral"],
            "Jumlah Data": [s["positif"], s["negatif"], s["netral"]]
        })
        
        col_bar_rt, col_pie_rt = st.columns(2)
        with col_bar_rt:
            fig_bar_rt = px.bar(df_rt, x="Sentimen", y="Jumlah Data", color="Sentimen", color_discrete_map=COLOR_MAP, text_auto=True)
            fig_bar_rt.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_bar_rt, use_container_width=True)
            
        with col_pie_rt:
            fig_pie_rt = px.pie(df_rt, names="Sentimen", values="Jumlah Data", color="Sentimen", color_discrete_map=COLOR_MAP, hole=0.3)
            fig_pie_rt.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie_rt, use_container_width=True)
    else:
        st.info("Belum ada data grafik ulasan satuan pada sesi ini.")

    # ==========================================
    # BAGIAN 2: STATISTIK BATCH ANALYSIS (DATA MANAGEMENT)
    # ==========================================
    if st.session_state.dataset is not None:
        st.divider()
        
        filename = st.session_state.uploaded_filename
        st.markdown(f"<h4>📁 Statistik Batch Analysis (File: {filename})</h4>", unsafe_allow_html=True)
        
        df_batch = st.session_state.dataset
        batch_total = len(df_batch)
        batch_pos = len(df_batch[df_batch['Sentimen'] == "Positif"])
        batch_neg = len(df_batch[df_batch['Sentimen'] == "Negatif"])
        batch_net = len(df_batch[df_batch['Sentimen'] == "Netral"])
        
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Batch</div><div class="metric-value">{batch_total}</div></div>', unsafe_allow_html=True)
        with b2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{batch_pos}</div></div>', unsafe_allow_html=True)
        with b3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{batch_neg}</div></div>', unsafe_allow_html=True)
        with b4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{batch_net}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:16px; font-weight:600; margin-bottom:10px; margin-top:10px;'>Grafik Sentimen File Upload</div>", unsafe_allow_html=True)
        df_batch_chart = pd.DataFrame({
            "Sentimen": ["Positif", "Negatif", "Netral"],
            "Jumlah Data": [batch_pos, batch_neg, batch_net]
        })
        
        col_bar_batch, col_pie_batch = st.columns(2)
        with col_bar_batch:
            fig_bar_batch = px.bar(df_batch_chart, x="Sentimen", y="Jumlah Data", color="Sentimen", color_discrete_map=COLOR_MAP, text_auto=True)
            fig_bar_batch.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_bar_batch, use_container_width=True)
            
        with col_pie_batch:
            fig_pie_batch = px.pie(df_batch_chart, names="Sentimen", values="Jumlah Data", color="Sentimen", color_discrete_map=COLOR_MAP, hole=0.3)
            fig_pie_batch.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie_batch, use_container_width=True)
            
        # --- TABEL PREVIEW & PAGINATION DI DASHBOARD ---
        st.markdown("<div style='font-size:16px; font-weight:600; margin-top:20px; margin-bottom:10px;'>Preview Hasil Analisis</div>", unsafe_allow_html=True)
        
        items_per_page_db = 10
        total_pages_db = max(1, (batch_total + items_per_page_db - 1) // items_per_page_db)
        
        if st.session_state.page_dashboard >= total_pages_db:
            st.session_state.page_dashboard = total_pages_db - 1
            
        start_idx_db = st.session_state.page_dashboard * items_per_page_db
        end_idx_db = start_idx_db + items_per_page_db
        
        st.dataframe(df_batch.iloc[start_idx_db:end_idx_db], use_container_width=True)
        
        col_prev_db, col_info_db, col_next_db = st.columns([1, 4, 1])
        with col_prev_db:
            st.button("Prev", key="btn_prev_dash",
                      on_click=lambda: st.session_state.update(page_dashboard=st.session_state.page_dashboard - 1), 
                      disabled=(st.session_state.page_dashboard == 0), 
                      use_container_width=True)
        with col_info_db:
            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 500;'>Halaman {st.session_state.page_dashboard + 1} dari {total_pages_db}</div>", unsafe_allow_html=True)
        with col_next_db:
            st.button("Next", key="btn_next_dash",
                      on_click=lambda: st.session_state.update(page_dashboard=st.session_state.page_dashboard + 1), 
                      disabled=(st.session_state.page_dashboard >= total_pages_db - 1), 
                      use_container_width=True)

# --- DATA MANAGEMENT ---
elif menu == "Data Management":
    st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
    if uploaded_file:
        if st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.dataset = None
            st.session_state.page = 0 
            st.session_state.page_dashboard = 0

    if st.session_state.uploaded_df is not None:
        df_view = st.session_state.uploaded_df
        st.write(f"📁 **{st.session_state.uploaded_filename}** — {len(df_view)} baris")
        st.dataframe(df_view.head(5), use_container_width=True)
        
        if st.button("Analisis masal"):
            with st.spinner("Menganalisis..."):
                text_col = next((c for c in ['text', 'ulasan', 'komentar', 'textDisplay'] if c in df_view.columns), df_view.columns[0])
                texts = df_view[text_col].astype(str).tolist()
                prog = st.progress(0)
                
                normalized = [normalize_text(t) for t in texts]
                prog.progress(0.4)
                
                seqs = tokenizer_ml.texts_to_sequences(normalized)
                padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                preds = model_ml.predict(padded, batch_size=512, verbose=0)
                prog.progress(1.0)
                
                labels = ["Netral", "Negatif", "Positif"]
                res_list = [labels[np.argmax(p)] for p in preds]
                conf_list = [int(np.max(p)*100) for p in preds]
                
                st.session_state.dataset = pd.DataFrame({
                    "Text Asli": texts, 
                    "Sentimen": res_list,
                    "Probabilitas(%)": conf_list
                })
                st.session_state.page = 0 
                st.session_state.page_dashboard = 0

    if st.session_state.dataset is not None:
        st.divider()
        st.subheader("Hasil Analisis")
        
        res_df = st.session_state.dataset
        total_n = len(res_df)
        p_n = len(res_df[res_df['Sentimen'] == "Positif"])
        neg_n = len(res_df[res_df['Sentimen'] == "Negatif"])
        net_n = len(res_df[res_df['Sentimen'] == "Netral"])
        
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total</div><div class="metric-value">{total_n}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-title">😊 Positif</div><div class="metric-value">{p_n}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-title">😞 Negatif</div><div class="metric-value">{neg_n}</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-title">😐 Netral</div><div class="metric-value">{net_n}</div></div>', unsafe_allow_html=True)
        
        # --- TABEL HASIL & PAGINATION DI DATA MANAGEMENT ---
        items_per_page = 10
        total_pages = max(1, (total_n + items_per_page - 1) // items_per_page)
        
        if st.session_state.page >= total_pages:
            st.session_state.page = total_pages - 1
            
        start_idx = st.session_state.page * items_per_page
        end_idx = start_idx + items_per_page
        
        st.dataframe(res_df.iloc[start_idx:end_idx], use_container_width=True)
        
        col_prev, col_info, col_next = st.columns([1, 4, 1])
        with col_prev:
            st.button("Prev", key="btn_prev_dm",
                      on_click=lambda: st.session_state.update(page=st.session_state.page - 1), 
                      disabled=(st.session_state.page == 0), 
                      use_container_width=True)
        with col_info:
            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 500;'>Halaman {st.session_state.page + 1} dari {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            st.button("Next", key="btn_next_dm",
                      on_click=lambda: st.session_state.update(page=st.session_state.page + 1), 
                      disabled=(st.session_state.page >= total_pages - 1), 
                      use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- DOWNLOAD & DELETE ACTION BAR ---
        col_csv, col_excel, col_spacer, col_del = st.columns([1.2, 1.2, 5, 2])
        
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        col_csv.download_button("⬇️ CSV", csv_data, "hasil_sentimen.csv", "text/csv", use_container_width=True)
        
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            res_df.to_excel(writer, index=False, sheet_name='Sentimen')
        col_excel.download_button("⬇️ Excel", output_excel.getvalue(), "hasil_sentimen.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        if col_del.button("🗑️ Hapus Hasil"):
            st.session_state.dataset = None
            st.session_state.page = 0
            st.session_state.page_dashboard = 0
            st.rerun()

# --- PREDICTION ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2>Sentiment Prediction</h2>", unsafe_allow_html=True)
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Keren banget!", height=150)
        if st.button("Analisis"):
            if input_text.strip():
                res, emo, conf = get_prediction(input_text)
                
                # --- UPDATE LOCAL STATE STATISTIK ---
                st.session_state.single_stats["total"] += 1
                if res == "Positif":
                    st.session_state.single_stats["positif"] += 1
                elif res == "Negatif":
                    st.session_state.single_stats["negatif"] += 1
                elif res == "Netral":
                    st.session_state.single_stats["netral"] += 1
                
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.write(f"Probabilitas: {conf}%")
                st.progress(conf/100)
