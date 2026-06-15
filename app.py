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
import plotly.graph_objects as go
import plotly.figure_factory as ff

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
        
    .stTextArea textarea {
        font-size: 18px !important;
    }

    /* Membesarkan teks di dalam TABEL (Data Management) */
    [data-testid="stDataFrame"], [data-testid="stTable"], table, th, td {
        font-size: 16px !important;
    }

    /* Membesarkan judul-judul halaman */
    h2 { font-size: 34px !important; font-weight: 700 !important; }
    h3 { font-size: 26px !important; font-weight: 600 !important; }
    h4 { font-size: 22px !important; font-weight: 600 !important; }

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
        font-size: 40px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .sidebar-subtitle {
        font-size: 25px;
        color: #1e293b;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* RADIO MENU STYLING */
    div.row-widget.stRadio > div {
        gap: 18px; 
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }

    [data-testid="stSidebar"] label {
        font-size: 22px !important;
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
    .metric-title { 
        color: #6b7280; 
        font-size: 1.1rem; 
        font-weight: 600; 
        text-transform: uppercase; 
    }
    .metric-value { 
        color: #1f2937; 
        font-size: 2rem; 
        font-weight: 700; 
    }

    /* TOMBOL UTAMA */

    div[data-testid="stButton"] > button {
        background-color: #f97316 !important;
        color: #ffffff !important;
        border: 1px solid #f97316 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #ea580c !important;
        border-color: #ea580c !important;
        color: #ffffff !important;
    }

    /* ========================================= */
    /* 2. TOMBOL UPLOAD FILE (Tidak Berwarna)    */
    /* ========================================= */
    div[data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        border-color: #f97316 !important;
        color: #f97316 !important;
        background-color: #fffaf5 !important;
    }

    /* TOMBOL DOWNLOAD */
    div[data-testid="stDownloadButton"] > button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #0f172a !important;
        border-color: #0f172a !important;
        color: #ffffff !important;
    }

    /* TOMBOL HAPUS HASIL */
    div[data-testid="column"]:nth-of-type(4) div[data-testid="stButton"] > button {
        background-color: #ffffff !important;
        color: #ef4444 !important; /* Teks Merah */
        border: 1px solid #ef4444 !important; /* Garis Tepi Merah */
    }
    div[data-testid="column"]:nth-of-type(4) div[data-testid="stButton"] > button:hover {
        background-color: #ef4444 !important; /* Background jadi merah saat disentuh */
        color: #ffffff !important; /* Teks jadi putih saat disentuh */
        border-color: #ef4444 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# =========================================================================
# DEFINISI  struktur untuk load file PKL agar pkl tidak error
# =========================================================================
class TextPreprocessor:
    def __init__(self, *args, **kwargs):
        pass
    def transform(self, text):
        return text
    def fit(self, X, y=None):
        return self

class KerasPredictor:
    def __init__(self, *args, **kwargs):
        pass
    def transform(self, text):
        return text
    def fit(self, X, y=None):
        return self
    
# --- SESSION STATE INITIALIZATION ---
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None
if 'page' not in st.session_state: st.session_state.page = 0 
if 'page_dashboard' not in st.session_state: st.session_state.page_dashboard = 0 

if 'single_stats' not in st.session_state:
    st.session_state.single_stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_sentiment_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'models', 'best_model_S1_&_S2_tanpa_SMOTE.h5')
        pkl_path = os.path.join(BASE_DIR, 'models', 'pipeline_s12_raw.pkl')
        

        with open(pkl_path, 'rb') as f:
            pipeline = pickle.load(f)
            
        model = tf.keras.models.load_model(model_path)
        return model, pipeline
    except Exception as e:
        st.error(f"Gagal memuat model/pipeline: {e}")
        return None, None

model_ml, pipeline_ml = load_sentiment_model()

# --- HELPER: PENCARI TOKENIZER OTOMATIS  ---
def find_keras_tokenizer(obj, depth=0):
    """Mencari Keras Tokenizer sampai ke akar-akar objek Ferdinan"""
    if depth > 5: return None
    if hasattr(obj, 'texts_to_sequences'): return obj
    
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k.startswith('__'): continue
            res = find_keras_tokenizer(v, depth+1)
            if res: return res
            
    if isinstance(obj, (list, tuple)):
        for item in obj:
            res = find_keras_tokenizer(item, depth+1)
            if res: return res
            
    if isinstance(obj, dict):
        for k, v in obj.items():
            res = find_keras_tokenizer(v, depth+1)
            if res: return res
            
    return None

def extract_sequences(pipeline, texts_list):
    """Mengekstraksi token secara aman, disamakan persis dengan Colab (.lower() saja)"""
    # Regex re.sub dihapus agar inputnya sama persis dengan Colab
    clean_texts = [str(t).lower() for t in texts_list]
    tokenizer = find_keras_tokenizer(pipeline)
    
    if tokenizer:
        seqs = tokenizer.texts_to_sequences(clean_texts)
        return seqs if seqs else [[0]]
    return None

def get_prediction(text):
    if model_ml and pipeline_ml:
        try:
            seq = extract_sequences(pipeline_ml, [text])
            if seq is None: return "Error Tokenizer", "⚠️", 0
            
            # PERBAIKAN 1: maxlen jadi 32 & truncating='post' sesuai Colab
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=32, padding='post', truncating='post')
            prediction = model_ml.predict(padded, verbose=0)
            
            labels, emojis = ["Negatif", "Netral", "Positif"], ["😞", "😐", "😀"]
            idx = np.argmax(prediction)
            
            return labels[idx], emojis[idx], int(np.max(prediction) * 100)
        except Exception as e:
            return f"Error: {e}", "⚠️", 0
    return "Error Model", "⚠️", 0

# --- PENGATURAN WARNA GRAFIK ---
COLOR_MAP = {"Positif": "#3b82f6", "Negatif": "#ef4444", "Netral": "#9ca3af"}

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Sentiment🙂</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Analisis Sentimen Ruangguru</div>', unsafe_allow_html=True)
    menu = st.radio("NAVIGATION", ["Dashboard", "Data Management", "Sentiment Prediction"])

# --- DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4>📊 Statistik Analisis Tunggal (Real-time Session)</h4>", unsafe_allow_html=True)
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
            
        st.markdown("<div style='font-size:16px; font-weight:600; margin-top:20px; margin-bottom:10px;'>Daftar Hasil Pengujian Sesi Aktif</div>", unsafe_allow_html=True)
        df_history = pd.DataFrame(reversed(st.session_state.prediction_history))
        st.dataframe(df_history, use_container_width=True)
            
    else:
        st.info("Belum ada data grafik maupun daftar pengujian tunggal pada sesi ini.")

    if st.session_state.dataset is not None:
        st.divider()
        filename = st.session_state.uploaded_filename
        df_batch = st.session_state.dataset
        batch_total = len(df_batch)
        batch_pos = len(df_batch[df_batch['Sentimen'] == "Positif"])
        batch_neg = len(df_batch[df_batch['Sentimen'] == "Negatif"])
        batch_net = len(df_batch[df_batch['Sentimen'] == "Netral"])
        
        # --- FITUR CONFUSION MATRIX DAN AKURASI ---
        if 'Label Asli' in df_batch.columns:
            correct_preds = (df_batch['Label Asli'] == df_batch['Sentimen']).sum()
            accuracy_val = (correct_preds / batch_total) * 100
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981, #059669); padding: 30px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px;">
                    <h3 style="margin: 0; font-size: 1.5rem; font-weight: 500;">Akurasi Analisis File Massal</h3>
                    <h1 style="margin: 0; font-size: 4.5rem; font-weight: 800;">{accuracy_val:.1f}%</h1>
                    <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">File: {filename} ({batch_total} Baris Data)</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4>Confusion Matrix Hasil Analisis Massal</h4>", unsafe_allow_html=True)
            
            categories = ['Positif', 'Negatif', 'Netral']
            df_batch['Label Asli'] = pd.Categorical(df_batch['Label Asli'], categories=categories)
            df_batch['Sentimen'] = pd.Categorical(df_batch['Sentimen'], categories=categories)
            
            ct = pd.crosstab(df_batch['Label Asli'], df_batch['Sentimen'], dropna=False)
            
            z_dynamic = ct.values.tolist()
            x_dynamic = [f"Prediksi {c}" for c in ct.columns]
            y_dynamic = [f"Aktual {r}" for r in ct.index]
            
            fig_cm = ff.create_annotated_heatmap(z_dynamic, x=x_dynamic, y=y_dynamic, colorscale='Blues', showscale=True)
            fig_cm.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), font=dict(size=14))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.divider()
        # --- AKHIR FITUR CONFUSION MATRIX ---

        st.markdown(f"<h4>📁 Statistik Analisis Massal (File: {filename})</h4>", unsafe_allow_html=True)
        
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Massal</div><div class="metric-value">{batch_total}</div></div>', unsafe_allow_html=True)
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
            
        st.markdown("<div style='font-size:16px; font-weight:600; margin-top:20px; margin-bottom:10px;'>Preview Hasil Analisis Massal</div>", unsafe_allow_html=True)
        
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
                      disabled=(st.session_state.page_dashboard == 0), use_container_width=True)
        with col_info_db:
            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 500;'>Halaman {st.session_state.page_dashboard + 1} dari {total_pages_db}</div>", unsafe_allow_html=True)
        with col_next_db:
            st.button("Next", key="btn_next_dash",
                      on_click=lambda: st.session_state.update(page_dashboard=st.session_state.page_dashboard + 1), 
                      disabled=(st.session_state.page_dashboard >= total_pages_db - 1), use_container_width=True)

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
        
        if st.button("Jalankan Analisis Massal"):
            with st.spinner("Sedang menganalisis dataset secara massal..."):
                text_col = next((c for c in ['text', 'ulasan', 'komentar', 'textDisplay'] if c in df_view.columns), df_view.columns[0])
                texts = df_view[text_col].astype(str).tolist()
                
                seqs = extract_sequences(pipeline_ml, texts)
                
                if seqs is None:
                    st.error("Gagal melakukan tokenisasi. Tokenizer asli Keras tidak ditemukan.")
                else:
                    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=32, padding='post', truncating='post')
                    preds = model_ml.predict(padded, batch_size=512, verbose=0)
                    
                    # PERBAIKAN 2: Urutan disesuaikan (0: Negatif, 1: Netral, 2: Positif)
                    labels = ["Negatif", "Netral", "Positif"]
                    res_list = [labels[np.argmax(p)] for p in preds]
                    conf_list = [int(np.max(p)*100) for p in preds]
                    
                    # --- FITUR DETEKSI LABEL ASLI UNTUK MATRIKS ---
                    data_result = {
                        "Text Asli": texts, 
                        "Sentimen": res_list,
                        "Probabilitas(%)": conf_list
                    }
                    
                   # --- DETEKSI LABEL ASLI & MAPPING OTOMATIS ---
                    label_col = next((c for c in ['label', 'sentimen', 'sentiment', 'actual', 'Label Asli', 'sentimen_asli'] if c in df_view.columns), None)
                    if label_col:
                        # Ambil data mentah dan jadikan string
                        raw_labels = df_view[label_col].astype(str).str.strip()
                        
                        # Mapping otomatis dari angka ke teks
                        mapping_angka = {'1': 'Positif', '1.0': 'Positif', '0': 'Netral', '0.0': 'Netral', '-1': 'Negatif', '-1.0': 'Negatif'}
                        mapped_labels = raw_labels.map(lambda x: mapping_angka.get(x, x)).str.capitalize()
                        data_result["Label Asli"] = mapped_labels.values
                    else:
                        potential_cols = [c for c in df_view.columns if c != text_col]
                        if potential_cols:
                            data_result["Label Asli"] = df_view[potential_cols[0]].astype(str).str.strip().str.capitalize().values
                    
                    st.session_state.dataset = pd.DataFrame(data_result)
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
        
        items_per_page = 10
        total_pages = max(1, (total_n + items_per_page - 1) // items_per_page)
        
        if st.session_state.page >= total_pages:
            st.session_state.page = total_pages - 1
            
        start_idx = st.session_state.page * items_per_page
        end_idx = start_idx + items_per_page
        
        st.dataframe(res_df.iloc[start_idx:end_idx], use_container_width=True)
        
        col_prev, col_info, col_next = st.columns([1, 4, 1])
        with col_prev:
            st.button("Prev", key="btn_prev_down",
                      on_click=lambda: st.session_state.update(page=st.session_state.page - 1), 
                      disabled=(st.session_state.page == 0), use_container_width=True)
        with col_info:
            st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: 500;'>Halaman {st.session_state.page + 1} dari {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            st.button("Next", key="btn_next_down",
                      on_click=lambda: st.session_state.update(page=st.session_state.page + 1), 
                      disabled=(st.session_state.page >= total_pages - 1), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
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
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Seru banget!", height=150)
        
        if st.button("Analisis"):
            if input_text.strip():
                with st.spinner("Sedang menganalisis sentimen..."):
                    time.sleep(0.6) # Jeda animasi sebentar
                    res, emo, conf = get_prediction(input_text)
                
                st.session_state.single_stats["total"] += 1
                if res == "Positif": st.session_state.single_stats["positif"] += 1
                elif res == "Negatif": st.session_state.single_stats["negatif"] += 1
                elif res == "Netral": st.session_state.single_stats["netral"] += 1
                
           
                st.session_state.prediction_history.append({
                    "Teks Ulasan": input_text,
                    "Hasil Klasifikasi": f"{res} {emo}",
                    "Tingkat Keyakinan": f"{conf}%"
                })
                
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.write(f"Probabilitas: {conf}%")
                st.progress(conf/100)