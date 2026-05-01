import streamlit as st
import pandas as pd
import time
import re
import tensorflow as tf  
import numpy as np       
import pickle            
import io
import os
import firebase_admin
from firebase_admin import credentials, firestore

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- INISIALISASI FIREBASE (FIX: MENGGUNAKAN STREAMLIT SECRETS) ---
if not firebase_admin._apps:
    try:
        # Cek apakah aplikasi berjalan di Cloud (Secrets tersedia)
        if "firebase" in st.secrets:
            # Mengambil kredensial dari fitur Secrets di Streamlit Cloud
            firebase_details = dict(st.secrets["firebase"])
            
            # Memperbaiki karakter newline pada private_key agar terbaca benar oleh SDK
            if "private_key" in firebase_details:
                firebase_details["private_key"] = firebase_details["private_key"].replace("\\n", "\n")
            
            cred = credentials.Certificate(firebase_details)
            firebase_admin.initialize_app(cred)
        else:
            # Jika dijalankan lokal dan file json masih ada (untuk testing saja)
            if os.path.exists("firebase-key.json"):
                cred = credentials.Certificate("firebase-key.json")
                firebase_admin.initialize_app(cred)
            else:
                st.error("Kredensial Firebase tidak ditemukan. Pastikan sudah setting Secrets di Streamlit Cloud.")
    except Exception as e:
        st.error(f"Gagal menghubungkan ke Firebase: {e}")

try:
    db = firestore.client()
except Exception:
    db = None
    st.error("Koneksi Database Firestore gagal.")

# --- CSS FIGMA STYLE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; padding-top: 2rem; }
    [data-testid="stSidebar"] .stMarkdown p { color: #4b5563 !important; font-size: 0.95rem; font-weight: 500; margin-bottom: -10px; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; margin-bottom: 1rem; }
    .metric-title { color: #6b7280 !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.025em; }
    .metric-value { color: #1f2937 !important; font-size: 1.75rem; font-weight: 700; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
    .stButton>button { background: #f97316 !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; width: 100%; }
    @media (max-width: 768px) { .metric-card { padding: 15px; } .metric-value { font-size: 1.5rem; } }
    </style>
    """, unsafe_allow_html=True)

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
        return None, None, {}

model_ml, tokenizer_ml, norm_dict = load_sentiment_model()

# --- PREPROCESSING & PREDICTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def normalize_text(text):
    text = clean_text(text)
    words = text.split()
    normalized = [norm_dict.get(word, word) for word in words]
    return " ".join(normalized).strip()

def get_stopwords():
    return set([
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
        'adalah', 'pada', 'juga', 'dalam', 'ada', 'tidak', 'saya', 'kami',
        'kita', 'mereka', 'akan', 'sudah', 'bisa', 'karena', 'lebih', 'atau',
        'tapi', 'kalau', 'jika', 'maka', 'sangat', 'sekali', 'saja', 'aja',
        'nya', 'lah', 'deh', 'dong', 'nih', 'sih', 'ya', 'yg', 'dgn', 'utk',
        'dr', 'sy', 'gw', 'gue', 'lo', 'lu', 'aku', 'kamu', 'dia'
    ])

def remove_stopwords(text):
    stopwords = get_stopwords()
    words = str(text).split()
    return " ".join([w for w in words if w not in stopwords]).strip()

def simple_stem(word):
    prefixes = ['me', 'mem', 'men', 'meng', 'meny', 'ber', 'ter', 'per', 'ke', 'se', 'di', 'pe']
    suffixes = ['kan', 'an', 'i', 'nya']
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            word = word[len(p):]
            break
    for s in suffixes:
        if word.endswith(s) and len(word) > len(s) + 2:
            word = word[:-len(s)]
            break
    return word

def stem_text(text):
    words = str(text).split()
    return " ".join([simple_stem(w) for w in words]).strip()

def get_prediction(text):
    if model_ml:
        normalized = normalize_text(text)
        seq = tokenizer_ml.texts_to_sequences([normalized])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')
        prediction = model_ml.predict(padded, verbose=0)
        labels = ["Netral", "Negatif", "Positif"]
        emojis = ["😐", "😞", "😀"]
        idx = np.argmax(prediction)
        conf = float(np.max(prediction) * 100)
        return labels[idx], emojis[idx], int(conf)
    return "Error", "⚠️", 0

def build_excel(df_result):
    excel_buffer = io.BytesIO()
    for engine in ['xlsxwriter', 'openpyxl']:
        try:
            with pd.ExcelWriter(excel_buffer, engine=engine) as writer:
                df_result.to_excel(writer, index=False, sheet_name='Hasil Sentimen')
            return excel_buffer.getvalue(), True
        except Exception:
            excel_buffer = io.BytesIO()
            continue
    return None, False

# --- FUNGSI AMBIL DATA FIREBASE ---
def get_firebase_stats():
    if db:
        try:
            stats_ref = db.collection("global_stats").document("current_stats").get()
            if stats_ref.exists:
                return stats_ref.to_dict()
        except Exception:
            pass
    return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_firebase_history():
    if db:
        try:
            docs = db.collection("analysis_history").order_by("Waktu", direction=firestore.Query.DESCENDING).limit(50).stream()
            history_list = [doc.to_dict() for doc in docs]
            return history_list
        except Exception:
            return []
    return []

# --- SESSION STATE (LOKAL DATA MANAGEMENT) ---
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    st.markdown("<p>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    st.write("")
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")

# ==========================================
# HALAMAN 1: DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.markdown("<h2 style='color:#1f2937;'>Dashboard</h2>", unsafe_allow_html=True)
    st.write("Overview Statistik Real-time")
    
    s = get_firebase_stats()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s.get("total", 0)}</div></div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif 😊</div><div class="metric-value">{s.get("positif", 0)}</div></div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif 😞</div><div class="metric-value">{s.get("negatif", 0)}</div></div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral 😐</div><div class="metric-value">{s.get("netral", 0)}</div></div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    history_data = get_firebase_history()
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("Belum ada aktivitas analisis yang tersimpan di database.")

# ==========================================
# HALAMAN 2: DATA MANAGEMENT
# ==========================================
elif menu == "Data Management":
    st.markdown("<h2 style='color:#1f2937;'>Data Management</h2>", unsafe_allow_html=True)
    st.write("Proses dataset dalam jumlah besar (CSV/Excel)")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            if st.session_state.uploaded_filename != uploaded_file.name:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    st.session_state.uploaded_df = df_upload
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.dataset = None
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        if st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df
            st.write(f"📁 File: **{st.session_state.uploaded_filename}** — {len(df)} baris data")
            st.dataframe(df.head(10), use_container_width=True)
            
            col_btn1, col_btn2 = st.columns([2, 6])
            with col_btn1:
                run_analysis = st.button("🔍 Jalankan Batch Analysis")
            with col_btn2:
                if st.button("🗑️ Hapus File"):
                    st.session_state.uploaded_df = None
                    st.session_state.uploaded_filename = None
                    st.session_state.dataset = None
                    st.rerun()

            if run_analysis:
                with st.spinner("Sedang menganalisis dataset..."):
                    total = len(df)
                    text_col = df.columns[0]
                    for candidate in ['textDisplay', 'text', 'ulasan', 'review', 'komentar']:
                        if candidate in df.columns:
                            text_col = candidate
                            break
                    
                    texts_asli = df[text_col].astype(str).tolist()
                    
                    # Batch prediction
                    norm_texts = [normalize_text(t) for t in texts_asli]
                    seqs = tokenizer_ml.texts_to_sequences(norm_texts)
                    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                    all_preds = model_ml.predict(padded, batch_size=512, verbose=0)

                    labels = ["Netral", "Negatif", "Positif"]
                    results = []
                    for i in range(total):
                        idx = np.argmax(all_preds[i])
                        results.append({
                            "Text Asli": texts_asli[i],
                            "Sentimen": labels[idx],
                            "Keyakinan (%)": int(np.max(all_preds[i]) * 100)
                        })

                    st.session_state.dataset = pd.DataFrame(results)
                    st.success(f"✅ Batch Analysis Selesai!")

    if st.session_state.dataset is not None:
        with st.container(border=True):
            st.markdown("### Hasil Analisis")
            st.dataframe(st.session_state.dataset, use_container_width=True)
            csv_data = st.session_state.dataset.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv_data, "hasil_sentimen.csv", "text/csv")

# ==========================================
# HALAMAN 3: SENTIMENT PREDICTION
# ==========================================
elif menu == "Sentiment Prediction":
    st.markdown("<h2 style='color:#1f2937;'>Sentiment Prediction</h2>", unsafe_allow_html=True)
    st.write("Analisis teks tunggal secara real-time")
    
    with st.container(border=True):
        st.subheader("Sentiment Analysis")
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: aplikasinya mantap banget!", height=150)
        
        if st.button("Analisis Sentimen Sekarang"):
            if input_text:
                if model_ml is None:
                    st.error("Model tidak ditemukan!")
                else:
                    res, emo, conf = get_prediction(input_text)
                    waktu_sekarang = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    if db:
                        try:
                            # Simpan riwayat
                            db.collection("analysis_history").add({
                                "Teks Ulasan": input_text,
                                "Sentimen": res,
                                "Keyakinan": f"{conf}%",
                                "Waktu": waktu_sekarang
                            })
                            # Update metrik
                            db.collection("global_stats").document("current_stats").update({
                                "total": firestore.Increment(1),
                                res.lower(): firestore.Increment(1)
                            })
                        except Exception as e:
                            st.warning(f"Gagal simpan ke Firebase: {e}")
                    
                    st.divider()
                    st.markdown(f"### Hasil: {res} {emo}")
                    st.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
            else:
                st.warning("Silakan ketikkan ulasan terlebih dahulu.")
