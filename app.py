import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf  
import numpy as np        
import pickle             
import io
import firebase_admin
from firebase_admin import credentials, firestore

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        fb_credentials = dict(st.secrets["firebase"])
        cred = credentials.Certificate(fb_credentials)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

db = firestore.client()

# --- CUSTOM CSS ---
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
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS FOR FIREBASE ---
def save_to_firebase(text, result, confidence):
    try:
        data = {
            "teks": text,
            "hasil": result,
            "keyakinan": confidence,
            "waktu": firestore.SERVER_TIMESTAMP
        }
        db.collection("history_sentiment").add(data)
    except Exception as e:
        st.error(f"Gagal simpan ke database: {e}")

def get_stats_firebase():
    try:
        docs = db.collection("history_sentiment").stream()
        total, pos, neg, net = 0, 0, 0, 0
        for doc in docs:
            total += 1
            res = doc.to_dict().get("hasil", "").lower()
            if res == "positif": pos += 1
            elif res == "negatif": neg += 1
            elif res == "netral": net += 1
        return {"total": total, "positif": pos, "negatif": neg, "netral": net}
    except:
        return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_history_firebase(limit=10):
    try:
        docs = db.collection("history_sentiment").order_by("waktu", direction=firestore.Query.DESCENDING).limit(limit).stream()
        history = []
        for doc in docs:
            d = doc.to_dict()
            waktu = d['waktu'].strftime("%H:%M:%S") if d.get('waktu') else "N/A"
            history.append({"Teks": d.get("teks"), "Hasil": d.get("hasil"), "Waktu": waktu})
        return history
    except:
        return []

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

# --- PREPROCESSING & PREDICTION ---
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    normalized = [norm_dict.get(word, word) for word in words] if norm_dict else words
    return " ".join(normalized).strip()

def get_prediction(text):
    if model_ml and tokenizer_ml:
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

# --- SESSION STATE (Original for Data Management) ---
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    st.markdown("<p>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")

# --- PAGE: DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
    with st.spinner("Sinkronisasi database..."):
        s = get_stats_firebase()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s["positif"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s["negatif"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s["netral"]}</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru (Top 10)")
    hist = get_history_firebase(10)
    if hist:
        st.dataframe(pd.DataFrame(hist), use_container_width=True)
        with st.expander("Lihat Riwayat Lengkap"):
            full_hist = get_history_firebase(100)
            st.table(full_hist)
    else:
        st.info("Belum ada data di database.")

# --- PAGE: DATA MANAGEMENT (Original Functions Preserved) ---
elif menu == "Data Management":
    st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
    st.write("Proses dataset dalam jumlah besar (CSV/Excel) - **Lokal**")
    
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
                    text_col = next((c for c in ['textDisplay', 'text', 'ulasan', 'review', 'komentar', 'content'] if c in df.columns), df.columns[0])
                    texts_asli = df[text_col].astype(str).tolist()

                    progress_bar = st.progress(0, text="Preprocessing...")
                    texts_normalized = [normalize_text(t) for t in texts_asli]
                    
                    progress_bar.progress(0.4, text="Tokenisasi...")
                    seqs = tokenizer_ml.texts_to_sequences(texts_normalized)
                    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                    
                    progress_bar.progress(0.7, text="Prediksi model...")
                    all_preds = model_ml.predict(padded, batch_size=512, verbose=0)
                    progress_bar.progress(1.0, text="Selesai!")

                    labels = ["Netral", "Negatif", "Positif"]
                    results = []
                    for i in range(len(texts_asli)):
                        idx = np.argmax(all_preds[i])
                        results.append({
                            "Text Asli": texts_asli[i],
                            "Sentimen": labels[idx],
                            "Keyakinan (%)": int(np.max(all_preds[i]) * 100)
                        })

                    st.session_state.dataset = pd.DataFrame(results)
                    st.success(f"✅ Berhasil memproses {len(texts_asli)} data.")

    if st.session_state.dataset is not None:
        with st.container(border=True):
            st.dataframe(st.session_state.dataset, use_container_width=True)
            csv_data = st.session_state.dataset.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv_data, "hasil_sentimen.csv", "text/csv")

# --- PAGE: SENTIMENT PREDICTION ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2>Sentiment Prediction</h2>", unsafe_allow_html=True)
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Aplikasi ini sangat membantu!", height=150)
        if st.button("Analisis Sentimen Sekarang"):
            if input_text.strip():
                res, emo, conf = get_prediction(input_text)
                save_to_firebase(input_text, res, conf) # Simpan hasil ke Firebase
                
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.write(f"Tingkat Keyakinan: {conf}%")
                st.progress(conf/100)
            else:
                st.warning("Silakan ketikkan ulasan terlebih dahulu.")
