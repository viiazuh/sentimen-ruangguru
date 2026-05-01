import streamlit as st
import pandas as pd
import time
import re
import tensorflow as tf  
import numpy as np       
import pickle            
import io
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. INISIALISASI FIREBASE (KEAMANAN SECRETS) ---
# Menggunakan logika 'if secrets else file' agar tidak error di Cloud
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            # Rencana A: Berjalan di Streamlit Cloud menggunakan Secrets
            firebase_details = dict(st.secrets["firebase"])
            # Penting: Mengubah string \n menjadi karakter newline asli agar kunci terbaca
            if "private_key" in firebase_details:
                firebase_details["private_key"] = firebase_details["private_key"].replace("\\n", "\n")
            
            cred = credentials.Certificate(firebase_details)
            firebase_admin.initialize_app(cred)
        else:
            # Rencana B: Berjalan lokal di Garuda Linux menggunakan file fisik
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

# Inisialisasi Firestore Client
try:
    db = firestore.client()
except Exception:
    db = None

# --- 3. CSS CUSTOM (INTER FONT & FIGMA STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; padding-top: 2rem; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; margin-bottom: 1rem; }
    .metric-title { color: #6b7280 !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937 !important; font-size: 1.75rem; font-weight: 700; }
    .stButton>button { background: #f97316 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; height: 45px; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_sentiment_model():
    try:
        # Load model Hybrid LSTM-GRU dan komponen pendukung
        model = tf.keras.models.load_model('models/model_hybrid_coc.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('models/normalization_dicts.pkl', 'rb') as f:
            norm_dict = pickle.load(f)
        return model, tokenizer, norm_dict
    except Exception:
        return None, None, {}

model_ml, tokenizer_ml, norm_dict = load_sentiment_model()

# --- 5. PREPROCESSING & PREDICTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def normalize_text(text):
    text = clean_text(text)
    words = text.split()
    # Menggunakan kamus normalisasi untuk memperbaiki kata tidak baku
    normalized = [norm_dict.get(word, word) for word in words]
    return " ".join(normalized).strip()

def get_stopwords():
    # Daftar kata henti manual sesuai dengan metodologi di skripsi kamu
    return set(['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan', 'adalah', 'pada', 'juga', 'dalam', 'ada', 'tidak', 'saya', 'kami'])

def remove_stopwords(text):
    sw = get_stopwords()
    words = str(text).split()
    return " ".join([w for w in words if w not in sw]).strip()

def simple_stem(word):
    # Stemming sederhana untuk mendukung proses pra-pemrosesan teks
    prefixes = ['me', 'mem', 'men', 'meng', 'meny', 'ber', 'ter', 'per', 'ke', 'se', 'di', 'pe']
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            word = word[len(p):]
            break
    return word

def stem_text(text):
    words = str(text).split()
    return " ".join([simple_stem(w) for w in words]).strip()

def get_prediction(text):
    if model_ml:
        normalized = normalize_text(text)
        # Konversi teks ke urutan angka sesuai tokenizer
        seq = tokenizer_ml.texts_to_sequences([normalized])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')
        prediction = model_ml.predict(padded, verbose=0)
        labels = ["Netral", "Negatif", "Positif"]
        emojis = ["😐", "😞", "😀"]
        idx = np.argmax(prediction)
        conf = float(np.max(prediction) * 100)
        return labels[idx], emojis[idx], int(conf)
    return "Error", "⚠️", 0

# --- 6. FUNGSI DATABASE (BAHASA INDONESIA) ---
def get_firebase_stats():
    if db:
        try:
            # Mengambil statistik dari koleksi bahasa Indonesia sesuai permintaan
            stats_ref = db.collection("statistik_global").document("data_terkini").get()
            if stats_ref.exists:
                return stats_ref.to_dict()
        except Exception:
            pass
    return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_firebase_history():
    if db:
        try:
            # Mengambil riwayat analisis terbaru
            docs = db.collection("riwayat_analisis").order_by("Waktu", direction=firestore.Query.DESCENDING).limit(50).stream()
            return [doc.to_dict() for doc in docs]
        except Exception:
            return []
    return []

# --- 7. SIDEBAR ---
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
    st.write("Overview Statistik Real-time (Firebase)")
    
    s = get_firebase_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s.get("total", 0)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s.get("positif", 0)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s.get("negatif", 0)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s.get("netral", 0)}</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    history_data = get_firebase_history()
    if history_data:
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("Belum ada riwayat tersimpan di database.")

# ==========================================
# HALAMAN 2: DATA MANAGEMENT (BATCH ANALYSIS)
# ==========================================
elif menu == "Data Management":
    st.markdown("<h2 style='color:#1f2937;'>Data Management</h2>", unsafe_allow_html=True)
    st.write("Proses dataset massal (Lokal Sesi)")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🔍 Jalankan Batch Analysis"):
            with st.spinner("Menganalisis data..."):
                text_col = df.columns[0]
                texts_asli = df[text_col].astype(str).tolist()
                
                processed_texts = [normalize_text(t) for t in texts_asli]
                seqs = tokenizer_ml.texts_to_sequences(processed_texts)
                padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                
                all_preds = model_ml.predict(padded, batch_size=256, verbose=0)
                labels = ["Netral", "Negatif", "Positif"]
                
                results = []
                for i in range(len(texts_asli)):
                    idx = np.argmax(all_preds[i])
                    results.append({
                        "Teks Asli": texts_asli[i],
                        "Hasil": labels[idx],
                        "Keyakinan": f"{int(np.max(all_preds[i])*100)}%"
                    })
                
                st.session_state.batch_result = pd.DataFrame(results)
                st.success("Analisis Batch Selesai!")

    if "batch_result" in st.session_state:
        st.dataframe(st.session_state.batch_result, use_container_width=True)

# ==========================================
# HALAMAN 3: SENTIMENT PREDICTION (SAVE TO DB)
# ==========================================
elif menu == "Sentiment Prediction":
    st.markdown("<h2 style='color:#1f2937;'>Sentiment Prediction</h2>", unsafe_allow_html=True)
    st.write("Analisis teks tunggal & simpan riwayat")
    
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan:", placeholder="Ketik di sini...", height=150)
        
        if st.button("Analisis Sentimen Sekarang"):
            if input_text:
                res, emo, conf = get_prediction(input_text)
                waktu = time.strftime("%Y-%m-%d %H:%M:%S")
                
                if db:
                    try:
                        # Menyimpan hasil ke riwayat_analisis di Firebase
                        db.collection("riwayat_analisis").document().set({
                            "Teks": input_text, "Hasil": res, "Keyakinan": f"{conf}%", "Waktu": waktu
                        })
                        # Mengupdate statistik_global menggunakan Increment
                        db.collection("statistik_global").document("data_terkini").set({
                            "total": firestore.Increment(1),
                            res.lower(): firestore.Increment(1)
                        }, merge=True)
                    except Exception as e:
                        st.error(f"Gagal simpan ke Firebase: {e}")
                
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")
