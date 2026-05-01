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

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- INISIALISASI FIREBASE (DENGAN FIX SECRETS) ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            # Rencana A: Ambil dari Brankas Digital (Streamlit Secrets)
            firebase_details = dict(st.secrets["firebase"])
            if "private_key" in firebase_details:
                firebase_details["private_key"] = firebase_details["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(firebase_details)
            firebase_admin.initialize_app(cred)
        else:
            # Rencana B: Ambil dari file lokal (Testing di Laptop)
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

try:
    db = firestore.client()
except Exception:
    db = None

# --- CSS FIGMA STYLE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; padding-top: 2rem; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; margin-bottom: 1rem; }
    .metric-title { color: #6b7280 !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937 !important; font-size: 1.75rem; font-weight: 700; }
    .stButton>button { background: #f97316 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; }
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
    except Exception:
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
    return set(['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan', 'adalah', 'pada', 'juga', 'dalam', 'ada', 'tidak'])

def remove_stopwords(text):
    sw = get_stopwords()
    words = str(text).split()
    return " ".join([w for w in words if w not in sw]).strip()

def simple_stem(word):
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
    try:
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Hasil Sentimen')
        return excel_buffer.getvalue(), True
    except Exception:
        return None, False

# --- DATABASE FUNCTIONS ---
def get_firebase_stats():
    if db:
        try:
            stats_ref = db.collection("statistik_global").document("data_terkini").get()
            if stats_ref.exists:
                return stats_ref.to_dict()
        except Exception:
            pass
    return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_firebase_history():
    if db:
        try:
            docs = db.collection("riwayat_analisis").order_by("Waktu", direction=firestore.Query.DESCENDING).limit(100).stream()
            return [doc.to_dict() for doc in docs]
        except Exception:
            return []
    return []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## Sentiment<span>🙂</span>", unsafe_allow_html=True)
    st.write("Project Analisis Sentimen Ruangguru")
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"])

# ==========================================
# DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.markdown("## Dashboard")
    s = get_firebase_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s.get("total", 0)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s.get("positif", 0)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s.get("negatif", 0)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s.get("netral", 0)}</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    hist = get_firebase_history()
    if hist:
        st.dataframe(pd.DataFrame(hist), use_container_width=True)
    else:
        st.info("Belum ada data di database.")

# ==========================================
# DATA MANAGEMENT (BATCH LOKAL)
# ==========================================
elif menu == "Data Management":
    st.markdown("## Data Management")
    uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)
        if st.button("🔍 Jalankan Batch Analysis"):
            with st.spinner("Menganalisis..."):
                text_col = df.columns[0]
                texts_asli = df[text_col].astype(str).tolist()
                
                # Proses Batch (Lokal)
                seqs = tokenizer_ml.texts_to_sequences([normalize_text(t) for t in texts_asli])
                padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                all_preds = model_ml.predict(padded, verbose=0)
                
                labels = ["Netral", "Negatif", "Positif"]
                results = []
                for i in range(len(texts_asli)):
                    idx = np.argmax(all_preds[i])
                    results.append({"Text": texts_asli[i], "Sentimen": labels[idx], "Keyakinan": f"{int(np.max(all_preds[i])*100)}%"})
                
                st.session_state.dataset = pd.DataFrame(results)
                st.success("Batch Analysis Selesai!")

    if st.session_state.get('dataset') is not None:
        st.dataframe(st.session_state.dataset, use_container_width=True)

# ==========================================
# SENTIMENT PREDICTION (SIMPAN FIREBASE)
# ==========================================
elif menu == "Sentiment Prediction":
    st.markdown("## Sentiment Prediction")
    input_text = st.text_area("Masukkan teks ulasan:", height=150)
    if st.button("Analisis Sentimen Sekarang"):
        if input_text:
            res, emo, conf = get_prediction(input_text)
            waktu = time.strftime("%Y-%m-%d %H:%M:%S")
            if db:
                try:
                    db.collection("riwayat_analisis").document().set({
                        "Teks": input_text, "Hasil": res, "Keyakinan": f"{conf}%", "Waktu": waktu
                    })
                    db.collection("statistik_global").document("data_terkini").set({
                        "total": firestore.Increment(1),
                        res.lower(): firestore.Increment(1)
                    }, merge=True)
                except:
                    pass
            st.divider()
            st.markdown(f"### Hasil: {res} {emo}")
            st.progress(conf/100, text=f"Keyakinan: {conf}%")
