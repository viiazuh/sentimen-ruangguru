import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf  
import numpy as np       
import pickle            
import io

# ---  SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")


st.markdown("""
    <style>
    /* 1. IMPORT FONT INTER AGAR SAMA DENGAN FIGMA */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* 2. COLORS & BACKGROUND */
    .stApp { 
        background-color: #f7f9fc !important; 
        color: #1f2937 !important; 
    }

    /* 3. SIDEBAR STYLING */
    [data-testid="stSidebar"] { 
        background-color: white !important; 
        border-right: 1px solid #e5e7eb !important; 
        padding-top: 2rem;
    }
    
    /* Menyesuaikan teks sidebar agar tidak terlalu gelap/pudar */
    [data-testid="stSidebar"] .stMarkdown p { 
        color: #4b5563 !important; 
        font-size: 0.95rem; 
        font-weight: 500; 
        margin-bottom: -10px;
    }

    /* 4. MARGIN & CARD STYLING (Dashboard) */
    .metric-card {
        background-color: white; 
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        border: 1px solid #f3f4f6;
        margin-bottom: 1rem;
    }
    
    .metric-title { 
        color: #6b7280 !important; 
        font-size: 0.85rem; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .metric-value { 
        color: #1f2937 !important; 
        font-size: 1.75rem; 
        font-weight: 700; 
    }

    /* 5. CUSTOM TABLE (Dataframe) */
    /* Membuat tabel Streamlit terlihat lebih clean mirip Figma */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    /* 6. BUTTONS */
    .stButton>button {
        background: #f97316 !important; /* Warna Orange Figma */
        color: white !important; 
        border: none !important; 
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important; 
        font-weight: 600 !important;
        width: 100%; /* Biar responsif di mobile */
    }
    
    /* Responsivitas untuk layar kecil */
    @media (max-width: 768px) {
        .metric-card { padding: 15px; }
        .metric-value { font-size: 1.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---  MODEL LOADING ---
@st.cache_resource
def load_sentiment_model():
    model = tf.keras.models.load_model('models/model_hybrid_coc.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('models/normalization_dicts.pkl', 'rb') as f:
        norm_dict = pickle.load(f)
    return model, tokenizer, norm_dict

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
        scores = {
            "positif": float(prediction[0][2] * 100),
            "negatif": float(prediction[0][1] * 100),
            "netral":  float(prediction[0][0] * 100),
        }
        return labels[idx], emojis[idx], int(conf), scores
    return "Error", "⚠️", 0, {}

def build_excel(df_result):
    """Coba buat Excel, fallback ke CSV kalau semua engine gagal"""
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

# ---  SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    st.markdown("<p>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    st.write("")
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")
    # st.markdown("<div style='margin-top: 200px;'></div>", unsafe_allow_html=True)
    # st.divider()


# DASHBOARD PAGE

if menu == "Dashboard":
    st.markdown("<h2 style='color:#1f2937;'>Dashboard</h2>", unsafe_allow_html=True)
    st.write("Overview Statistik Real-time")
    
    s = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif</div><div class="metric-value">{s["positif"]}</div></div><div class="icon-box bg-green">😊</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif</div><div class="metric-value">{s["negatif"]}</div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral</div><div class="metric-value">{s["netral"]}</div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Belum ada aktivitas analisis.")

# DATA MANAGEMENT PAGE
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
            st.write("Preview 10 Data Teratas:")
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
                    text_col = None
                    for candidate in ['textDisplay', 'text', 'ulasan', 'review', 'komentar', 'content']:
                        if candidate in df.columns:
                            text_col = candidate
                            break
                    if text_col is None:
                        text_col = df.columns[0]
                    texts_asli = df[text_col].astype(str).tolist()

                    progress_bar = st.progress(0, text="Step 1/3: Preprocessing teks...")
                    texts_clean      = [clean_text(t)           for t in texts_asli]
                    texts_normalized = [normalize_text(t)       for t in texts_asli]
                    texts_stopword   = [remove_stopwords(n)     for n in texts_normalized]
                    texts_stemmed    = [stem_text(s)            for s in texts_stopword]
                    progress_bar.progress(0.33, text="Step 2/3: Tokenisasi & padding...")

                    seqs   = tokenizer_ml.texts_to_sequences(texts_normalized)
                    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                    progress_bar.progress(0.66, text="Step 3/3: Prediksi model (batch)...")

                    all_preds = model_ml.predict(padded, batch_size=512, verbose=0)
                    progress_bar.progress(1.0, text="Menyusun hasil...")

                    labels = ["Netral", "Negatif", "Positif"]
                    idxs   = np.argmax(all_preds, axis=1)
                    confs  = np.max(all_preds, axis=1) * 100

                    results = []
                    for i in range(total):
                        idx  = idxs[i]
                        pred = all_preds[i]
                        results.append({
                            "Text Asli":       texts_asli[i],
                            "Text Cleaned":    texts_clean[i],
                            "Text Normalized": texts_normalized[i],
                            "Text Stopword":   texts_stopword[i],
                            "Text Stemmed":    texts_stemmed[i],
                            "Sentimen":        labels[idx],
                            "Positif (%)":     round(float(pred[2]) * 100, 2),
                            "Negatif (%)":     round(float(pred[1]) * 100, 2),
                            "Netral (%)":      round(float(pred[0]) * 100, 2),
                            "Keyakinan (%)":   int(confs[i]),
                        })

                    st.session_state.dataset = pd.DataFrame(results)
                    st.success(f"✅ Batch Analysis Selesai! {total} data diproses.")

    if st.session_state.dataset is not None:
        with st.container(border=True):
            st.markdown("### Hasil Analisis")

            df_result = st.session_state.dataset
            cnt = df_result["Sentimen"].value_counts()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total",      len(df_result))
            r2.metric("😀 Positif", cnt.get("Positif", 0))
            r3.metric("😞 Negatif", cnt.get("Negatif", 0))
            r4.metric("😐 Netral",  cnt.get("Netral",  0))

            st.dataframe(df_result, use_container_width=True)
            csv_data = df_result.to_csv(index=False).encode('utf-8')
            excel_data, excel_ok = build_excel(df_result)

            c1, c2, c3 = st.columns([1, 1, 5])
            with c1:
                st.download_button("⬇️ CSV", csv_data, "hasil_sentimen.csv", "text/csv")
            with c2:
                if excel_ok:
                    st.download_button("⬇️ Excel", excel_data, "hasil_sentimen.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("Excel tidak tersedia.")
            with c3:
                col_spacer, col_hapus = st.columns([4, 1])
                with col_hapus:
                    if st.button("🗑️ Hapus Hasil"):
                        st.session_state.dataset = None
                        st.rerun()

# SENTIMENT PREDICTION PAGE
elif menu == "Sentiment Prediction":
    st.markdown("<h2 style='color:#1f2937;'>Sentiment Prediction</h2>", unsafe_allow_html=True)
    st.write("Analisis teks tunggal secara real-time")
    
    with st.container(border=True):
        st.subheader("Sentiment Analysis")
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: keren banget acaranya...", height=150)
        
        if st.button("Analisis Sentimen Sekarang"):
            if input_text:
                res, emo, conf, scores = get_prediction(input_text)
                
                st.session_state.stats["total"] += 1
                st.session_state.stats[res.lower()] += 1
                st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
                
                st.divider()
                col_res, col_conf = st.columns(2)
                col_res.markdown(f"### Hasil: {res} {emo}")
                col_conf.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
            else:
                st.warning("Silakan ketikkan ulasan terlebih dahulu.")
