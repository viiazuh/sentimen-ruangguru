import streamlit as st
import pandas as pd
import time
import re
import random

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. FORCE LIGHT MODE & CUSTOM CSS (Orange Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; }
    
    /* Card Styling */
    .metric-card {
        background-color: white; padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f3f4f6;
        display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;
    }
    .metric-title { color: #6b7280; font-size: 0.9rem; font-weight: 500; }
    .metric-value { color: #1f2937; font-size: 2rem; font-weight: 700; margin-top: 8px; }
    .metric-sub { color: #9ca3af; font-size: 0.8rem; margin-top: 8px; }
    
    /* Icon Boxes */
    .icon-box { width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
    .bg-blue { background-color: #dbeafe; color: #3b82f6; }
    .bg-green { background-color: #d1fae5; color: #10b981; }
    .bg-red { background-color: #fee2e2; color: #ef4444; }
    .bg-gray { background-color: #f3f4f6; color: #6b7280; }

    /* Button Styling Orange */
    .stButton>button {
        background: linear-gradient(135deg, #fb923c, #f97316) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important; font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC & PREPROCESSING ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_prediction(text):
    # Dummy logic untuk tim ML nanti
    pos_words = ["bagus", "puas", "mantap", "keren", "oke"]
    neg_words = ["buruk", "kecewa", "parah", "jelek", "lambat"]
    score = 0
    cleaned = clean_text(text)
    for w in pos_words: 
        if w in cleaned: score += 1
    for w in neg_words: 
        if w in cleaned: score -= 1
    if score > 0: return "Positif", "😀", random.randint(75, 98)
    if score < 0: return "Negatif", "😞", random.randint(70, 95)
    return "Netral", "😐", random.randint(50, 70)

# --- 4. SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#1f2937;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"])
    st.divider()
    st.caption("Project Analisis Sentimen Ruangguru")

# --- 6. PAGE ROUTING ---

# --- DASHBOARD ---
if menu == "Dashboard":
    st.header("Dashboard")
    st.write("Overview Statistik")
    s = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div><div class="metric-sub">{s["total"]} data diproses</div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif</div><div class="metric-value">{s["positif"]}</div><div class="metric-sub">Berdasarkan ulasan</div></div><div class="icon-box bg-green">😊</div></div>', unsafe_allow_html=True)
    with c3: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif</div><div class="metric-value">{s["negatif"]}</div><div class="metric-sub">Berdasarkan ulasan</div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral</div><div class="metric-value">{s["netral"]}</div><div class="metric-sub">Berdasarkan ulasan</div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Belum ada aktivitas analisis.")

# --- DATA MANAGEMENT ---
elif menu == "Data Management":
    st.header("Data Management")
    st.write("Kelola dataset ulasan Anda")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Preview Dataset:")
            st.dataframe(df.head(5), use_container_width=True)
            
            if st.button("Proses Batch Analysis"):
                with st.spinner("Menganalisis data..."):
                    results = []
                    for i, row in df.iterrows():
                        txt = str(row.iloc[0])
                        sentiment, _, _ = get_prediction(txt)
                        results.append({"Real Text": txt, "Clean Text": clean_text(txt), "Prediksi": sentiment})
                    st.session_state.dataset = pd.DataFrame(results)
                    st.success("Analisis selesai!")

    if st.session_state.dataset is not None:
        st.divider()
        st.dataframe(st.session_state.dataset, use_container_width=True)
        st.download_button("Download CSV", st.session_state.dataset.to_csv(index=False), "hasil_sentimen.csv")
        if st.button("Hapus Semua Data"):
            st.session_state.dataset = None
            st.rerun()

# --- SENTIMENT PREDICTION ---
elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    st.write("Coba analisis teks tunggal")
    
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Ketik di sini...", height=150)
        if st.button("Analisis Sentimen"):
            if input_text:
                res, emo, conf = get_prediction(input_text)
                
                # Update Stats Global
                st.session_state.stats["total"] += 1
                st.session_state.stats[res.lower()] += 1
                st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
                
                st.divider()
                col_res, col_conf = st.columns(2)
                col_res.markdown(f"### Hasil: {res} {emo}")
                col_conf.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
            else:
                st.warning("Masukkan teks terlebih dahulu!")