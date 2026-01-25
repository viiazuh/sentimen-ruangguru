import streamlit as st
import pandas as pd
import re
import time
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .stButton>button { background: linear-gradient(135deg, #fb923c, #f97316); color: white; border: none; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC & PREPROCESSING ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_prediction(text):
    # Mockup Model (Siap diganti tim ML dengan FastText/Naive Bayes)
    pos_words = ["bagus", "senang", "mantap", "puas", "keren", "oke"]
    neg_words = ["buruk", "jelek", "kecewa", "parah", "marah", "lambat"]
    
    score = 0
    cleaned = clean_text(text)
    for w in pos_words: 
        if w in cleaned: score += 1
    for w in neg_words: 
        if w in cleaned: score -= 1
        
    if score > 0: return "Positif", "😀", random.randint(75, 98)
    if score < 0: return "Negatif", "😞", random.randint(70, 95)
    return "Netral", "😐", random.randint(50, 70)

# --- SESSION STATE ---
# Kita tetap butuh session state untuk menyimpan history selama tab browser dibuka
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

# --- SIDEBAR (TANPA LOGIN INFO & LOGOUT) ---
with st.sidebar:
    st.title("Sentiment Pro 🙂")
    st.write("Navigasi Aplikasi")
    menu = st.radio("Menu:", ["Dashboard", "Data Management", "Sentiment Prediction"])
    st.divider()
    st.caption("Project Analisis Sentimen Ruangguru") #

# --- MAIN CONTENT ---
if menu == "Dashboard":
    st.header("Overview Dashboard")
    s = st.session_state.stats
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", s["total"])
    col2.metric("Positif", s["positif"])
    col3.metric("Negatif", s["negatif"])
    col4.metric("Netral", s["netral"])
    
    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(10))
    else:
        st.info("Belum ada teks yang dianalisis.")

elif menu == "Data Management":
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("Preview Data:")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Jalankan Batch Analysis"):
            with st.spinner("Sedang memproses..."):
                df['Clean Text'] = df.iloc[:, 0].apply(clean_text)
                df['Hasil Prediksi'] = df['Clean Text'].apply(lambda x: get_prediction(x)[0])
                time.sleep(1)
            st.success("Analisis Batch Selesai!")
            st.dataframe(df, use_container_width=True)
            st.download_button("Simpan Hasil (.csv)", df.to_csv(index=False), "hasil_sentimen.csv")

elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    input_text = st.text_area("Masukkan teks ulasan:", placeholder="Tulis sesuatu di sini...")
    
    if st.button("Analisis Sentimen"):
        if input_text:
            res, emo, conf = get_prediction(input_text)
            
            # Update Stats Global
            st.session_state.stats["total"] += 1
            st.session_state.stats[res.lower()] += 1
            st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
            
            # Tampilkan Hasil
            st.divider()
            c1, c2 = st.columns(2)
            c1.markdown(f"### Hasil: {res} {emo}")
            c2.progress(conf/100, text=f"Confidence: {conf}%")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")