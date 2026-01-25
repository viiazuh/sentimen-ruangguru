import streamlit as st
import pandas as pd
import re
import time
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# Custom Styling (Orange Accent ala React kamu)
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .stButton>button { background: linear-gradient(135deg, #fb923c, #f97316); color: white; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 4px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC & PREPROCESSING ---
def clean_text(text):
    # Logika cleaning (Case folding & Regex)
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Placeholder: Kamu bisa masukkan mapping 'myslang_alay.csv' di sini nanti
    return text.strip()

def get_prediction(text):
    # MOCKUP MODEL (Siap diganti tim ML dengan FastText/Naive Bayes)
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
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

# --- SIDEBAR ---
with st.sidebar:
    st.title("Sentiment Pro 🙂")
    st.info("Log in sebagai: **Admin (kadus7)**")
    menu = st.radio("Navigasi", ["Dashboard", "Data Management", "Sentiment Prediction"])
    st.divider()
    if st.button("Logout"):
        st.toast("Berhasil keluar!")

# --- MAIN CONTENT ---
if menu == "Dashboard":
    st.header("Overview Dashboard")
    s = st.session_state.stats
    
    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", s["total"])
    col2.metric("Positif", s["positif"], "Success", delta_color="normal")
    col3.metric("Negatif", s["negatif"], "- Warning", delta_color="inverse")
    col4.metric("Netral", s["netral"])
    
    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(5))
    else:
        st.write("Belum ada data dianalisis.")

elif menu == "Data Management":
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload file dataset (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("Preview Data:")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Proses Batch Analysis"):
            with st.status("Memproses teks..."):
                # Simulasikan preprocessing & prediksi
                df['Clean Text'] = df.iloc[:, 0].apply(clean_text)
                df['Hasil Prediksi'] = df['Clean Text'].apply(lambda x: get_prediction(x)[0])
                time.sleep(1.5)
            st.success("Analisis selesai!")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Hasil", df.to_csv(index=False), "hasil_sentimen.csv")

elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks untuk dianalisis", placeholder="Contoh: Aplikasi ini sangat membantu...")
        col_btn, _ = st.columns([1, 3])
        if col_btn.button("Analisis Sekarang"):
            if input_text:
                res, emo, conf = get_prediction(input_text)
                
                # Update Session
                st.session_state.stats["total"] += 1
                st.session_state.stats[res.lower()] += 1
                st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
                
                # Display Result
                st.divider()
                c1, c2 = st.columns(2)
                c1.markdown(f"### Hasil: {res} {emo}")
                c2.progress(conf/100, text=f"Confidence: {conf}%")
            else:
                st.warning("Mohon masukkan teks terlebih dahulu.")