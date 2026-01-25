import streamlit as st
import pandas as pd
import time
import re

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. INJECT CUSTOM CSS (Kunci Kemiripan UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #f7f9fc;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #f3f4f6;
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .metric-title { color: #6b7280; font-size: 0.875rem; margin-bottom: 0.25rem; }
    .metric-value { color: #1f2937; font-size: 1.875rem; font-weight: 700; }
    .metric-sub { color: #9ca3af; font-size: 0.75rem; margin-top: 0.25rem; }
    
    /* Icon Container */
    .icon-box {
        width: 2.5rem; height: 2.5rem;
        border-radius: 0.5rem;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem;
    }
    .bg-blue { background-color: #dbeafe; color: #3b82f6; }
    .bg-green { background-color: #d1fae5; color: #10b981; }
    .bg-red { background-color: #fee2e2; color: #ef4444; }
    .bg-gray { background-color: #f3f4f6; color: #6b7280; }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #fb923c, #f97316);
        color: white; border: none; border-radius: 0.75rem;
        padding: 0.6rem 1.5rem; font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3); }

    /* Table & Container */
    .main-container {
        background-color: white; padding: 2rem; border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #f3f4f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "pos": 0, "neg": 0, "neu": 0}

# --- 4. HELPER COMPONENTS ---
def draw_metric_card(title, value, sub, icon, color_class):
    st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            <div class="icon-box {color_class}">{icon}</div>
        </div>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='color:#1f2937;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    menu = st.radio("Navigasi", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("Project Analisis Sentimen Ruangguru")

# --- 6. PAGE VIEWS ---

if menu == "Dashboard":
    st.header("Dashboard")
    st.write("Overview Statistik")
    
    # Responsif Card Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: draw_metric_card("Total Data", st.session_state.stats['total'], "0 data diproses", "📊", "bg-blue")
    with c2: draw_metric_card("Positif", st.session_state.stats['pos'], "0% dari total", "😊", "bg-green")
    with c3: draw_metric_card("Negatif", st.session_state.stats['neg'], "0% dari total", "😞", "bg-red")
    with c4: draw_metric_card("Netral", st.session_state.stats['neu'], "0% dari total", "😐", "bg-gray")

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.subheader("Aktivitas Terbaru")
    st.table(pd.DataFrame({"Aktivitas": ["Menyiapkan sistem analisis"], "Waktu": ["5 menit lalu"]}))
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Data Management":
    st.header("Data Management")
    st.write("Kelola dataset teks Anda dan label analisis sentimen")
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col_up, col_del = st.columns([4, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload", type=["csv", "xlsx"], label_visibility="collapsed")
    with col_del:
        st.button("🗑️ Delete All")
    
    # Search Bar
    st.text_input("🔍 Cari teks asli...", placeholder="Ketik untuk mencari...")
    
    # Table Header Dummy (Biar mirip screenshot kamu)
    cols = ["NO", "REAL TEXT", "CLEAN TEXT", "PREDIKSI"]
    st.dataframe(pd.DataFrame(columns=cols), use_container_width=True)
    st.write("Showing 0 to 0 of 0 entries")
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    st.write("Analisis sentimen teks")
    
    st.markdown('<div style="background-color: #fff7ed; padding: 2rem; border-radius: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.subheader("Sentiment Analysis")
    text_input = st.text_area("Masukkan teks", placeholder="Ketik teks yang ingin dianalisis...", height=200)
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔍 Analisis Sentimen"):
            with st.spinner("Menganalisis..."):
                time.sleep(1)
                st.success("Selesai!")
    st.markdown('</div></div>', unsafe_allow_html=True)