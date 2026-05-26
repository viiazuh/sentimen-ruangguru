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
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.express as px

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

# --- CUSTOM CSS (PRESISI FIGMA & INTER FONT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp { 
        font-family: 'Inter', sans-serif !important; 
    }
    
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }

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
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .sidebar-subtitle {
        font-size: 16px;
        color: #1e293b;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* RADIO MENU STYLING */
    div.row-widget.stRadio > div {
        gap: 15px; 
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }

    [data-testid="stSidebar"] label {
        font-size: 18px !important;
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
    .metric-title { color: #6b7280; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937; font-size: 1.75rem; font-weight: 700; }

    /* BUTTONS */
    .stButton>button { 
        background: #f97316 !important; 
        color: white !important; 
        border-radius: 8px !important; 
        font-weight: 600 !important; 
        border: none !important;
        width: 100%;
    }
    
    /* Tombol Download Khusus agar lebih kecil/rapi */
    [data-testid="stDownloadButton"] > button {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FIREBASE HELPERS ---
def save_to_firebase(text, result, confidence):
    try:
        db.collection("history_sentiment").add({
            "teks": text, "hasil": result, "Probabilitas": confidence, "waktu": firestore.SERVER_TIMESTAMP
        })
    except: pass

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
    except: return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_history_firebase(limit=10):
    try:
        docs = db.collection("history_sentiment").order_by("waktu", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [{"Teks": d.to_dict().get("teks"), "Hasil": d.to_dict().get("hasil"), "Waktu": d.to_dict()['waktu'].strftime("%H:%M:%S") if d.to_dict().get('waktu') else "N/A"} for d in docs]
    except: return []

# --- 🔥 PERBAIKAN REVISI: FIREBASE BATCH DENGAN SUB-COLLECTION ---
def save_batch_to_firebase(filename, df_results):
    try:
        # 1. Simpan Metadata Sesi di dokumen utama batch_sessions
        doc_ref = db.collection("batch_sessions").document()
        doc_ref.set({
            "nama_file": filename,
            "total_baris": len(df_results),
            "waktu_eksekusi": firestore.SERVER_TIMESTAMP
        })
        
        # 2. Simpan setiap baris hasil prediksi ke dalam Sub-collection 'detail_ulasan'
        records = df_results.to_dict(orient="records")
        
        # Menggunakan WriteBatch Firestore (Maksimal 500 operasi per commit)
        batch = db.batch()
        count = 0
        
        for record in records:
            sub_doc_ref = doc
