import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Konfigurasi path artefak
# =========================
ARTIFACT_DIR = "model_artifacts"
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "credit_risk_model.pkl")  # isi dengan model finalmu


# =========================
# Load artefak (cached)
# =========================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"metadata.json tidak ditemukan di {ARTIFACT_DIR}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"features.pkl tidak ditemukan di {ARTIFACT_DIR}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"scaler.pkl tidak ditemukan di {ARTIFACT_DIR}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Simpan dulu final model kamu dengan joblib.dump ke path tersebut."
        )

    # metadata.json
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # features.pkl -> list nama kolom yang dipakai model
    features = joblib.load(FEATURES_PATH)

    # scaler.pkl -> StandardScaler / scaler lain yang kamu pakai
    scaler = joblib.load(SCALER_PATH)

    # credit_risk_model.pkl -> XGBoost / LightGBM final
    model = joblib.load(MODEL_PATH)

    return metadata, features, scaler, model


metadata, FEATURES, scaler, model = load_artifacts()


# =========================
# Fungsi preprocessing & prediksi
# =========================
def prepare_features_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - Pastikan hanya pakai fitur yang model harapkan (urutan = FEATURES)
    - Scale numeric features dengan scaler yang sama seperti di training
    """
    df = df_raw.copy()

    # Hanya ambil kolom yang ada di FEATURES (sisa di-drop)
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Beberapa kolom yang dibutuhkan model tidak ada di file CSV:\n"
            + ", ".join(missing_cols)
        )

    df = df[FEATURES].copy()

    # Deteksi kolom numerik berdasarkan dtype
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Scale numeric features
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    return df_scaled


def predict_batch(df_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    df_raw -> DataFrame dari CSV user
    Return -> df dengan kolom tambahan: prob_default, label, risk_flag
    """
    X = prepare_features_for_model(df_raw)
    prob_default = model.predict_proba(X)[:, 1]  # asumsi kelas 1 = Bad Loan

    labels = (prob_default >= threshold).astype(int)

    result = df_raw.copy()
    result["prob_default"] = prob_default
    result["label"] = labels
    result["risk_flag"] = np.where(
        labels == 1, "High Risk (Bad Loan)", "Low Risk (Good Loan)"
    )

    return result


# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Credit Risk Prediction – Lending Company",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk Prediction – Lending Company")

st.markdown(
    """
Aplikasi ini menggunakan **model machine learning** untuk memprediksi risiko kredit
berdasarkan data aplikasi pinjaman.

Model & artefak (metadata, scaler, fitur) diambil dari proses training di notebook.
"""
)

# =========================
# Tampilkan ringkasan performa model dari metadata.json
# =========================
metrics = metadata.get("metrics", {})
optimal_threshold = metadata.get("optimal_threshold", 0.5)
model_type = metadata.get("model_type", "Unknown Model")

st.subheader("📈 Model Summary")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Model", model_type)
with c2:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
with c3:
    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
with c4:
    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
with c5:
    st.metric("Optimal Threshold", f"{optimal_threshold:.2f}")

st.markdown("---")

# =========================
# Sidebar: threshold & info
# =========================
st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold (probabilitas default)",
    min_value=0.1,
    max_value=0.9,
    value=float(optimal_threshold),
    step=0.05,
)
st.sidebar.caption(
    "Semakin **rendah** threshold → model lebih ketat menandai **High Risk** "
    "(lebih banyak aplikasi ditolak)."
)

st.sidebar.markdown("---")
st.sidebar.write("Jumlah fitur yang digunakan model:", len(FEATURES))


# =========================
# Batch Prediction (Upload CSV)
# =========================
st.subheader("📂 Batch Prediction – Upload CSV")

st.markdown(
    """
Upload file **CSV** dengan struktur kolom yang sama seperti dataset final yang kamu pakai
saat training model (sebelum split train/test).  
Aplikasi akan mengembalikan probabilitas default dan label risiko untuk setiap baris.
"""
)

uploaded_file = st.file_uploader("Upload CSV data peminjam", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("Preview data (5 baris teratas):")
        st.dataframe(df_input.head())

        st.info(
            f"File mengandung {df_input.shape[0]} baris dan {df_input.shape[1]} kolom."
        )

        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Menjalankan model untuk seluruh baris..."):
                df_result = predict_batch(df_input, threshold=threshold)

            st.success("Prediksi selesai!")

            st.subheader("📊 Hasil Prediksi (5 baris pertama)")
            st.dataframe(
                df_result.head(),
                use_container_width=True,
            )

            # Ringkas distribusi prediksi
            st.subheader("📌 Ringkasan Prediksi")
            summary = df_result["risk_flag"].value_counts().rename("count")
            st.table(summary.to_frame())

            # Download hasil
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil Prediksi CSV",
                data=csv_bytes,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Terjadi error saat membaca file / melakukan prediksi:\n\n{e}")
else:
    st.info("Silakan upload file CSV untuk memulai batch prediction.")
