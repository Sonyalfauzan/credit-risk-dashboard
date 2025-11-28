import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# 1. KONFIGURASI PATH ARTEFAK (SESUAI STRUKTUR REPO)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Semua artefak (metadata.json, features.pkl, scaler.pkl, credit_risk_model.pkl)
# ada di root repo, sejajar dengan app.py
ARTIFACT_DIR = BASE_DIR

METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "credit_risk_model.pkl")

# File sample (kecil) untuk demo batch
SAMPLE_PATH = os.path.join(ARTIFACT_DIR, "sample_applications.csv")


# =====================================================
# 2. LOAD ARTEFAK MODEL & SAMPLE DATA
# =====================================================

@st.cache_resource
def load_artifacts():
    """Load metadata, daftar fitur, scaler, dan model dari artefak yang sudah disimpan."""
    missing = []
    if not os.path.exists(METADATA_PATH):
        missing.append("metadata.json")
    if not os.path.exists(FEATURES_PATH):
        missing.append("features.pkl")
    if not os.path.exists(SCALER_PATH):
        missing.append("scaler.pkl")
    if not os.path.exists(MODEL_PATH):
        missing.append("credit_risk_model.pkl")

    if missing:
        raise FileNotFoundError(
            "Artefak berikut tidak ditemukan di direktori aplikasi: "
            + ", ".join(missing)
        )

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    return metadata, features, scaler, model


@st.cache_data
def load_sample_data():
    """Load sample CSV kecil untuk demo. Kalau tidak ada / kosong, return None."""
    if not os.path.exists(SAMPLE_PATH):
        return None
    df = pd.read_csv(SAMPLE_PATH)
    if df.shape[0] == 0:
        # file hanya header / kosong
        return None
    return df


# Coba load artefak. Kalau gagal, tampilkan pesan error dan stop.
try:
    metadata, FEATURES, scaler, model = load_artifacts()
except FileNotFoundError as e:
    st.set_page_config(
        page_title="Credit Risk Prediction – ERROR",
        page_icon="💳",
        layout="wide",
    )
    st.title("💳 Credit Risk Prediction – Configuration Error")
    st.error(
        "Aplikasi tidak dapat menemukan file artefak yang dibutuhkan.\n\n"
        f"Detail: {e}\n\n"
        "Pastikan file berikut ada di repo (sejajar dengan app.py):\n"
        "- metadata.json\n- features.pkl\n- scaler.pkl\n- credit_risk_model.pkl"
    )
    st.stop()


# =====================================================
# 3. FUNGSI PREPROCESSING & PREDIKSI
# =====================================================

def prepare_features_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - Memastikan hanya memakai fitur yang diharapkan model (urutannya = FEATURES).
    - Men-scale fitur numerik dengan scaler yang sama seperti saat training.
    """
    df = df_raw.copy()

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Beberapa kolom yang dibutuhkan model tidak ada di data:\n"
            + ", ".join(missing_cols)
        )

    df = df[FEATURES].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_scaled = df.copy()
    if numeric_cols:
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    return df_scaled


def predict_batch(df_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    df_raw -> DataFrame
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


def predict_single(input_dict: dict, threshold: float) -> dict:
    """Helper untuk single applicant dari form Streamlit."""
    df = pd.DataFrame([input_dict])
    res_df = predict_batch(df, threshold)
    row = res_df.iloc[0]

    return {
        "prob_default": float(row["prob_default"]),
        "label": int(row["label"]),
        "risk_flag": row["risk_flag"],
        "raw_input": input_dict,
    }


# =====================================================
# 4. KONFIG HALAMAN & HEADER
# =====================================================

st.set_page_config(
    page_title="Credit Risk Prediction – Lending Company",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk Prediction – Lending Company")

st.markdown(
    """
Dashboard ini menampilkan **demo model Credit Risk Prediction** yang dikembangkan
dari data aplikasi pinjaman (mis. LendingClub 2007–2014).

Model sudah dilatih offline dengan dataset besar (≈229 MB), kemudian disimpan sebagai
artefak yang ringan sehingga pengguna **tidak perlu meng-upload dataset training**.
"""
)

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

# Sidebar
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider(
    "Decision Threshold (probabilitas default)",
    min_value=0.1,
    max_value=0.9,
    value=float(optimal_threshold),
    step=0.05,
)
st.sidebar.caption(
    "Semakin **rendah** threshold → lebih banyak aplikasi yang ditandai sebagai **High Risk**."
)
st.sidebar.markdown("---")
st.sidebar.write("Jumlah fitur yang digunakan model:", len(FEATURES))


# =====================================================
# 5. TABS: SINGLE, DEMO SAMPLE, BATCH OPTIONAL
# =====================================================

tab_single, tab_demo, tab_batch = st.tabs(
    ["🧍 Single Prediction (Form)",
     "📊 Demo with Sample Data",
     "📂 Batch Prediction (Upload CSV)"]
)

# ------------------------- TAB 1: SINGLE -------------------------
with tab_single:
    st.subheader("🧍 Single Applicant – Form Input")

    st.markdown(
        """
Form ini hanya contoh template.  
**Kamu perlu menyesuaikan nama field dan mapping ke fitur** supaya match dengan
`features.pkl` yang dipakai modelmu.
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Loan Amount (USD)", 500.0, 40000.0, 10000.0, 500.0)
        term = st.selectbox("Term", ["36 months", "60 months"])
        int_rate = st.number_input("Interest Rate (%)", 5.0, 35.0, 13.5, 0.1)

    with col2:
        annual_inc = st.number_input("Annual Income (USD)", 10000.0, 300000.0, 60000.0, 1000.0)
        dti = st.number_input("Debt-to-Income Ratio (DTI)", 0.0, 40.0, 18.0, 0.1)
        credit_history_years = st.number_input("Credit History (years)", 0.0, 40.0, 7.0, 0.5)

    with col3:
        grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"])
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "small_business",
                "major_purchase",
                "car",
                "other",
            ],
        )

    # ⚠️ Penting: mapping ini HARUS kamu samakan dengan fitur real di dataset final
    input_features = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "credit_history_years": credit_history_years,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
        # tambahkan fitur lain jika modelmu butuh
    }

    if st.button("🔍 Predict Risk (Single Applicant)"):
        try:
            out = predict_single(input_features, threshold)
            color = "red" if out["label"] == 1 else "green"
            decision = "REJECT / High Risk" if out["label"] == 1 else "APPROVE / Low Risk"

            st.markdown(
                f"""
                <div style="padding:1rem;border-radius:0.5rem;border:1px solid #ddd;">
                    <h4>Decision</h4>
                    <p style="color:{color};font-weight:bold;font-size:1.2rem;">{decision}</p>
                    <p>Risk Flag: <b>{out['risk_flag']}</b></p>
                    <p>Probability of Default: <b>{out['prob_default']:.3f}</b></p>
                    <p>Threshold: <b>{threshold:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Show Input Details"):
                st.json(out["raw_input"])

        except Exception as e:
            st.error(
                "Terjadi error saat melakukan prediksi.\n\n"
                "Kemungkinan besar karena nama fitur di `input_features` "
                "belum match dengan `features.pkl`.\n\n"
                f"Detail error:\n{e}"
            )


# ------------------------- TAB 2: DEMO SAMPLE -------------------------
with tab_demo:
    st.subheader("📊 Demo with Sample Data")

    st.markdown(
        """
Tab ini memakai **sample data kecil** yang sudah disimpan di repo (`sample_applications.csv`).  
Ini berguna untuk demo ke recruiter/teman tanpa perlu dataset 229 MB.
"""
    )

    sample_df = load_sample_data()

    if sample_df is None:
        st.warning(
            "Sample data belum tersedia atau kosong.\n\n"
            "- Buat file kecil (mis. 100 baris) dari dataset finalmu\n"
            "- Simpan sebagai `sample_applications.csv` di root repo.\n"
            "- Pastikan kolomnya sama seperti yang dipakai model."
        )
    else:
        st.write(
            f"Sample data shape: {sample_df.shape[0]} rows x {sample_df.shape[1]} columns."
        )
        st.dataframe(sample_df.head())

        max_rows = sample_df.shape[0]

        # Slider aman untuk berbagai ukuran
        if max_rows <= 10:
            n_rows = st.slider(
                "Berapa banyak baris sample yang mau diprediksi?",
                min_value=1,
                max_value=max_rows,
                value=max_rows,
                step=1,
            )
        else:
            n_rows = st.slider(
                "Berapa banyak baris sample yang mau diprediksi?",
                min_value=10,
                max_value=min(200, max_rows),
                value=min(50, max_rows),
                step=10,
            )

        if st.button("🚀 Run Prediction on Sample Data"):
            try:
                df_sample = sample_df.head(n_rows)
                df_res = predict_batch(df_sample, threshold)

                st.success("Prediksi untuk sample data selesai!")
                st.dataframe(df_res.head(), use_container_width=True)

                st.subheader("📌 Ringkasan Prediksi (Sample)")
                summary = df_res["risk_flag"].value_counts().rename("count")
                st.table(summary.to_frame())

            except Exception as e:
                st.error(f"Error saat memproses sample data:\n\n{e}")


# ------------------------- TAB 3: BATCH (UPLOAD OPSIONAL) -------------------------
with tab_batch:
    st.subheader("📂 Batch Prediction – Upload CSV (Opsional)")

    st.markdown(
        """
Tab ini **opsional** untuk user yang punya data sendiri.  
Tidak disarankan untuk upload dataset training besar (ratusan MB).  
Gunakan saja subset kecil atau data baru yang ingin diuji.
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
                st.dataframe(df_result.head(), use_container_width=True)

                st.subheader("📌 Ringkasan Prediksi")
                summary = df_result["risk_flag"].value_counts().rename("count")
                st.table(summary.to_frame())

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
        st.info(
            "Belum ada file yang di-upload. Untuk demo biasa, cukup gunakan tab "
            "**Single Prediction** atau **Demo with Sample Data**."
        )
