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


def parse_emp_length_to_years(val):
    """Parse kolom emp_length (string) menjadi numeric years."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s in ["", "n/a"]:
        return 0.0
    if s == "10+ years":
        return 10.0
    if s == "< 1 year":
        return 0.5
    # contoh: "3 years", "1 year"
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return float(digits) if digits != "" else 0.0
    except ValueError:
        return 0.0


def build_features_from_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun DataFrame fitur lengkap (semua kolom di FEATURES)
    dari data mentah LendingClub (seperti loan_amnt, term, grade, dst.).
    """
    # Mulai dengan DataFrame semua 0.0
    df_feat = pd.DataFrame(0.0, index=df_raw.index, columns=FEATURES, dtype=float)

    # Kolom yang bisa di-copy langsung dari raw
    cols_direct = [
        "Unnamed: 0",
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mths_since_last_major_derog",
        "tot_coll_amt",
        "tot_cur_bal",
        "total_rev_hi_lim",
        "sub_grade",
        "purpose",
        "addr_state",
    ]
    for col in cols_direct:
        if col in df_raw.columns and col in df_feat.columns:
            df_feat[col] = df_raw[col]

    # Rasio-rasio engineered
    if {"loan_amnt", "annual_inc"} <= set(df_raw.columns) and "loan_to_income_ratio" in df_feat.columns:
        inc = df_raw["annual_inc"].replace(0, np.nan)
        ratio = (df_raw["loan_amnt"] / inc).replace([np.inf, -np.inf], np.nan)
        df_feat["loan_to_income_ratio"] = ratio.fillna(0).astype(float)

    if {"installment", "annual_inc"} <= set(df_raw.columns) and "installment_to_income_ratio" in df_feat.columns:
        inc = df_raw["annual_inc"].replace(0, np.nan)
        ratio2 = (df_raw["installment"] / inc).replace([np.inf, -np.inf], np.nan)
        df_feat["installment_to_income_ratio"] = ratio2.fillna(0).astype(float)

    if {"revol_bal", "annual_inc"} <= set(df_raw.columns) and "revol_bal_to_income_ratio" in df_feat.columns:
        inc = df_raw["annual_inc"].replace(0, np.nan)
        ratio3 = (df_raw["revol_bal"] / inc).replace([np.inf, -np.inf], np.nan)
        df_feat["revol_bal_to_income_ratio"] = ratio3.fillna(0).astype(float)

    # emp_length_years
    if "emp_length" in df_raw.columns and "emp_length_years" in df_feat.columns:
        df_feat["emp_length_years"] = df_raw["emp_length"].map(parse_emp_length_to_years).astype(float)

    # term_months dari 'term'
    if "term" in df_raw.columns and "term_months" in df_feat.columns:
        term_numeric = df_raw["term"].astype(str).str.extract(r"(\\d+)")[0].astype(float)
        df_feat["term_months"] = term_numeric.fillna(36.0)

    # One-hot grade_B..G
    if "grade" in df_raw.columns:
        for g in ["B", "C", "D", "E", "F", "G"]:
            col = f"grade_{g}"
            if col in df_feat.columns:
                df_feat[col] = (df_raw["grade"] == g).astype(float)

    # One-hot home_ownership
    if "home_ownership" in df_raw.columns:
        for h in ["NONE", "OTHER", "OWN", "RENT", "MORTGAGE"]:
            col = f"home_ownership_{h}"
            if col in df_feat.columns:
                df_feat[col] = (df_raw["home_ownership"] == h).astype(float)

    # verification_status one-hot
    if "verification_status" in df_raw.columns:
        vs = df_raw["verification_status"].astype(str)
        if "verification_status_Source Verified" in df_feat.columns:
            df_feat["verification_status_Source Verified"] = (vs == "Source Verified").astype(float)
        if "verification_status_Verified" in df_feat.columns:
            df_feat["verification_status_Verified"] = (vs == "Verified").astype(float)

    # initial_list_status_w
    if "initial_list_status" in df_raw.columns and "initial_list_status_w" in df_feat.columns:
        ils = df_raw["initial_list_status"].astype(str)
        df_feat["initial_list_status_w"] = (ils == "w").astype(float)

    return df_feat


def predict_batch(df_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Batch prediction untuk berbagai format input:
    - Jika df_raw sudah berisi semua kolom FEATURES → gunakan langsung.
    - Jika belum (data mentah LendingClub) → bangun fitur terlebih dahulu.
    """
    has_all_features = all(col in df_raw.columns for col in FEATURES)

    if has_all_features:
        df_features = df_raw
    else:
        df_features = build_features_from_raw_df(df_raw)

    X = prepare_features_for_model(df_features)
    prob_default = model.predict_proba(X)[:, 1]  # asumsi kelas 1 = Bad Loan

    labels = (prob_default >= threshold).astype(int)

    # Hasil dikembalikan dengan kolom asli + kolom prediksi
    result = df_raw.copy()
    result["prob_default"] = prob_default
    result["label"] = labels
    result["risk_flag"] = np.where(
        labels == 1, "High Risk (Bad Loan)", "Low Risk (Good Loan)"
    )

    return result



def build_feature_row_from_form(form_inputs: dict) -> pd.DataFrame:
    """
    Bangun satu baris fitur lengkap (semua kolom di FEATURES)
    dari input form yang lebih sederhana.
    Kolom-kolom yang tidak diisi user di-set ke 0 sebagai default.
    """
    # Inisialisasi semua fitur dengan 0.0 (numerik)
    row = {feat: 0.0 for feat in FEATURES}

    # Ambil nilai dari form
    loan_amnt = float(form_inputs.get("loan_amnt", 0.0))
    term_str = form_inputs.get("term", "36 months")
    int_rate = float(form_inputs.get("int_rate", 0.0))
    annual_inc = float(form_inputs.get("annual_inc", 0.0))
    dti = float(form_inputs.get("dti", 0.0))
    credit_history_years = float(form_inputs.get("credit_history_years", 0.0))
    grade = form_inputs.get("grade", "A")
    home_ownership = form_inputs.get("home_ownership", "RENT")
    purpose = form_inputs.get("purpose", "debt_consolidation")

    # ----- Fitur dasar pinjaman -----
    if "loan_amnt" in row:
        row["loan_amnt"] = loan_amnt
    if "funded_amnt" in row:
        row["funded_amnt"] = loan_amnt
    if "funded_amnt_inv" in row:
        row["funded_amnt_inv"] = loan_amnt

    if "int_rate" in row:
        row["int_rate"] = int_rate

    if "annual_inc" in row:
        row["annual_inc"] = annual_inc
    if "dti" in row:
        row["dti"] = dti
    if "credit_history_years" in row:
        row["credit_history_years"] = credit_history_years

    # Term → term_months
    if term_str == "60 months":
        term_months = 60
    else:
        term_months = 36
    if "term_months" in row:
        row["term_months"] = float(term_months)

    # Installment (perkiraan kasar)
    if "installment" in row and term_months > 0:
        row["installment"] = loan_amnt / term_months

    # ----- Engineered ratios -----
    if "loan_to_income_ratio" in row and annual_inc > 0:
        row["loan_to_income_ratio"] = loan_amnt / annual_inc
    if "installment_to_income_ratio" in row and annual_inc > 0 and row.get("installment", 0) > 0:
        row["installment_to_income_ratio"] = row["installment"] / annual_inc
    if "revol_bal_to_income_ratio" in row and annual_inc > 0 and "revol_bal" in row:
        row["revol_bal_to_income_ratio"] = row.get("revol_bal", 0.0) / annual_inc

    # ----- One-hot grade (grade_B–grade_G) -----
    grade_map = ["B", "C", "D", "E", "F", "G"]
    for g in grade_map:
        col = f"grade_{g}"
        if col in row:
            row[col] = 1.0 if grade == g else 0.0
    # grade A = semua grade_B..G = 0

    # ----- One-hot home_ownership -----
    home_cols = ["NONE", "OTHER", "OWN", "RENT", "MORTGAGE"]
    for h in home_cols:
        col = f"home_ownership_{h}"
        if col in row:
            row[col] = 1.0 if home_ownership == h else 0.0

    # ----- Purpose (jika ada kolom 'purpose') -----
    if "purpose" in row:
        # Kalau mau, kamu bisa mapping purpose string → kode numerik sesuai training.
        # Di sini dibiarkan 0 sebagai baseline.
        pass

    # ----- Kolom lain yang spesifik, set default aman -----
    if "Unnamed: 0" in row:
        row["Unnamed: 0"] = 0.0

    return pd.DataFrame([row])


def predict_single(form_inputs: dict, threshold: float) -> dict:
    """Single applicant prediction: bangun baris fitur lengkap dari form, lalu prediksi."""
    X_raw = build_feature_row_from_form(form_inputs)
    X = prepare_features_for_model(X_raw)
    prob_default = float(model.predict_proba(X)[:, 1][0])
    label = int(prob_default >= threshold)
    risk_flag = "High Risk (Bad Loan)" if label == 1 else "Low Risk (Good Loan)"
    return {
        "prob_default": prob_default,
        "label": label,
        "risk_flag": risk_flag,
        "raw_input": form_inputs,
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
**Mapping dari form → fitur model** sudah dibuat di fungsi `build_feature_row_from_form()`.  
Kalau kamu ubah form, jangan lupa update mapping di sana.
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
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"])
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
