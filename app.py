import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import io

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS & CONFIGURATIONS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = BASE_DIR

# Artifact paths
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "credit_risk_model.pkl")
SAMPLE_PATH = os.path.join(ARTIFACT_DIR, "sample_applications.csv")

# Feature mappings for form-to-model conversion
FEATURE_MAPPINGS = {
    'basic': ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 
              'annual_inc', 'dti', 'credit_history_years', 'installment'],
    'categorical': {
        'grade': ['B', 'C', 'D', 'E', 'F', 'G'],
        'home_ownership': ['NONE', 'OTHER', 'OWN', 'RENT', 'MORTGAGE']
    },
    'engineered': ['loan_to_income_ratio', 'installment_to_income_ratio', 
                   'revol_bal_to_income_ratio', 'term_months']
}

# Validation constraints
MAX_UPLOAD_SIZE_MB = 10  # Reduced to 10MB for practical use
ALLOWED_FILE_TYPES = ['.csv']

# =====================================================
# SYNTHETIC DATA GENERATOR (FOR DEMO)
# =====================================================

def generate_demo_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic demo data for testing without uploading real dataset.
    This allows anyone to test the model immediately.
    """
    np.random.seed(42)
    
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    home_ownerships = ['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE']
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business']
    
    # Generate random home ownership for each sample
    random_home_ownership = np.random.choice(home_ownerships, n_samples)
    
    data = {
        'loan_amnt': np.random.uniform(1000, 35000, n_samples),
        'funded_amnt': np.random.uniform(1000, 35000, n_samples),
        'funded_amnt_inv': np.random.uniform(1000, 35000, n_samples),
        'term_months': np.random.choice([36, 60], n_samples).astype(float),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'installment': np.random.uniform(30, 1200, n_samples),
        'grade': np.random.choice(grades, n_samples),
        'annual_inc': np.random.uniform(20000, 150000, n_samples),
        'dti': np.random.uniform(0, 35, n_samples),
        'credit_history_years': np.random.uniform(0, 30, n_samples),
        'home_ownership': random_home_ownership,
        'purpose': np.random.choice(purposes, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add engineered features
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['annual_inc']
    df['installment_to_income_ratio'] = df['installment'] / df['annual_inc']
    df['revol_bal_to_income_ratio'] = np.random.uniform(0, 0.5, n_samples)
    
    # Add one-hot encoded columns for grade
    for g in FEATURE_MAPPINGS['categorical']['grade']:
        df[f'grade_{g}'] = np.where(df['grade'] == g, 1.0, 0.0)
    
    # Add one-hot encoded columns for home_ownership
    for h in FEATURE_MAPPINGS['categorical']['home_ownership']:
        df[f'home_ownership_{h}'] = np.where(df['home_ownership'] == h, 1.0, 0.0)
    
    # Add other potentially required columns with defaults
    df['Unnamed: 0'] = np.arange(n_samples).astype(float)
    
    # Add additional common features that might be needed (with safe defaults)
    df['revol_bal'] = np.random.uniform(0, 50000, n_samples)
    df['revol_util'] = np.random.uniform(0, 100, n_samples)
    df['total_acc'] = np.random.uniform(5, 50, n_samples)
    df['open_acc'] = np.random.uniform(1, 30, n_samples)
    df['pub_rec'] = np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]).astype(float)
    df['delinq_2yrs'] = np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03]).astype(float)
    df['inq_last_6mths'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]).astype(float)
    
    logger.info(f"Generated {n_samples} synthetic samples with {df.shape[1]} features")
    
    return df

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def safe_float_conversion(value, default: float = 0.0) -> float:
    """Safely convert value to float with fallback to default."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to float, using default {default}")
        return default


def validate_uploaded_file(uploaded_file, max_size_mb: int = MAX_UPLOAD_SIZE_MB) -> bool:
    """Validate uploaded CSV file for security and size constraints."""
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"File terlalu besar. Maksimal {max_size_mb}MB. "
            f"File Anda: {uploaded_file.size / (1024*1024):.2f}MB\n\n"
            f"💡 **Tip**: Gunakan subset data (sampel 1000-5000 baris) untuk testing."
        )
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in ALLOWED_FILE_TYPES:
        raise ValueError(f"Hanya file CSV yang diperbolehkan. File Anda: {file_ext}")
    
    logger.info(f"File validation passed: {uploaded_file.name} ({uploaded_file.size} bytes)")
    return True


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to match model training format."""
    rename_map = {
        'Unnamed: 0': 'Unnamed_0',
        'verification_status_Source Verified': 'verification_status_Source_Verified'
    }
    
    df = df.rename(columns=rename_map)
    df.columns = [col.replace(" ", "_").replace(":", "_") for col in df.columns]
    
    return df


def validate_features(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Validate that DataFrame contains all expected features with correct data types."""
    missing = set(expected_features) - set(df.columns)
    if missing:
        logger.warning(f"Missing features will be filled with 0: {missing}")
        for col in missing:
            df[col] = 0.0
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.warning(f"Converted column {col} to numeric (with coercion)")
            except Exception as e:
                raise ValueError(f"Column {col} cannot be converted to numeric: {str(e)}")
    
    # Handle NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        logger.warning(f"Found NaN values in columns: {nan_counts[nan_counts > 0].to_dict()}")
        df = df.fillna(0)
    
    return df

# =====================================================
# ARTIFACT LOADING
# =====================================================

@st.cache_resource(ttl=3600)
def load_artifacts() -> Tuple[Dict, List[str], object, object]:
    """Load model artifacts with 1-hour cache."""
    logger.info("Loading model artifacts...")
    
    required_files = {
        "metadata.json": METADATA_PATH,
        "features.pkl": FEATURES_PATH,
        "scaler.pkl": SCALER_PATH,
        "credit_risk_model.pkl": MODEL_PATH
    }
    
    missing = [name for name, path in required_files.items() if not os.path.exists(path)]
    
    if missing:
        error_msg = (
            f"Required artifacts not found: {', '.join(missing)}. "
            f"Ensure these files exist in: {ARTIFACT_DIR}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        
        features = joblib.load(FEATURES_PATH)
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        
        logger.info(f"Artifacts loaded successfully. Model: {metadata.get('model_type', 'Unknown')}")
        logger.info(f"Number of features: {len(features)}")
        
        return metadata, features, scaler, model
    
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}", exc_info=True)
        raise


@st.cache_data(ttl=1800)
def load_sample_data() -> Optional[pd.DataFrame]:
    """Load sample CSV for demo (30 min cache)."""
    if not os.path.exists(SAMPLE_PATH):
        logger.info("Sample data file not found")
        return None
    
    try:
        df = pd.read_csv(SAMPLE_PATH)
        if df.shape[0] == 0:
            logger.warning("Sample data file is empty")
            return None
        
        logger.info(f"Sample data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        return None

# =====================================================
# FEATURE ENGINEERING & PREPROCESSING
# =====================================================

def prepare_features_for_model(df_raw: pd.DataFrame, features: List[str], scaler: object) -> pd.DataFrame:
    """Prepare raw features for model prediction."""
    logger.info(f"Preparing features for {len(df_raw)} samples...")
    
    # Select only required features
    df = df_raw[features].copy() if all(f in df_raw.columns for f in features) else df_raw.copy()
    
    # Validate and handle data types
    df = validate_features(df, features)
    
    # Reorder to match expected features
    df = df[features]
    
    # Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        try:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            logger.info(f"Scaled {len(numeric_cols)} numeric features")
        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}")
            raise ValueError(f"Feature scaling failed: {str(e)}")
    
    # Standardize column names
    df = standardize_column_names(df)
    
    logger.info("Feature preparation completed")
    return df


def calculate_engineered_features(form_inputs: Dict, loan_amnt: float, 
                                  annual_inc: float, installment: float) -> Dict:
    """Calculate engineered ratio features."""
    safe_annual_inc = max(annual_inc, 1.0)
    
    engineered = {
        'loan_to_income_ratio': loan_amnt / safe_annual_inc,
        'installment_to_income_ratio': installment / safe_annual_inc,
        'revol_bal_to_income_ratio': safe_float_conversion(form_inputs.get('revol_bal', 0)) / safe_annual_inc
    }
    
    return engineered


def build_feature_row_from_form(form_inputs: Dict, features: List[str]) -> pd.DataFrame:
    """Build complete feature row from form inputs."""
    logger.info("Building feature row from form...")
    
    # Initialize with defaults
    row = {feat: 0.0 for feat in features}
    
    # Extract form values
    loan_amnt = safe_float_conversion(form_inputs.get("loan_amnt", 0))
    int_rate = safe_float_conversion(form_inputs.get("int_rate", 0))
    annual_inc = safe_float_conversion(form_inputs.get("annual_inc", 0))
    dti = safe_float_conversion(form_inputs.get("dti", 0))
    credit_history_years = safe_float_conversion(form_inputs.get("credit_history_years", 0))
    
    term_str = form_inputs.get("term", "36 months")
    term_months = 60 if term_str == "60 months" else 36
    installment = loan_amnt / term_months if term_months > 0 else 0.0
    
    # Map basic features
    basic_mapping = {
        'loan_amnt': loan_amnt,
        'funded_amnt': loan_amnt,
        'funded_amnt_inv': loan_amnt,
        'int_rate': int_rate,
        'annual_inc': annual_inc,
        'dti': dti,
        'credit_history_years': credit_history_years,
        'term_months': float(term_months),
        'installment': installment
    }
    
    for key, value in basic_mapping.items():
        if key in row:
            row[key] = value
    
    # Engineered features
    engineered = calculate_engineered_features(form_inputs, loan_amnt, annual_inc, installment)
    for key, value in engineered.items():
        if key in row:
            row[key] = value
    
    # One-hot encode grade
    grade = form_inputs.get("grade", "A")
    for g in FEATURE_MAPPINGS['categorical']['grade']:
        col = f"grade_{g}"
        if col in row:
            row[col] = 1.0 if grade == g else 0.0
    
    # One-hot encode home_ownership
    home_ownership = form_inputs.get("home_ownership", "RENT")
    for h in FEATURE_MAPPINGS['categorical']['home_ownership']:
        col = f"home_ownership_{h}"
        if col in row:
            row[col] = 1.0 if home_ownership == h else 0.0
    
    if "Unnamed: 0" in row:
        row["Unnamed: 0"] = 0.0
    
    logger.info(f"Feature row built with {len(row)} features")
    return pd.DataFrame([row])

# =====================================================
# PREDICTION FUNCTIONS
# =====================================================

def predict_batch(df_raw: pd.DataFrame, features: List[str], scaler: object, 
                  model: object, threshold: float) -> pd.DataFrame:
    """Perform batch prediction."""
    logger.info(f"Batch prediction: {len(df_raw)} samples, threshold={threshold}")
    
    try:
        X = prepare_features_for_model(df_raw, features, scaler)
        prob_default = model.predict_proba(X)[:, 1]
        labels = (prob_default >= threshold).astype(int)
        
        result = df_raw.copy()
        result["prob_default"] = prob_default
        result["label"] = labels
        result["risk_flag"] = np.where(
            labels == 1, "High Risk (Bad Loan)", "Low Risk (Good Loan)"
        )
        
        logger.info(f"Prediction done. High Risk: {labels.sum()}, Low Risk: {(1-labels).sum()}")
        return result
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise


def predict_single(form_inputs: Dict, features: List[str], scaler: object, 
                   model: object, threshold: float) -> Dict:
    """Perform single prediction from form."""
    logger.info("Single prediction starting...")
    
    try:
        X_raw = build_feature_row_from_form(form_inputs, features)
        X = prepare_features_for_model(X_raw, features, scaler)
        
        prob_default = float(model.predict_proba(X)[:, 1][0])
        label = int(prob_default >= threshold)
        risk_flag = "High Risk (Bad Loan)" if label == 1 else "Low Risk (Good Loan)"
        
        result = {
            "prob_default": prob_default,
            "label": label,
            "risk_flag": risk_flag,
            "threshold": threshold,
            "raw_input": form_inputs,
            "prediction_time": datetime.now().isoformat()
        }
        
        logger.info(f"Single prediction done: {risk_flag} (prob={prob_default:.3f})")
        return result
    
    except Exception as e:
        logger.error(f"Single prediction failed: {str(e)}", exc_info=True)
        raise

# =====================================================
# STREAMLIT APP INITIALIZATION
# =====================================================

def initialize_app():
    """Initialize Streamlit app."""
    st.set_page_config(
        page_title="Credit Risk Prediction – Lending Company",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        metadata, features, scaler, model = load_artifacts()
        return metadata, features, scaler, model, None
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(f"Initialization failed: {error_msg}")
        return None, None, None, None, error_msg


# Initialize
metadata, FEATURES, scaler, model, error = initialize_app()

# Handle initialization errors
if error:
    st.title("💳 Credit Risk Prediction – Configuration Error")
    st.error(
        "⚠️ **Aplikasi tidak dapat dimulai karena artefak model tidak ditemukan.**\n\n"
        f"**Detail Error:**\n```\n{error}\n```\n\n"
        "**Solusi:**\n"
        "1. Pastikan file berikut ada di direktori aplikasi:\n"
        "   - `metadata.json`\n"
        "   - `features.pkl`\n"
        "   - `scaler.pkl`\n"
        "   - `credit_risk_model.pkl`\n\n"
        "2. Jalankan script training terlebih dahulu.\n\n"
        "3. Verifikasi path direktori sesuai struktur repo."
    )
    st.stop()

# =====================================================
# MAIN APP LAYOUT
# =====================================================

st.title("💳 Credit Risk Prediction – Lending Company")

st.markdown(
    """
    Dashboard prediksi risiko kredit berbasis **Machine Learning** untuk menilai kelayakan aplikasi pinjaman.
    Model dilatih menggunakan data historis LendingClub (2007–2014).
    
    **🎯 Fitur Utama:**
    - 🧍 **Single Prediction**: Input manual via form
    - 🎲 **Generate Demo Data**: Buat data sintetis untuk testing (TIDAK PERLU UPLOAD!)
    - 📊 **Sample Data**: Demo dengan data sample (jika tersedia)
    - 📂 **Batch Upload**: Upload CSV kecil untuk testing (maks 10MB)
    """
)

# Add explanation expander
with st.expander("📚 **Panduan Lengkap Penggunaan Aplikasi** (Klik untuk membuka)"):
    st.markdown("""
    ### 🎓 Apa itu Credit Risk Prediction?
    
    **Credit Risk Prediction** adalah sistem yang menggunakan **Machine Learning** untuk memprediksi 
    kemungkinan seorang pemohon pinjaman akan **gagal bayar (default)** atau tidak.
    
    **Analogi sederhana:**
    > Seperti ketika bank menilai apakah Anda layak diberi pinjaman, tapi menggunakan AI yang belajar 
    > dari data ratusan ribu aplikasi pinjaman sebelumnya.
    
    ---
    
    ### 🤔 Mengapa ini Penting?
    
    - **Untuk Bank/Lender**: Mengurangi risiko kerugian dari pinjaman bermasalah
    - **Untuk Peminjam**: Proses approval lebih cepat dan objektif
    - **Untuk Bisnis**: Otomasi keputusan kredit dengan akurasi tinggi
    
    ---
    
    ### 🎯 Cara Kerja Sistem Ini:
    
    ```
    INPUT                    PROSES                      OUTPUT
    ┌──────────────┐        ┌──────────────┐          ┌──────────────┐
    │ Data Pemohon │   →    │ Model ML     │    →     │ Keputusan    │
    │ - Gaji       │        │ (XGBoost)    │          │ - APPROVE ✅ │
    │ - Pinjaman   │        │ Analisis     │          │ - REJECT  ❌ │
    │ - Grade Kredit│       │ 87 Fitur     │          │ + Probabilitas│
    └──────────────┘        └──────────────┘          └──────────────┘
    ```
    
    ---
    
    ### 📋 Cara Menggunakan Aplikasi:
    
    #### **Tab 1: 🧍 Single Prediction (Paling Mudah untuk Pemula)**
    
    **Kapan digunakan?**
    - Testing satu aplikasi pinjaman
    - Demo ke orang lain
    - Memahami cara kerja model
    
    **Langkah-langkah:**
    1. **Isi Form dengan Data Pemohon:**
       - **Loan Amount**: Jumlah uang yang ingin dipinjam (contoh: $10,000)
       - **Term**: Jangka waktu pinjaman (36 bulan = 3 tahun, 60 bulan = 5 tahun)
       - **Interest Rate**: Bunga pinjaman per tahun (contoh: 13.5%)
       - **Annual Income**: Gaji per tahun (contoh: $60,000)
       - **DTI (Debt-to-Income)**: Persentase hutang dibanding gaji (contoh: 18%)
       - **Credit History**: Berapa lama memiliki riwayat kredit (contoh: 7 tahun)
       - **Credit Grade**: Rating kredit A (terbaik) hingga G (terburuk)
       - **Home Ownership**: Status kepemilikan rumah (Rent/Mortgage/Own)
       - **Purpose**: Tujuan pinjaman (konsolidasi hutang, kartu kredit, dll)
    
    2. **Klik "Predict Risk"**
    
    3. **Baca Hasil:**
       - 🟢 **APPROVE (Low Risk)**: Pemohon kemungkinan besar akan bayar lancar
       - 🔴 **REJECT (High Risk)**: Pemohon berisiko tinggi gagal bayar
       - **Probability**: Persentase risiko (contoh: 25% = risiko rendah, 75% = risiko tinggi)
    
    ---
    
    #### **Tab 2: 🎲 Generate Demo Data (RECOMMENDED!)**
    
    **Kapan digunakan?**
    - **Tidak punya dataset** tapi ingin test aplikasi
    - Demo cepat untuk recruiter/interviewer
    - Testing model dengan banyak skenario sekaligus
    
    **Keunggulan:**
    - ✅ Tidak perlu download/upload file apapun
    - ✅ Data dibuat otomatis dalam hitungan detik
    - ✅ Langsung lihat hasil prediksi ratusan aplikasi
    
    **Langkah-langkah:**
    1. Pilih jumlah data yang ingin di-generate (10-500)
    2. Klik "Generate & Predict"
    3. Lihat summary & download hasilnya
    
    ---
    
    #### **Tab 3: 📊 Sample Data**
    
    **Kapan digunakan?**
    - Jika repo sudah menyediakan sample data
    - Testing dengan data "real" yang sudah di-preprocessed
    
    ---
    
    #### **Tab 4: 📂 Batch Upload**
    
    **Kapan digunakan?**
    - Punya data CSV sendiri (max 10MB)
    - Testing dengan data custom
    
    **Format CSV yang Dibutuhkan:**
    - Minimal kolom: loan_amnt, int_rate, annual_inc, grade, dll
    - Tidak perlu semua kolom (missing akan diisi otomatis)
    
    ---
    
    ### 📊 Memahami Hasil Prediksi:
    
    #### **1. Default Probability (Probabilitas Gagal Bayar)**
    - **0% - 30%**: Risiko Rendah 🟢 → APPROVE
    - **30% - 50%**: Risiko Menengah 🟡 → Review Manual
    - **50% - 100%**: Risiko Tinggi 🔴 → REJECT
    
    #### **2. Risk Flag**
    - **Low Risk (Good Loan)**: Pemohon diprediksi akan bayar lancar ✅
    - **High Risk (Bad Loan)**: Pemohon berisiko tinggi gagal bayar ❌
    
    #### **3. Threshold (Ambang Batas Keputusan)**
    - Default: **0.30** (30%)
    - Jika probability ≥ threshold → HIGH RISK
    - Jika probability < threshold → LOW RISK
    - Bisa diatur di sidebar (lebih rendah = lebih ketat, lebih tinggi = lebih longgar)
    
    ---
    
    ### 💡 Tips untuk Hasil Terbaik:
    
    1. **Input yang Akurat**: Pastikan data yang dimasukkan realistis dan benar
    2. **Pahami Context**: Model ini tool bantu, bukan pengganti keputusan manusia
    3. **Gunakan Threshold Bijak**: 
       - Bisnis konservatif → threshold rendah (0.20-0.30)
       - Bisnis agresif → threshold tinggi (0.40-0.50)
    4. **Review Manual untuk Borderline Cases**: Probability di sekitar threshold perlu review tambahan
    
    ---
    
    ### ⚠️ Disclaimer:
    
    - Model ini untuk **tujuan demonstrasi dan edukasi**
    - Keputusan kredit aktual harus melibatkan **review manual** oleh credit analyst
    - Pertimbangkan faktor lain: karakter peminjam, kondisi ekonomi, dll
    - Model dilatih dengan data 2007-2014, kondisi ekonomi bisa berubah
    """)

# Metrics
metrics = metadata.get("metrics", {})
optimal_threshold = metadata.get("optimal_threshold", 0.5)
model_type = metadata.get("model_type", "Unknown Model")
model_version = metadata.get("model_version", "N/A")
training_date = metadata.get("training_date", "N/A")

st.subheader("📈 Model Performance Summary")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Model", model_type)
with col2:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
with col3:
    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
with col4:
    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
with col5:
    st.metric("Optimal Threshold", f"{optimal_threshold:.2f}")

# Add metrics explanation
with st.expander("❓ **Penjelasan Metrik Model** (Apa arti angka-angka di atas?)"):
    st.markdown(f"""
    ### 🤖 Model: **{model_type}**
    
    **XGBoost (eXtreme Gradient Boosting)** adalah algoritma Machine Learning yang sangat powerful untuk prediksi.
    
    **Analogi sederhana:**
    > Bayangkan 100 ahli kredit senior yang voting apakah aplikasi harus disetujui atau ditolak. 
    > XGBoost menggabungkan pendapat mereka dengan cara yang sangat cerdas.
    
    **Kenapa XGBoost?**
    - ✅ Akurasi tinggi untuk prediksi finansial
    - ✅ Bisa menangani data kompleks dengan banyak fitur
    - ✅ Robust terhadap data yang tidak sempurna
    - ✅ Digunakan oleh bank-bank besar dan fintech
    
    **"Tuned"** artinya model ini sudah dioptimasi dengan hyperparameter terbaik melalui proses trial & error.
    
    ---
    
    ### 📊 Accuracy: **{metrics.get('accuracy', 0):.1%}**
    
    **Apa artinya?**
    - Dari 100 aplikasi pinjaman, model benar memprediksi **{metrics.get('accuracy', 0)*100:.0f} aplikasi**
    - {100 - metrics.get('accuracy', 0)*100:.0f} sisanya salah prediksi
    
    **Interpretasi:**
    - 75-80%: Bagus ✅ (standar industri fintech)
    - 80-85%: Sangat Bagus 🌟
    - >85%: Excellent 🏆 (tapi hati-hati overfitting)
    
    **Catatan:** Accuracy saja tidak cukup! Kita perlu lihat metrik lain.
    
    ---
    
    ### ⚖️ F1-Score: **{metrics.get('f1_score', 0):.3f}**
    
    **Apa artinya?**
    F1-Score adalah **keseimbangan** antara:
    - **Precision**: Dari yang diprediksi HIGH RISK, berapa yang benar-benar HIGH RISK?
    - **Recall**: Dari semua yang benar-benar HIGH RISK, berapa yang berhasil ditangkap?
    
    **Analogi sederhana:**
    > Seperti dokter mendiagnosis penyakit:
    > - Precision: Jika dokter bilang Anda sakit, seberapa yakin diagnosis itu benar?
    > - Recall: Dari semua orang yang benar-benar sakit, berapa yang berhasil didiagnosis?
    
    **Interpretasi untuk F1-Score {metrics.get('f1_score', 0):.3f}:**
    - 0.0 - 0.3: Lemah ⚠️ (model kesulitan deteksi bad loans)
    - 0.3 - 0.5: Cukup ✅ (reasonable untuk imbalanced data)
    - 0.5 - 0.7: Bagus 🌟
    - >0.7: Excellent 🏆
    
    **Kenapa F1-Score penting di Credit Risk?**
    - Data tidak seimbang: Good loans jauh lebih banyak dari Bad loans
    - Kita butuh model yang bisa **detect bad loans** tanpa terlalu banyak **false alarm**
    
    **Nilai {metrics.get('f1_score', 0):.3f} artinya:**
    Model cukup baik mendeteksi bad loans, cocok untuk data yang imbalanced seperti credit risk.
    
    ---
    
    ### 📈 ROC-AUC: **{metrics.get('roc_auc', 0):.3f}**
    
    **Apa artinya?**
    ROC-AUC mengukur **kemampuan model membedakan** antara Good Loan dan Bad Loan.
    - Nilai 0.5: Model seperti tebakan acak (lempar koin) 🪙
    - Nilai 1.0: Model sempurna (tidak realistis) 
    
    **Interpretasi ROC-AUC {metrics.get('roc_auc', 0):.3f}:**
    - 0.5 - 0.6: Lemah (hampir random)
    - 0.6 - 0.7: Cukup ✅
    - **0.7 - 0.8: Bagus** 🌟 ← Model kita di sini!
    - 0.8 - 0.9: Sangat Bagus
    - >0.9: Excellent (atau overfitting)
    
    **Analogi sederhana:**
    > Jika Anda tunjukkan 2 aplikasi (1 good, 1 bad) ke model, ada {metrics.get('roc_auc', 0)*100:.0f}% 
    > kemungkinan model bisa correctly rank mana yang lebih berisiko.
    
    **ROC-AUC {metrics.get('roc_auc', 0):.1%} artinya:**
    Model memiliki kemampuan diskriminasi yang bagus untuk membedakan peminjam berisiko tinggi vs rendah.
    
    ---
    
    ### 🎯 Optimal Threshold: **{optimal_threshold:.2f}** (30%)
    
    **Apa artinya?**
    Ini adalah **titik potong** untuk memutuskan APPROVE atau REJECT.
    
    **Cara kerja:**
    - Jika probability ≥ 0.30 (30%) → **HIGH RISK** → REJECT ❌
    - Jika probability < 0.30 (30%) → **LOW RISK** → APPROVE ✅
    
    **Kenapa 0.30?**
    Nilai ini dipilih berdasarkan **analisis cost-benefit**:
    - Terlalu rendah (0.10): Terlalu banyak reject, kehilangan customer bagus
    - Terlalu tinggi (0.60): Terlalu banyak approve, banyak bad loans lolos
    - **0.30**: Sweet spot yang balance risk dan opportunity
    
    **Analogi bisnis:**
    ```
    Threshold 0.20 (Konservatif)     Threshold 0.30 (Balanced)      Threshold 0.50 (Agresif)
    ├─────────────────────────────┼─────────────────────────────┼──────────────────────────┤
    Reject banyak                  Balance                       Approve banyak
    Kehilangan revenue             Optimal risk/reward           Risiko tinggi bad loans
    Safety first                   Recommended ✅                Growth aggressive
    ```
    
    **Bisa diubah?**
    Ya! Di sidebar Anda bisa adjust threshold sesuai risk appetite:
    - **Conservative (0.20-0.25)**: Untuk lembaga yang sangat risk-averse
    - **Balanced (0.30-0.35)**: Recommended untuk most cases ✅
    - **Aggressive (0.40-0.50)**: Untuk lembaga yang prioritaskan growth
    
    ---
    
    ### 📚 Kesimpulan Model Performance:
    
    | Metrik | Nilai | Rating | Interpretasi |
    |--------|-------|--------|--------------|
    | **Accuracy** | {metrics.get('accuracy', 0):.1%} | {'🌟' if metrics.get('accuracy', 0) >= 0.75 else '✅'} | Model benar {metrics.get('accuracy', 0)*100:.0f}% dari waktu |
    | **F1-Score** | {metrics.get('f1_score', 0):.3f} | {'✅' if metrics.get('f1_score', 0) >= 0.3 else '⚠️'} | Cukup baik untuk imbalanced data |
    | **ROC-AUC** | {metrics.get('roc_auc', 0):.3f} | {'🌟' if metrics.get('roc_auc', 0) >= 0.7 else '✅'} | Kemampuan diskriminasi bagus |
    | **Threshold** | {optimal_threshold:.2f} | ✅ | Balanced risk-reward |
    
    **Overall Assessment:**
    Model ini memiliki performa yang **solid dan production-ready** untuk credit risk prediction. 
    Cocok untuk lending companies yang ingin **mengotomasi screening awal** aplikasi pinjaman 
    sambil tetap melakukan **manual review untuk borderline cases**.
    
    ---
    
    ### 💼 Use Case di Dunia Nyata:
    
    **Skenario 1: Fintech Lending**
    - Model ini bisa proses 10,000 aplikasi/hari
    - Auto-approve low risk (< 20%)
    - Auto-reject high risk (> 60%)
    - Manual review untuk middle range (20-60%)
    - Result: 70% automated, 30% manual review
    
    **Skenario 2: Bank Digital**
    - Integrasikan dalam mobile app
    - Instant pre-approval dalam 60 detik
    - Improve customer experience
    - Reduce operational cost 50%
    
    **Skenario 3: P2P Lending**
    - Show risk score ke investor
    - Biarkan investor pilih risk level mereka
    - Transparansi = trust = more funding
    """)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(optimal_threshold),
    step=0.05,
    help="Probabilitas minimum untuk High Risk"
)

st.sidebar.caption(
    f"**Threshold: {threshold:.2f}**\n\n"
    "- ⬇️ Rendah → Lebih konservatif\n"
    "- ⬆️ Tinggi → Lebih permisif"
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ Model Information"):
    st.write(f"**Version:** {model_version}")
    st.write(f"**Training Date:** {training_date}")
    st.write(f"**Features:** {len(FEATURES)}")
    st.write(f"**Type:** {model_type}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload Model"):
    st.cache_resource.clear()
    st.cache_data.clear()
    logger.info("Cache cleared")
    st.rerun()

# =====================================================
# TABS
# =====================================================

tab_single, tab_generate, tab_sample, tab_batch = st.tabs([
    "🧍 Single Prediction",
    "🎲 Generate Demo Data",
    "📊 Sample Data",
    "📂 Batch Upload"
])

# =====================================================
# TAB 1: SINGLE PREDICTION
# =====================================================

with tab_single:
    st.subheader("🧍 Single Applicant Prediction")
    
    st.markdown(
        """
        Masukkan informasi pemohon untuk prediksi individual.
        Model akan menghitung **probabilitas default** dan memberikan rekomendasi.
        """
    )
    
    # Add input guide
    st.info("""
    💡 **Panduan Singkat Input Form:**
    - Isi semua field dengan data pemohon pinjaman
    - Tidak perlu 100% akurat - ini untuk testing/demo
    - Gunakan nilai default sebagai contoh
    - Hover di tanda (?) untuk penjelasan detail setiap field
    """)
    
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💰 Loan Details**")
            loan_amnt = st.number_input(
                "Loan Amount ($)", 
                min_value=500.0, 
                max_value=40000.0, 
                value=10000.0, 
                step=500.0,
                help="""
                💵 JUMLAH PINJAMAN
                
                Berapa banyak uang yang ingin dipinjam?
                
                Range umum: $1,000 - $40,000
                
                Contoh:
                - $5,000: Pinjaman kecil (kartu kredit, emergency)
                - $10,000: Pinjaman sedang (konsolidasi hutang)
                - $25,000: Pinjaman besar (bisnis, renovasi)
                
                Tips: Pinjaman besar + income rendah = risiko tinggi
                """
            )
            term = st.selectbox(
                "Term", 
                ["36 months", "60 months"],
                help="""
                📅 JANGKA WAKTU PINJAMAN
                
                Berapa lama waktu untuk melunasi?
                
                - 36 months (3 tahun): Cicilan lebih besar, total bunga lebih kecil
                - 60 months (5 tahun): Cicilan lebih kecil, total bunga lebih besar
                
                Contoh cicilan $10,000 @ 13.5%:
                - 36 bulan: ~$340/bulan
                - 60 bulan: ~$230/bulan
                
                Tips: Term lebih panjang = lebih mudah bayar tapi total cost lebih mahal
                """
            )
            int_rate = st.number_input(
                "Interest Rate (%)", 
                min_value=5.0, 
                max_value=35.0, 
                value=13.5, 
                step=0.1,
                help="""
                📊 SUKU BUNGA PER TAHUN
                
                Berapa persen bunga pinjaman?
                
                Range berdasarkan risk:
                - 5-10%: Excellent credit (Grade A-B)
                - 10-15%: Good credit (Grade B-C)
                - 15-20%: Fair credit (Grade C-D)
                - 20-30%: Poor credit (Grade E-F)
                - >30%: Very high risk (Grade G)
                
                Tips: Interest rate tinggi = kompensasi untuk risiko tinggi
                """
            )
        
        with col2:
            st.markdown("**👤 Applicant Profile**")
            annual_inc = st.number_input(
                "Annual Income ($)", 
                min_value=10000.0, 
                max_value=500000.0, 
                value=60000.0, 
                step=1000.0,
                help="""
                💰 PENDAPATAN TAHUNAN (KOTOR)
                
                Berapa total gaji/income per tahun sebelum pajak?
                
                Benchmark USA:
                - $30,000-$40,000: Low income
                - $50,000-$70,000: Middle class (median)
                - $80,000-$120,000: Upper middle class
                - >$150,000: High income
                
                Tips: Income tinggi = kemampuan bayar lebih baik = risiko lebih rendah
                
                Untuk freelancer: Rata-rata income 12 bulan terakhir
                """
            )
            dti = st.number_input(
                "DTI - Debt-to-Income (%)", 
                min_value=0.0, 
                max_value=50.0, 
                value=18.0, 
                step=0.1,
                help="""
                📊 DEBT-TO-INCOME RATIO
                
                Persentase hutang bulanan terhadap income bulanan
                
                Rumus: (Total hutang per bulan / Income per bulan) × 100
                
                Contoh:
                Income $5,000/bulan
                Hutang (KPR + kartu kredit + cicilan): $900/bulan
                DTI = (900/5000) × 100 = 18%
                
                Interpretasi:
                - 0-20%: Excellent (sedikit hutang)
                - 20-30%: Good (hutang terkontrol)
                - 30-40%: Borderline (mulai berat)
                - >40%: High risk (hutang overload)
                
                Tips: Lender biasanya prefer DTI < 36%
                """
            )
            credit_history_years = st.number_input(
                "Credit History (years)", 
                min_value=0.0, 
                max_value=40.0, 
                value=7.0, 
                step=0.5,
                help="""
                📅 LAMA RIWAYAT KREDIT
                
                Berapa lama sudah memiliki track record kredit?
                
                Dihitung sejak pertama kali:
                - Buka kartu kredit
                - Ambil pinjaman
                - Menjadi authorized user
                
                Interpretasi:
                - 0-2 tahun: Credit newbie (thin file)
                - 3-5 tahun: Masih muda, tapi ada data
                - 5-10 tahun: Cukup mature
                - >10 tahun: Established (good sign)
                
                Tips: Semakin panjang history (asalkan bagus) = semakin dipercaya lender
                
                Note: Ini bukan umur, tapi lama punya credit history!
                """
            )
        
        with col3:
            st.markdown("**📊 Credit Profile**")
            grade = st.selectbox(
                "Credit Grade", 
                ["A", "B", "C", "D", "E", "F", "G"],
                index=1,  # Default ke B
                help="""
                🏆 CREDIT GRADE / RATING
                
                Rating kredit berdasarkan credit score & risk assessment
                
                Grade breakdown:
                - A: Excellent (Credit Score 720+) - Top 20% peminjam
                - B: Very Good (680-719) - Low risk
                - C: Good (640-679) - Moderate risk
                - D: Fair (600-639) - Increased risk
                - E: Poor (560-599) - High risk
                - F: Very Poor (520-559) - Very high risk
                - G: Bad (< 520) - Highest risk
                
                Faktor yang mempengaruhi:
                - Payment history (35%)
                - Credit utilization (30%)
                - Length of credit history (15%)
                - New credit (10%)
                - Credit mix (10%)
                
                Tips: Grade A-B dapat rate terbaik, E-G dapat rate tinggi atau ditolak
                """
            )
            home_ownership = st.selectbox(
                "Home Ownership", 
                ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"],
                help="""
                🏠 STATUS KEPEMILIKAN RUMAH
                
                Bagaimana status tempat tinggal saat ini?
                
                Pilihan:
                - RENT: Sewa/kontrak rumah (paling umum untuk young adults)
                - MORTGAGE: Punya rumah tapi masih cicil KPR
                - OWN: Punya rumah lunas (free and clear)
                - OTHER: Tinggal dengan ortu/saudara/company housing
                - NONE: Homeless / tidak fixed address
                
                Interpretasi risk:
                - OWN (Lunas): Best - asset stabil, tidak ada beban KPR
                - MORTGAGE: Good - komitmen jangka panjang, tapi ada cicilan
                - RENT: Neutral - fleksibel tapi tidak ada asset
                - OTHER: Depends - perlu context lebih
                - NONE: Red flag - tidak ada stabilitas
                
                Tips: OWN/MORTGAGE lebih dipercaya karena tunjukkan stabilitas
                """
            )
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
                help="""
                🎯 TUJUAN PINJAMAN
                
                Untuk apa uang ini akan digunakan?
                
                Tujuan umum:
                
                1. debt_consolidation (40%)
                   - Gabungkan banyak hutang jadi satu
                   - Biasanya untuk lower interest rate
                   - Relatively lower risk
                
                2. credit_card (20%)
                   - Bayar credit card debt
                   - High interest debt → lower interest loan
                   - Smart financial move
                
                3. home_improvement (15%)
                   - Renovasi/perbaikan rumah
                   - Increase home value
                   - Moderate risk
                
                4. small_business (10%)
                   - Modal usaha kecil
                   - Higher risk (business bisa gagal)
                   - Perlu business plan
                
                5. major_purchase (5%)
                   - Beli barang besar (furniture, appliance)
                   - Moderate risk
                
                6. car (5%)
                   - Beli mobil
                   - Asset-backed, lower risk
                
                7. other (5%)
                   - Medical, wedding, vacation, dll
                   - Risk varies
                
                Tips: Debt consolidation dan credit card payoff dilihat positif karena tunjukkan financial responsibility
                """
            )
        
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submit = st.form_submit_button("🔍 Predict Risk", use_container_width=True, type="primary")
    
    if submit:
        inputs = {
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
            "annual_inc": annual_inc, "dti": dti, "credit_history_years": credit_history_years,
            "grade": grade, "home_ownership": home_ownership, "purpose": purpose
        }
        
        try:
            with st.spinner("Processing..."):
                result = predict_single(inputs, FEATURES, scaler, model, threshold)
            
            prob = result["prob_default"]
            label = result["label"]
            risk = result["risk_flag"]
            
            color = "#ff4444" if label == 1 else "#44ff44"
            decision = "❌ REJECT" if label == 1 else "✅ APPROVE"
            recommendation = (
                "Risiko tinggi untuk default. Disarankan **REJECT** atau review manual." 
                if label == 1 else 
                "Risiko rendah. Aplikasi dapat **APPROVED** dengan monitoring standar."
            )
            
            st.markdown("### 📋 Prediction Result")
            st.markdown(
                f"""
                <div style="padding:2rem;border-radius:0.5rem;border:2px solid {color};">
                    <h2 style="color:{color};margin:0;">{decision}</h2>
                    <p style="font-size:1.1rem;margin-top:1rem;">
                        <b>Risk:</b> {risk}<br>
                        <b>Default Probability:</b> {prob:.1%}<br>
                        <b>Threshold:</b> {threshold:.2f}<br>
                    </p>
                    <hr>
                    <p><b>Recommendation:</b><br>{recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Default Prob", f"{prob:.1%}")
            with col_b:
                st.metric("Threshold", f"{threshold:.0%}")
            with col_c:
                st.metric("Margin", f"{abs(prob - threshold):.1%}")
            
            with st.expander("📝 Input Details"):
                st.json(result["raw_input"])
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Single prediction error: {e}", exc_info=True)

# =====================================================
# TAB 2: GENERATE DEMO DATA (NO UPLOAD NEEDED!)
# =====================================================

with tab_generate:
    st.subheader("🎲 Generate Synthetic Demo Data")
    
    st.markdown(
        """
        **💡 Fitur Terbaik untuk Testing Tanpa Dataset!**
        
        Tidak punya dataset 229 MB? Tidak masalah! Generate data sintetis langsung di aplikasi.
        Data ini **dibuat otomatis** menggunakan distribusi statistik yang realistis.
        
        **Keunggulan:**
        - ✅ Tidak perlu upload file besar
        - ✅ Instant testing untuk siapa saja
        - ✅ Berguna untuk demo ke recruiter/stakeholder
        - ✅ Data random tapi realistis
        """
    )
    
    st.info(
        "💡 **Perfect untuk:**\n"
        "- Orang yang ingin test proyek Anda tapi tidak punya dataset\n"
        "- Demo cepat tanpa persiapan data\n"
        "- Testing model dengan berbagai skenario\n"
        "- Presentasi ke recruiter/interviewer"
    )
    
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        n_samples = st.slider(
            "Jumlah Data Sintetis",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Berapa banyak aplikasi pinjaman yang akan di-generate"
        )
    
    with col_gen2:
        st.metric("Estimated Processing Time", f"~{n_samples * 0.01:.1f}s")
        st.caption("Waktu tergantung ukuran data & performa sistem")
    
    if st.button("🎲 Generate & Predict Demo Data", use_container_width=True, type="primary"):
        try:
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.text("Generating synthetic data...")
            progress_bar.progress(20)
            
            # Generate data
            df_demo = generate_demo_data(n_samples)
            
            status.text(f"Running predictions on {n_samples} samples...")
            progress_bar.progress(50)
            
            # Predict
            df_result = predict_batch(df_demo, FEATURES, scaler, model, threshold)
            
            progress_bar.progress(90)
            status.text("Finalizing...")
            progress_bar.progress(100)
            
            progress_bar.empty()
            status.empty()
            
            st.success(f"✅ Generated & predicted {n_samples} synthetic applications!")
            
            # Summary
            st.markdown("### 📊 Prediction Summary")
            
            summary = df_result["risk_flag"].value_counts()
            total = summary.sum()
            high_risk = summary.get("High Risk (Bad Loan)", 0)
            low_risk = summary.get("Low Risk (Good Loan)", 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", f"{total:,}")
            with col2:
                st.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%")
            with col3:
                st.metric("Low Risk", f"{low_risk:,}", delta=f"{low_risk/total*100:.1f}%")
            with col4:
                st.metric("Avg Prob", f"{df_result['prob_default'].mean():.1%}")
            
            # Display results
            st.markdown("### 📋 Sample Results (10 baris)")
            display_cols = ["loan_amnt", "annual_inc", "grade", "dti", "prob_default", "risk_flag"]
            available_cols = [col for col in display_cols if col in df_result.columns]
            st.dataframe(df_result[available_cols].head(10), use_container_width=True)
            
            # Add table explanation
            with st.expander("📖 **Penjelasan Kolom Tabel** (Klik untuk memahami setiap kolom)"):
                st.markdown("""
                ### 📊 Penjelasan Setiap Kolom dalam Tabel Hasil:
                
                | Kolom | Penjelasan | Contoh | Tips Membaca |
                |-------|------------|--------|--------------|
                | **loan_amnt** | Jumlah pinjaman yang dimohonkan (dalam USD) | $10,000 | Semakin besar pinjaman, semakin tinggi risiko jika gagal bayar |
                | **annual_inc** | Pendapatan tahunan pemohon (dalam USD) | $60,000 | Indikator kemampuan bayar. Income tinggi = risiko lebih rendah |
                | **grade** | Rating kredit pemohon dari A (terbaik) hingga G (terburuk) | B, C, D | A-B: Low risk, C-D: Medium, E-G: High risk |
                | **dti** | Debt-to-Income ratio dalam persen (%) | 18.5% | Total hutang ÷ Income. <30% = bagus, >40% = berisiko |
                | **prob_default** | **Probabilitas gagal bayar** yang dihitung model | 0.25 (25%) | Angka kunci! 0-30%: Low risk, 30-50%: Medium, 50-100%: High risk |
                | **risk_flag** | **Label keputusan akhir** berdasarkan threshold | High Risk / Low Risk | Rekomendasi: APPROVE jika Low Risk, REJECT jika High Risk |
                
                ---
                
                ### 🎯 Cara Membaca Tabel dengan Benar:
                
                **Contoh Baris 1:**
                ```
                loan_amnt: $15,000 | annual_inc: $50,000 | grade: C | dti: 22% | prob_default: 0.35 | risk_flag: High Risk
                ```
                
                **Interpretasi:**
                1. **Pemohon ingin pinjam $15,000** dengan gaji $50,000/tahun
                2. **Credit grade C** = peminjam menengah (bukan excellent, tapi bukan buruk)
                3. **DTI 22%** = hutang 22% dari income (masih reasonable)
                4. **Probability 35%** = model prediksi ada 35% chance gagal bayar
                5. **Risk Flag: High Risk** = karena 35% > threshold 30%, masuk kategori HIGH RISK
                
                **Keputusan:** 
                - ❌ **REJECT** - atau minimal perlu **review manual** lebih detail
                - Alasan: Meskipun DTI bagus, tapi kombinasi grade C + pinjaman relatif besar membuat risiko > 30%
                
                ---
                
                **Contoh Baris 2:**
                ```
                loan_amnt: $8,000 | annual_inc: $75,000 | grade: A | dti: 12% | prob_default: 0.15 | risk_flag: Low Risk
                ```
                
                **Interpretasi:**
                1. **Pemohon ingin pinjam $8,000** dengan gaji $75,000/tahun (ratio rendah)
                2. **Credit grade A** = peminjam excellent!
                3. **DTI 12%** = hutang sangat rendah (bagus sekali)
                4. **Probability 15%** = hanya 15% chance gagal bayar
                5. **Risk Flag: Low Risk** = 15% < threshold 30%, kategori LOW RISK
                
                **Keputusan:**
                - ✅ **APPROVE** dengan percaya diri
                - Alasan: Profile excellent, income tinggi, hutang rendah, pinjaman wajar
                
                ---
                
                ### 💡 Tips Analisis Cepat:
                
                **Lihat kombinasi ini untuk quick assessment:**
                
                1. **Grade + prob_default:**
                   - Grade A-B + prob < 20% = 🟢 Approve confident
                   - Grade C-D + prob 20-40% = 🟡 Review manual
                   - Grade E-G + prob > 40% = 🔴 Reject atau tinggi bunga
                
                2. **loan_amnt vs annual_inc (Loan-to-Income Ratio):**
                   - Pinjaman < 20% income = 🟢 Safe
                   - Pinjaman 20-40% income = 🟡 Moderate
                   - Pinjaman > 40% income = 🔴 Risky
                
                3. **DTI (Debt-to-Income):**
                   - < 20% = 🟢 Excellent
                   - 20-30% = 🟡 Good
                   - 30-40% = 🟠 Borderline
                   - > 40% = 🔴 Red flag
                
                ---
                
                ### 📊 Skenario Decision Making:
                
                **Skenario A: Conservative Lending (Risk-Averse)**
                - Approve hanya jika prob_default < 20%
                - Grade minimal B
                - DTI maksimal 25%
                - Result: Reject rate tinggi, tapi bad loan rate minimal
                
                **Skenario B: Balanced Approach (Recommended)**
                - Approve jika prob_default < 30%
                - Accept grade hingga C dengan review
                - DTI maksimal 35%
                - Result: Balance antara growth dan risk
                
                **Skenario C: Aggressive Growth**
                - Approve hingga prob_default 40%
                - Accept grade hingga D
                - DTI maksimal 40%
                - Kompensasi dengan bunga lebih tinggi
                - Result: Volume tinggi, tapi perlu monitoring ketat
                
                ---
                
                ### ⚠️ Red Flags yang Perlu Diperhatikan:
                
                Kombinasi berbahaya (perlu extra scrutiny):
                
                1. **High loan + Low income:**
                   - loan_amnt > $20,000 + annual_inc < $40,000
                   - Ratio pinjaman > 50% dari income tahunan
                
                2. **Poor grade + High DTI:**
                   - Grade D/E/F/G + DTI > 30%
                   - Sudah banyak hutang + track record buruk
                
                3. **High prob walaupun profile bagus:**
                   - Grade A/B tapi prob_default > 30%
                   - Ada hidden factors yang ditangkap model
                   - Perlu investigasi lebih lanjut
                
                4. **Inconsistency:**
                   - Income tinggi + Grade buruk (suspicious)
                   - DTI rendah tapi prob_default tinggi (check data quality)
                """)
            
            # Distribution
            with st.expander("📈 Probability Distribution"):
                st.bar_chart(df_result["prob_default"].value_counts(bins=20).sort_index())
                st.caption("""
                **Cara membaca chart:**
                - X-axis: Range probabilitas default (0.0 - 1.0)
                - Y-axis: Jumlah aplikasi dalam range tersebut
                - Distribusi ideal: Mayoritas di kiri (low probability) 
                - Jika banyak di kanan (high probability) = portfolio berisiko tinggi
                """)
            
            # Download
            st.markdown("### ⬇️ Download Results")
            csv_data = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Generated Data with Predictions",
                data=csv_data,
                file_name=f"demo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(
                "💡 **Tip**: Data ini bisa di-download dan digunakan untuk testing lebih lanjut!"
            )
        
        except Exception as e:
            st.error(f"❌ Error generating data: {str(e)}")
            logger.error(f"Demo generation error: {e}", exc_info=True)

# =====================================================
# TAB 3: SAMPLE DATA (IF AVAILABLE)
# =====================================================

with tab_sample:
    st.subheader("📊 Demo with Sample Data")
    
    st.markdown(
        """
        Tab ini menggunakan **sample data** yang sudah disediakan (jika ada).
        Berguna untuk demonstrasi dengan data real yang sudah di-preprocessed.
        """
    )
    
    sample_df = load_sample_data()
    
    if sample_df is None:
        st.warning(
            "⚠️ **Sample data tidak tersedia**\n\n"
            "**Alternatif:**\n"
            "1. Gunakan tab **Generate Demo Data** untuk testing instant\n"
            "2. Atau buat file `sample_applications.csv` (50-200 baris) di root directory\n\n"
            "**Rekomendasi**: Gunakan fitur Generate Demo Data untuk kemudahan!"
        )
        
        st.info(
            "💡 **Tidak ada sample data?**\n\n"
            "Klik tab **🎲 Generate Demo Data** untuk membuat data testing secara otomatis!"
        )
    else:
        st.success(f"✅ Sample loaded: **{sample_df.shape[0]:,} rows** × **{sample_df.shape[1]} columns**")
        
        with st.expander("👁️ Preview Sample Data"):
            st.dataframe(sample_df.head(10), use_container_width=True)
        
        max_rows = sample_df.shape[0]
        
        if max_rows <= 10:
            n_rows = st.slider("Jumlah baris", 1, max_rows, max_rows, 1)
        else:
            n_rows = st.slider("Jumlah baris", 10, min(200, max_rows), min(50, max_rows), 10)
        
        if st.button("🚀 Run Prediction", use_container_width=True):
            try:
                progress_bar = st.progress(0)
                status = st.empty()
                
                status.text(f"Processing {n_rows} rows...")
                progress_bar.progress(25)
                
                df_sample_subset = sample_df.head(n_rows)
                
                status.text("Running predictions...")
                progress_bar.progress(50)
                
                df_result = predict_batch(df_sample_subset, FEATURES, scaler, model, threshold)
                
                progress_bar.progress(90)
                status.text("Done!")
                progress_bar.progress(100)
                
                progress_bar.empty()
                status.empty()
                
                st.success(f"✅ Prediction completed for {n_rows} applications!")
                
                # Summary
                st.markdown("### 📊 Summary")
                
                summary = df_result["risk_flag"].value_counts()
                total = summary.sum()
                high_risk = summary.get("High Risk (Bad Loan)", 0)
                low_risk = summary.get("Low Risk (Good Loan)", 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", f"{total:,}")
                with col2:
                    st.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%")
                with col3:
                    st.metric("Low Risk", f"{low_risk:,}", delta=f"{low_risk/total*100:.1f}%")
                with col4:
                    st.metric("Avg Prob", f"{df_result['prob_default'].mean():.1%}")
                
                # Results
                st.markdown("### 📋 Predictions")
                display_cols = ["loan_amnt", "annual_inc", "grade", "prob_default", "risk_flag"]
                available_cols = [col for col in display_cols if col in df_result.columns]
                st.dataframe(df_result[available_cols].head(10), use_container_width=True)
                
                # Distribution
                with st.expander("📈 Distribution"):
                    st.bar_chart(df_result["prob_default"].value_counts(bins=20).sort_index())
                
                # Download
                csv_data = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results",
                    data=csv_data,
                    file_name=f"sample_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Sample prediction error: {e}", exc_info=True)

# =====================================================
# TAB 4: BATCH UPLOAD (OPTIONAL, SMALL FILES ONLY)
# =====================================================

with tab_batch:
    st.subheader("📂 Batch Prediction – Upload CSV")
    
    st.markdown(
        """
        Upload CSV untuk prediksi batch. **Fitur ini opsional** - hanya untuk testing dengan data custom.
        
        **⚠️ Batasan:**
        - Maksimal ukuran: **10 MB** (bukan 229 MB!)
        - Untuk dataset besar, gunakan batch processing offline
        - Idealnya 1,000 - 10,000 baris
        """
    )
    
    st.info(
        "💡 **Rekomendasi:**\n\n"
        "Jika tidak punya data:\n"
        "1. Gunakan **🎲 Generate Demo Data** - paling mudah!\n"
        "2. Atau buat subset kecil dari dataset training (1,000-5,000 baris)\n"
        "3. Export ke CSV dan upload di sini"
    )
    
    uploaded_file = st.file_uploader(
        "Upload CSV (max 10MB)",
        type=["csv"],
        help="CSV dengan kolom sesuai training data"
    )
    
    if uploaded_file is not None:
        try:
            validate_uploaded_file(uploaded_file)
            
            with st.spinner("Reading file..."):
                df_input = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{df_input.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{df_input.shape[1]}")
            with col3:
                st.metric("Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
            
            with st.expander("👁️ Preview Data"):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            missing_features = [f for f in FEATURES if f not in df_input.columns]
            if missing_features:
                st.warning(
                    f"⚠️ Missing {len(missing_features)} features. "
                    f"Will use default values for: {', '.join(missing_features[:5])}..."
                )
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                try:
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    status.text(f"Processing {df_input.shape[0]:,} rows...")
                    progress_bar.progress(10)
                    
                    status.text("Preparing features...")
                    progress_bar.progress(30)
                    
                    status.text("Running predictions...")
                    progress_bar.progress(60)
                    
                    df_result = predict_batch(df_input, FEATURES, scaler, model, threshold)
                    
                    progress_bar.progress(90)
                    status.text("Finalizing...")
                    progress_bar.progress(100)
                    
                    progress_bar.empty()
                    status.empty()
                    
                    st.success(f"✅ Batch prediction completed for {df_result.shape[0]:,} applications!")
                    
                    # Summary
                    st.markdown("### 📊 Summary")
                    
                    summary = df_result["risk_flag"].value_counts()
                    total = summary.sum()
                    high_risk = summary.get("High Risk (Bad Loan)", 0)
                    low_risk = summary.get("Low Risk (Good Loan)", 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", f"{total:,}")
                    with col2:
                        st.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%")
                    with col3:
                        st.metric("Low Risk", f"{low_risk:,}", delta=f"{low_risk/total*100:.1f}%")
                    with col4:
                        st.metric("Avg Prob", f"{df_result['prob_default'].mean():.1%}")
                    
                    # Risk table
                    st.markdown("### 📋 Risk Distribution")
                    risk_table = pd.DataFrame({
                        "Risk Category": summary.index,
                        "Count": summary.values,
                        "Percentage": [f"{v/total*100:.1f}%" for v in summary.values]
                    })
                    st.table(risk_table)
                    
                    # Results preview
                    st.markdown("### 📄 Detailed Results (20 rows)")
                    display_cols = ["loan_amnt", "annual_inc", "grade", "dti", "prob_default", "label", "risk_flag"]
                    available_cols = [col for col in display_cols if col in df_result.columns]
                    st.dataframe(df_result[available_cols].head(20), use_container_width=True)
                    
                    # Distribution
                    with st.expander("📈 Probability Distribution"):
                        prob_bins = pd.cut(df_result["prob_default"], bins=10)
                        prob_dist = prob_bins.value_counts().sort_index()
                        st.bar_chart(prob_dist)
                    
                    # Download
                    st.markdown("### ⬇️ Download Results")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        csv_full = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv_full,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        df_high_risk = df_result[df_result["label"] == 1]
                        csv_high_risk = df_high_risk.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⚠️ Download High Risk Only",
                            data=csv_high_risk,
                            file_name=f"high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(
                        f"❌ **Batch Prediction Error**\n\n"
                        f"```\n{str(e)}\n```\n\n"
                        "**Troubleshooting:**\n"
                        "- Periksa format CSV (kolom, data types)\n"
                        "- Coba dengan sample kecil terlebih dahulu\n"
                        "- Gunakan Generate Demo Data untuk test model"
                    )
                    logger.error(f"Batch error: {e}", exc_info=True)
        
        except ValueError as e:
            st.error(f"❌ **Validation Error**\n\n{str(e)}")
            logger.error(f"Validation error: {e}")
        
        except Exception as e:
            st.error(f"❌ **File Reading Error**\n\n```\n{str(e)}\n```")
            logger.error(f"File reading error: {e}", exc_info=True)
    
    else:
        st.info(
            "📌 **Belum upload file?**\n\n"
            "**Pilihan terbaik:**\n"
            "1. 🎲 **Generate Demo Data** - Instant testing tanpa upload!\n"
            "2. 🧍 **Single Prediction** - Test satu per satu via form\n"
            "3. 📊 **Sample Data** - Jika tersedia sample di repo\n\n"
            "Upload hanya untuk data custom yang sudah Anda miliki."
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("💳 **Credit Risk Prediction System**")
    st.caption(f"Model Version: {model_version}")

with footer_col2:
    st.caption(f"📅 Training Date: {training_date}")
    st.caption(f"🔢 Features: {len(FEATURES)}")

with footer_col3:
    st.caption("🏗️ Built with Streamlit")
    st.caption("🤖 Powered by XGBoost")

st.markdown(
    """
    <div style="text-align:center;padding:1rem;color:#888;">
        <small>
        <b>💡 Pro Tip:</b> Tidak punya dataset 229 MB? Gunakan fitur <b>Generate Demo Data</b> 
        untuk instant testing tanpa perlu upload file apapun!<br><br>
        <i>Disclaimer: Model ini untuk tujuan demonstrasi. 
        Keputusan kredit aktual memerlukan review manual dan analisis komprehensif.</i>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
