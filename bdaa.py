import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = 'model_artifacts/final_catboost_model.pkl'
PREPROCESSOR = 'model_artifacts/preprocessor.pkl'

# AgeGroup mapping
AGE_LABELS = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
]

def age_to_agegroup(age_code: int) -> str:
    return AGE_LABELS[age_code - 1]

def age_to_ageband(age_code: int) -> str:
    if 1 <= age_code <= 3:
        return "18-34"
    if 4 <= age_code <= 6:
        return "35-49"
    if 7 <= age_code <= 9:
        return "50-64"
    if 10 <= age_code <= 11:
        return "65-74"
    return "75+"

def riskscore_to_profile(risk_score: int) -> str:
    if risk_score <= 1:
        return "Healthy"
    if risk_score <= 3:
        return "ModerateRisk"
    return "HighRisk"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR)

# Page configuration
st.set_page_config(
    page_title="HeartWise | Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(255, 90, 95, 0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(66, 133, 244, 0.10), transparent 28%),
        linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    color: #172033;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8edf5;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #111827;
    font-weight: 800;
    margin-top: 1.4rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #374151 !important;
}

.hero-card {
    background: linear-gradient(135deg, #761d2f 0%, #b8324b 52%, #ef6b70 100%);
    border-radius: 28px;
    padding: 2.2rem 2.4rem;
    color: white;
    box-shadow: 0 24px 60px rgba(118, 29, 47, 0.25);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: '';
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.14);
    right: -90px;
    top: -90px;
}

.hero-title {
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 0.7rem;
    letter-spacing: -0.04em;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: rgba(255, 255, 255, 0.86);
    max-width: 720px;
    line-height: 1.65;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.clean-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #e7edf5;
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
    margin-bottom: 1.25rem;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #172033;
    margin-bottom: 0.25rem;
}

.card-caption {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.15rem;
}

.mini-card {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-radius: 20px;
    padding: 1.1rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
}

.mini-label {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
}

.mini-value {
    font-size: 1.8rem;
    color: #111827;
    font-weight: 800;
}

.mini-note {
    color: #64748b;
    font-size: 0.86rem;
    margin-top: 0.2rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-weight: 800;
    font-size: 0.85rem;
}

.pill-green {
    background: #dcfce7;
    color: #166534;
}

.pill-yellow {
    background: #fef9c3;
    color: #854d0e;
}

.pill-red {
    background: #fee2e2;
    color: #991b1b;
}

.result-card {
    border-radius: 28px;
    padding: 1.7rem;
    color: #111827;
    margin-top: 1rem;
    border: 1px solid;
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
}

.result-low {
    background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
    border-color: #bbf7d0;
}

.result-moderate {
    background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    border-color: #fde68a;
}

.result-high {
    background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    border-color: #fecaca;
}

.result-percentage {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -0.05em;
    margin: 0.2rem 0;
}

.progress-shell {
    width: 100%;
    height: 18px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 1rem;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
}

.recommendation-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-top: 1rem;
    color: #334155;
    line-height: 1.55;
}

.factor-list {
    background: #f8fafc;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 0.75rem;
}

.factor-list strong {
    color: #111827;
}

.disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 20px;
    padding: 1.1rem;
    color: #7c2d12;
    line-height: 1.55;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 18px;
    padding: 0.95rem 1.3rem;
    font-weight: 800;
    font-size: 1rem;
    background: linear-gradient(135deg, #be123c 0%, #e11d48 100%);
    color: white;
    box-shadow: 0 14px 28px rgba(190, 18, 60, 0.24);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 36px rgba(190, 18, 60, 0.30);
}

.stButton > button:active {
    transform: translateY(0px);
}

[data-testid="stMetricValue"] {
    font-weight: 900;
    color: #111827;
}

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
}

.streamlit-expanderHeader {
    border-radius: 16px;
    font-weight: 800;
}

.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.92rem;
    padding: 1.5rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)

model = load_model()
preprocessor = load_preprocessor()

# Sidebar inputs
with st.sidebar:
    st.markdown("## Patient Profile")
    st.caption("Provide your health and lifestyle information to generate a heart disease risk assessment.")

    st.markdown("### Patient Details")
    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name"
    )

    if patient_name:
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            padding: 12px 16px;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            margin-bottom: 10px;
            font-weight: 600;
            color: #1e293b;
        ">
            👋 Hi, <strong>{patient_name}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Demographics")
    agegroup_label = st.selectbox("Age Group", AGE_LABELS, index=8)
    age_code = AGE_LABELS.index(agegroup_label) + 1
    sex = st.radio("Sex", ["Female", "Male"], index=1, horizontal=True)
    sex_val = 1 if sex == "Male" else 0

    st.markdown("### Body Measurement")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=80.0, value=24.5, step=0.1)

    if bmi < 18.5:
        st.markdown('<span class="status-pill pill-yellow">Underweight</span>', unsafe_allow_html=True)
    elif bmi < 25:
        st.markdown('<span class="status-pill pill-green">Normal weight</span>', unsafe_allow_html=True)
    elif bmi < 30:
        st.markdown('<span class="status-pill pill-yellow">Overweight</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-red">Obese</span>', unsafe_allow_html=True)

    st.markdown("### Health Status")
    phys_hlth = st.slider("Physical Health: poor days in last 30", 0, 30, 0)
    ment_hlth = st.slider("Mental Health: poor days in last 30", 0, 30, 0)

    st.markdown("### Medical History")
    col_a, col_b = st.columns(2)
    with col_a:
        highbp = st.checkbox("High Blood Pressure")
        diabetes = st.checkbox("Diabetes")
    with col_b:
        highchol = st.checkbox("High Cholesterol")
        stroke = st.checkbox("History of Stroke")

    st.markdown("### Lifestyle")
    smoker = st.checkbox("Current Smoker")
    physactivity = st.checkbox("Regular Physical Activity", value=True)
    hvy_alcohol = st.checkbox("Heavy Alcohol Consumption")
    fruits = st.checkbox("Regular Fruit Consumption", value=True)
    veggies = st.checkbox("Regular Vegetable Consumption", value=True)

# Feature engineering
health_stress_index = float(ment_hlth + phys_hlth)
disease_count = int(highbp) + int(highchol) + int(diabetes) + int(stroke)
obese_flag = float(1 if bmi >= 30 else 0)

risk_score = int(
    int(smoker) +
    int(hvy_alcohol) +
    (1 - int(physactivity)) +
    (1 - int(fruits)) +
    (1 - int(veggies))
)

age_group = age_to_agegroup(age_code)
age_band = age_to_ageband(age_code)
lifestyle_profile = riskscore_to_profile(risk_score)

# Dataframe
X = pd.DataFrame([{
    "Age": float(age_code),
    "PhysHlth": float(phys_hlth),
    "MentHlth": float(ment_hlth),
    "HealthStressIndex": float(health_stress_index),
    "DiseaseCount": float(disease_count),
    "ObeseFlag": float(obese_flag),
    "RiskScore": float(risk_score),
    "BMI": float(bmi),
    "Sex": float(sex_val),
    "HighBP": float(highbp),
    "HighChol": float(highchol),
    "Diabetes": float(diabetes),
    "Stroke": float(stroke),
    "Smoker": float(smoker),
    "PhysActivity": float(physactivity),
    "AgeGroup": age_group,
    "AgeBand": age_band,
    "LifestyleProfile": lifestyle_profile
}])

# Header
st.markdown("""
<div class="hero-card">
    <div class="hero-chip">🫀 AI Health Screening Dashboard</div>
    <div class="hero-title">Heart Disease Risk Predictor</div>
    <div class="hero-subtitle">
        A clean and structured cardiovascular risk screening interface using patient demographics,
        medical history, health status, and lifestyle factors.
    </div>
</div>
""", unsafe_allow_html=True)

# Main layout
left_col, right_col = st.columns([1.7, 1], gap="large")

with left_col:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Patient Risk Overview</div>
        <div class="card-caption">Key engineered indicators generated from the patient inputs.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Health Stress", f"{health_stress_index:.0f}", help="Physical + mental poor health days")
    with m2:
        st.metric("Disease Count", disease_count, help="High BP, high cholesterol, diabetes and stroke count")
    with m3:
        st.metric("Lifestyle Risk", risk_score, help="Risk score based on smoking, alcohol, activity, fruits and vegetables")
    with m4:
        st.metric("Age Band", age_band, help="Grouped age band used by the model")

    st.markdown("<br>", unsafe_allow_html=True)

    profile_col1, profile_col2, profile_col3 = st.columns(3)
    with profile_col1:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">BMI Status</div>
            <div class="mini-value">{bmi:.1f}</div>
            <div class="mini-note">Obese flag: {int(obese_flag)}</div>
        </div>
        """, unsafe_allow_html=True)
    with profile_col2:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Lifestyle Profile</div>
            <div class="mini-value" style="font-size:1.45rem;">{lifestyle_profile}</div>
            <div class="mini-note">Based on lifestyle risk score</div>
        </div>
        """, unsafe_allow_html=True)
    with profile_col3:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Age Group</div>
            <div class="mini-value" style="font-size:1.45rem;">{age_group}</div>
            <div class="mini-note">Encoded age value: {age_code}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyse Heart Disease Risk"):
        with st.spinner("Analysing patient information..."):
            X_processed = preprocessor.transform(X)

            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_processed)[0][1])
            else:
                proba = float(model.predict(X_processed)[0])

            risk_percentage = proba * 100

            if risk_percentage < 30:
                level = "Low Risk"
                result_class = "result-low"
                emoji = "✅"
                bar_colour = "#22c55e"
                recommendation = "Continue maintaining a balanced lifestyle, regular physical activity, and routine health checks."
            elif risk_percentage < 70:
                level = "Moderate Risk"
                result_class = "result-moderate"
                emoji = "⚠️"
                bar_colour = "#f59e0b"
                recommendation = "Consider improving lifestyle factors and monitoring blood pressure, cholesterol, BMI, and diabetes-related indicators regularly."
            else:
                level = "High Risk"
                result_class = "result-high"
                emoji = "🚨"
                bar_colour = "#ef4444"
                recommendation = "Seek advice from a qualified healthcare professional for proper clinical assessment and personalised guidance."

            st.markdown(f"""
            <div class="result-card {result_class}">
                <div style="font-size:1rem; font-weight:800; color:#475569;">Prediction Result for {patient_name}</div>
                <div style="font-size:1.65rem; font-weight:900; margin-top:0.2rem;">{emoji} {level}</div>
                <div class="result-percentage" style="color:{bar_colour};">{risk_percentage:.1f}%</div>
                <div style="color:#64748b; font-size:0.98rem;">Estimated probability of heart disease based on the current inputs.</div>
                <div class="progress-shell">
                    <div class="progress-fill" style="width:{risk_percentage}%; background:{bar_colour};"></div>
                </div>
                <div class="recommendation-box">
                    <strong>Recommendation:</strong> {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="clean-card">
        <div class="card-title">Clinical Input Guide</div>
        <div class="card-caption">This section summarises what the interface collects and why it matters.</div>
        <div class="factor-list">🧍 <strong>Demographics:</strong> Age and sex are used as baseline risk indicators.</div>
        <div class="factor-list">⚖️ <strong>BMI:</strong> Helps identify weight-related cardiovascular risk.</div>
        <div class="factor-list">🩺 <strong>Medical history:</strong> Blood pressure, cholesterol, diabetes and stroke are key chronic condition indicators.</div>
        <div class="factor-list">🏃 <strong>Lifestyle:</strong> Smoking, alcohol, exercise, fruits and vegetables contribute to the lifestyle profile.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Important disclaimer:</strong><br>
        This application is a predictive screening tool only. It should not be treated as a medical diagnosis.
        Users should consult qualified healthcare professionals for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Built for cardiovascular risk awareness using machine learning. Prevention, monitoring and professional consultation remain essential.
</div>
""", unsafe_allow_html=True)
