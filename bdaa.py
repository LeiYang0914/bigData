import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "gbt_model.pkl"

# AgeGroup mapping
AGE_LABELS = [
    "18-24","25-29","30-34","35-39","40-44","45-49",
    "50-54","55-59","60-64","65-69","70-74","75-79","80+"
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

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Animated header */
    .stTitle {
        color: #d32f2f;
        font-size: 3.5rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease;
    }
    
    h1 {
        text-align: center !important;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff5f8 0%, #f3e5f5 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: #2d2d2d !important;
    }
    
    /* Enhanced metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        color: #d32f2f;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #d32f2f, #e91e63, transparent) 1;
        animation: slideInLeft 0.8s ease;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Glowing info boxes */
    .info-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(33, 150, 243, 0.3);
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(33, 150, 243, 0.1) 0%, transparent 70%);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff9f0 0%, #fff3e0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .warning-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(255, 152, 0, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f1f8f4 0%, #e8f5e9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
        animation: successPulse 0.6s ease;
    }
    
    @keyframes successPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .danger-box {
        background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.2);
        animation: dangerShake 0.5s ease;
    }
    
    @keyframes dangerShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    /* Enhanced button with animation */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #d32f2f 0%, #e91e63 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 10px 30px rgba(211, 47, 47, 0.4);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(211, 47, 47, 0.6);
        background: linear-gradient(135deg, #e91e63 0%, #d32f2f 100%);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Floating animation for icons */
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Progress bar enhancement */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4caf50, #8bc34a, #cddc39);
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #d32f2f, #e91e63);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(211,47,47,0.1), rgba(233,30,99,0.1));
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Footer decoration */
    .footer-decoration {
        background: linear-gradient(90deg, transparent, #d32f2f, #e91e63, transparent);
        height: 3px;
        margin: 2rem 0 1rem 0;
        border-radius: 3px;
    }
    
    /* Pulse animation for important elements */
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Spinning loader */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Animated header with icon
st.markdown("""
    <div class="float">
        <h1 style="text-align: center; font-size: 4rem; margin: 0;">❤️</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("# Heart Disease Risk Predictor")
st.markdown('<p class="subtitle">✨ AI-powered cardiovascular health assessment tool ✨</p>', unsafe_allow_html=True)

model = load_model()

# Sidebar inputs
with st.sidebar:
    st.markdown("## 📋 Patient Information")
    st.markdown("---")
    
    # Demographics
    st.markdown("### 👤 Demographics")
    agegroup_label = st.selectbox("Age Group", AGE_LABELS, index=8)
    age_code = AGE_LABELS.index(agegroup_label) + 1
    sex = st.radio("Sex", ["Female", "Male"], index=1)
    sex_val = 1 if sex == "Male" else 0
    
    st.markdown("---")
    
    # Physical Measurements
    st.markdown("### 📏 Physical Measurements")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=80.0, value=24.5, step=0.1)
    
    # BMI interpretation with colored indicators
    if bmi < 18.5:
        st.markdown("🔵 **Underweight**")
    elif bmi < 25:
        st.markdown("🟢 **Normal weight**")
    elif bmi < 30:
        st.markdown("🟡 **Overweight**")
    else:
        st.markdown("🔴 **Obese**")
    
    st.markdown("---")
    
    # Health Status
    st.markdown("### 🏥 Health Status")
    phys_hlth = st.slider("Physical Health (poor days in last 30)", 0, 30, 0)
    ment_hlth = st.slider("Mental Health (poor days in last 30)", 0, 30, 0)
    
    st.markdown("---")
    
    # Medical Conditions
    st.markdown("### 💊 Medical Conditions")
    highbp = st.checkbox("High Blood Pressure")
    highchol = st.checkbox("High Cholesterol")
    diabetes = st.checkbox("Diabetes")
    stroke = st.checkbox("History of Stroke")
    
    st.markdown("---")
    
    # Lifestyle Factors
    st.markdown("### 🏃 Lifestyle Factors")
    smoker = st.checkbox("Current Smoker")
    physactivity = st.checkbox("Regular Physical Activity", value=True)
    hvy_alcohol = st.checkbox("Heavy Alcohol Consumption")
    fruits = st.checkbox("Regular Fruit Consumption", value=True)
    veggies = st.checkbox("Regular Vegetable Consumption", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📊 Health Assessment Summary</div>', unsafe_allow_html=True)
    
    # Calculate metrics
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
    
    # Display metrics in cards
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("💓 Health Stress Index", f"{health_stress_index:.0f}", 
                 help="Sum of physical and mental poor health days")
    
    with metric_col2:
        st.metric("🏥 Disease Count", disease_count,
                 help="Number of chronic conditions")
    
    with metric_col3:
        st.metric("⚠️ Lifestyle Risk Score", risk_score,
                 help="Risk score based on lifestyle factors (0-5)")
    
    # Feature engineering
    age_group = age_to_agegroup(age_code)
    age_band = age_to_ageband(age_code)
    lifestyle_profile = riskscore_to_profile(risk_score)
    
    # Create dataframe
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
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Analyze Heart Disease Risk"):
        with st.spinner("🔄 Analyzing your health data..."):
            pred = model.predict(X)[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0][1]
                
                st.markdown("### 📈 Prediction Results")
                
                if pred == 1:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3 style="color: #d32f2f; margin-top: 0;">⚠️ High Risk Detected</h3>
                        <p style="font-size: 1.1rem;">The model indicates a <strong>high risk</strong> of heart disease.</p>
                        <p style="font-size: 1.3rem; font-weight: 700; color: #d32f2f;">Risk Probability: {proba*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("⚕️ **Recommendation**: Please consult with a healthcare professional for a comprehensive cardiac evaluation.")
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="color: #4caf50; margin-top: 0;">✅ Low Risk Detected</h3>
                        <p style="font-size: 1.1rem;">The model indicates a <strong>low risk</strong> of heart disease.</p>
                        <p style="font-size: 1.3rem; font-weight: 700; color: #4caf50;">Risk Probability: {proba*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("💪 Continue maintaining a healthy lifestyle to keep your heart healthy!")
                
                # Enhanced risk level visualization
                st.markdown("#### 📊 Risk Level Visualization")
                risk_percentage = proba * 100
                
                if risk_percentage < 30:
                    color = "#4caf50"
                    level = "Low Risk"
                    emoji = "✅"
                elif risk_percentage < 70:
                    color = "#ff9800"
                    level = "Moderate Risk"
                    emoji = "⚠️"
                else:
                    color = "#f44336"
                    level = "High Risk"
                    emoji = "🚨"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            border-radius: 15px; padding: 20px; margin: 15px 0; 
                            box-shadow: 0 8px 20px rgba(0,0,0,0.1);">
                    <div style="background-color: #e0e0e0; border-radius: 50px; height: 40px; 
                                position: relative; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="background: linear-gradient(90deg, {color}, {color}dd); 
                                    width: {risk_percentage}%; height: 100%; border-radius: 50px; 
                                    display: flex; align-items: center; justify-content: center; 
                                    color: white; font-weight: bold; font-size: 1.1rem;
                                    box-shadow: 0 2px 10px {color}88;
                                    transition: all 0.5s ease;
                                    animation: slideIn 1s ease;">
                            {emoji} {level} - {risk_percentage:.1f}%
                        </div>
                    </div>
                </div>
                <style>
                    @keyframes slideIn {{
                        from {{ width: 0%; }}
                        to {{ width: {risk_percentage}%; }}
                    }}
                </style>
                """, unsafe_allow_html=True)
    
    # Show feature details (collapsible)
    with st.expander("🔬 View Detailed Feature Analysis"):
        st.dataframe(X, use_container_width=True)

with col2:
    st.markdown('<div class="section-header float">ℹ️ About</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4 style="margin-top: 0; color: #2196f3;">🤖 How it works</h4>
        <p>This tool uses advanced machine learning algorithms to assess heart disease risk based on multiple health and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎯 Key Risk Factors")
    st.markdown("""
    - 🎂 **Age & Sex**: Demographics influence risk
    - ⚖️ **BMI**: Body mass index indicator
    - 💉 **Blood Pressure**: Hypertension status
    - 🧪 **Cholesterol**: Lipid profile levels
    - 🍬 **Diabetes**: Blood sugar management
    - 🚬 **Lifestyle**: Smoking, exercise, nutrition
    - 🧠 **Mental Health**: Stress and wellbeing
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
        <h4 style="margin-top: 0; color: #ff9800;">⚠️ Important Disclaimer</h4>
        <p>This is a <strong>predictive tool</strong> and not a medical diagnosis. Always consult qualified healthcare professionals for medical advice and treatment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown('<div class="footer-decoration"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.1rem;">💡 <strong>Health Tip:</strong> Regular checkups and a balanced lifestyle are your best defense</p>
    <p style="font-size: 1rem; margin-top: 1rem;">🌟 <strong>Remember:</strong> Prevention is better than cure</p>
    <p style="font-size: 0.9rem; margin-top: 1rem; color: #999;">Made with ❤️ for cardiovascular health awareness</p>
</div>
""", unsafe_allow_html=True)