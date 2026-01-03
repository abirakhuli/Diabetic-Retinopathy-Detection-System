import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import time
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Page Config with wider layout
st.set_page_config(
    page_title="Diabetic Retinopathy Detector Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem !important;
        margin-bottom: 3rem !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .severity-0 { background-color: #e8f5e9 !important; }
    .severity-1 { background-color: #fff3e0 !important; }
    .severity-2 { background-color: #fff8e1 !important; }
    .severity-3 { background-color: #ffebee !important; }
    .severity-4 { background-color: #fce4ec !important; }
    
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
    }
    
    .stage-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .stage-dot {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        z-index: 2;
    }
    
    .stage-line {
        position: absolute;
        top: 20px;
        left: 20px;
        right: 20px;
        height: 3px;
        background: #e0e0e0;
        z-index: 1;
    }
    
    .risk-low { color: #4CAF50; font-weight: bold; }
    .risk-medium { color: #FF9800; font-weight: bold; }
    .risk-high { color: #F44336; font-weight: bold; }
    .risk-very-high { color: #9C27B0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load models with progress indicators
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... This might take a moment."):
        try:
            # Load Random Forest
            with open('rf_model.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            
            # Load EfficientNetB0 for feature extraction
            IMG_WIDTH, IMG_HEIGHT = 224, 224
            base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                       input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            cnn_model = Model(inputs=base_model.input, outputs=base_model.output)
            
            st.success("✅ Models loaded successfully!")
            return rf_model, cnn_model
            
        except FileNotFoundError:
            st.error("❌ Model file not found. Please check the file path.")
            return None, None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    This tool uses:
    - **EfficientNetB0** for feature extraction
    - **Random Forest** for classification
    - Trained on retinal images
    """)
    
    st.markdown("---")
    st.markdown("### 📊 DR Stages")
    st.markdown("""
    **0:** No DR - Healthy retina  
    **1:** Mild - Microaneurysms only  
    **2:** Moderate - More than just microaneurysms  
    **3:** Severe - Many hemorrhages  
    **4:** Proliferative - New blood vessels
    """)
    
    st.markdown("---")
    st.markdown("#### 🚨 Emergency Advice")
    st.warning("If experiencing sudden vision loss or severe eye pain, seek **immediate medical attention**!")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detector Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Diabetic Retinopathy Detection', unsafe_allow_html=True)

# Load models
rf_model, cnn_model = load_models()

# File upload section
col1, col2 = st.columns([3, 2])
with col1:
    uploaded_file = st.file_uploader(
        "📁 **Upload Retinal Image**", 
        type=["jpg", "png", "jpeg"],
        help="Upload a clear retinal fundus image for analysis"
    )

# Detailed information about DR stages
st.markdown("---")
st.markdown("### 📋 Understanding Diabetic Retinopathy Stages")

stages_info = {
    0: {
        "name": "No Diabetic Retinopathy",
        "description": "No abnormalities detected. Regular annual checkups recommended.",
        "symptoms": "No visible symptoms",
        "recommendation": "Maintain good blood sugar control and annual eye exams",
        "risk": "Very Low",
        "urgency": "Routine"
    },
    1: {
        "name": "Mild Nonproliferative Retinopathy",
        "description": "Early stage with microaneurysms (small balloon-like swellings in retina's blood vessels).",
        "symptoms": "Usually no symptoms",
        "recommendation": "Monitor closely, improve blood sugar control",
        "risk": "Low",
        "urgency": "Regular Monitoring"
    },
    2: {
        "name": "Moderate Nonproliferative Retinopathy",
        "description": "Blood vessels that nourish retina are swelling and distorting.",
        "symptoms": "Possible blurred vision",
        "recommendation": "Regular monitoring every 6-12 months",
        "risk": "Medium",
        "urgency": "Close Monitoring"
    },
    3: {
        "name": "Severe Nonproliferative Retinopathy",
        "description": "More blood vessels are blocked, depriving retina of nourishment.",
        "symptoms": "Significant vision problems",
        "recommendation": "Immediate ophthalmologist consultation",
        "risk": "High",
        "urgency": "Urgent"
    },
    4: {
        "name": "Proliferative Retinopathy",
        "description": "Advanced stage where new fragile blood vessels grow on retina.",
        "symptoms": "Severe vision loss, floaters",
        "recommendation": "Urgent medical attention required",
        "risk": "Very High",
        "urgency": "Emergency"
    }
}

# Display stages as cards
st.markdown("### 📊 Severity Scale Overview")
cols = st.columns(5)
colors = ['#4CAF50', '#FF9800', '#FFC107', '#F44336', '#9C27B0']
for i, (stage, info) in enumerate(stages_info.items()):
    with cols[i]:
        st.markdown(f'<div class="severity-{stage} info-box">', unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {colors[i]}; text-align: center;'>Stage {stage}</h3>", unsafe_allow_html=True)
        st.markdown(f"**{info['name']}**")
        st.markdown("---")
        st.markdown(f"**Risk:** <span class='risk-{info['risk'].lower().replace(' ', '-')}'>{info['risk']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Urgency:** {info['urgency']}")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(image, caption="Retinal Fundus Image", use_column_width=True)
        
        # Image metadata
        st.markdown("#### Image Details")
        metadata_col1, metadata_col2 = st.columns(2)
        with metadata_col1:
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())
        with metadata_col2:
            st.metric("Dimensions", f"{image.width} × {image.height}")
    
    with col2:
        st.markdown("### 🔍 Analysis Panel")
        
        if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
            if rf_model is not None and cnn_model is not None:
                with st.spinner("Analyzing retinal image..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # 1. Preprocess Image
                    for i in range(25):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    img_resized = image.resize((224, 224))
                    x = img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    
                    # 2. Extract Features
                    for i in range(25, 50):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    features = cnn_model.predict(x, verbose=0)
                    features_flat = features.flatten().reshape(1, -1)
                    
                    # 3. Get Prediction and Probabilities
                    for i in range(50, 75):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    prediction = rf_model.predict(features_flat)[0]
                    
                    # Get probabilities if available
                    if hasattr(rf_model, 'predict_proba'):
                        probabilities = rf_model.predict_proba(features_flat)[0]
                    else:
                        # If no probabilities, create dummy values
                        probabilities = np.zeros(5)
                        probabilities[prediction] = 1.0
                    
                    for i in range(75, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    st.session_state.prediction_made = True
                    st.session_state.prediction = prediction
                    st.session_state.probabilities = probabilities
                    st.session_state.features = features_flat
                    
                    st.rerun()

# Display results if prediction was made
if st.session_state.get('prediction_made', False):
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    prediction = st.session_state.prediction
    probabilities = st.session_state.probabilities
    
    # Create columns for results
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f'<div class="prediction-card severity-{prediction}">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 **Final Diagnosis**")
        st.markdown(f"<h1 style='color: {colors[prediction]}; text-align: center;'>Stage {prediction}</h1>", unsafe_allow_html=True)
        st.markdown(f"### {stages_info[prediction]['name']}")
        
        # # Confidence indicator
        # confidence = probabilities[prediction]
        # st.markdown(f"**Confidence: {confidence:.1%}**")
        
        # Confidence bar
        # st.progress(float(confidence))
        
        # if confidence < confidence_threshold:
        #     st.warning(f"⚠️ Low confidence prediction. Consider consulting an ophthalmologist.")
        
        # st.markdown("</div>", unsafe_allow_html=True)
        
        # Display detailed information
        st.markdown("#### 📝 Stage Details")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Description:** {stages_info[prediction]['description']}")
            st.warning(f"**Common Symptoms:** {stages_info[prediction]['symptoms']}")
        with col_b:
            risk_class = stages_info[prediction]['risk'].lower().replace(' ', '-')
            st.markdown(f"**Risk Level:** <span class='risk-{risk_class}'>{stages_info[prediction]['risk']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Urgency:** {stages_info[prediction]['urgency']}")
    
    with col2:
        # Visual stage indicator
        st.markdown("### Severity Scale")
        
        # Create a simple stage indicator using HTML/CSS
        st.markdown("""
        <div class="stage-indicator">
            <div class="stage-line"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for stage indicators
        indicator_cols = st.columns(5)
        for i in range(5):
            with indicator_cols[i]:
                is_current = i == prediction
                bg_color = colors[i] if is_current else "#e0e0e0"
                border = f"3px solid {colors[i]}" if is_current else "none"
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="stage-dot" style="background-color: {bg_color}; border: {border}; margin: 0 auto;">
                        {i}
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9rem; font-weight: {'bold' if is_current else 'normal'}">
                        {'Current' if is_current else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Simple bar chart using Streamlit
        st.markdown("### 📈 Confidence Distribution")
        
        # Create a dataframe for the bar chart
        prob_df = pd.DataFrame({
            'Stage': [f"Stage {i}" for i in range(5)],
            'Probability': probabilities,
            'Color': colors
        })
        
        # Display as a bar chart using st.bar_chart
        chart_data = pd.DataFrame({
            'Probability': probabilities
        }, index=[f"Stage {i}" for i in range(5)])
        
        st.bar_chart(chart_data, color='#764ba2')
    
    with col3:
        # Feature information
        st.markdown("### 🔬 Technical Details")
        
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.metric("Features Extracted", f"{st.session_state.features.shape[1]:,}")
            st.metric("Model Architecture", "EfficientNetB0")
        with tech_col2:
            st.metric("Classifier", "Random Forest")
            st.metric("Image Size", "224×224")
        
        # Risk assessment with colored badge
        risk_level = stages_info[prediction]['risk']
        risk_class = risk_level.lower().replace(' ', '-')
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Risk Assessment</h4>
            <h2 class="risk-{risk_class}">{risk_level}</h2>
            <p>Based on the detected stage {prediction}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Urgency indicator
        urgency = stages_info[prediction]['urgency']
        if urgency == "Emergency":
            st.error(f"🚨 {urgency} - Immediate attention required!")
        elif urgency == "Urgent":
            st.warning(f"⚠️ {urgency} - Consult within 1 month")
        else:
            st.info(f"📅 {urgency} - Regular monitoring recommended")
    
    # Next steps section
    st.markdown("---")
    st.markdown("## 🚑 Next Steps & Recommendations")
    
    next_steps_cols = st.columns(3)
    
    with next_steps_cols[0]:
        st.markdown("### 📋 Immediate Actions")
        st.markdown(f"""
        <div class="info-box">
            <p><strong>🩺 Medical Consultation:</strong></p>
            <p>{stages_info[prediction]['recommendation']}</p>
            <hr>
            <p><strong>⏰ Timeline:</strong></p>
            <p>{'Immediate' if prediction >= 3 else 'Within 1 month' if prediction == 2 else '6-12 months'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with next_steps_cols[1]:
        st.markdown("### 💊 Management Tips")
        tips = [
            "📉 **Blood Sugar Control:** Maintain HbA1c < 7%",
            "❤️ **Blood Pressure:** Keep below 130/80 mmHg",
            "🏃 **Regular Exercise:** 30 minutes daily",
            "🥗 **Healthy Diet:** Rich in greens and omega-3",
            "🚭 **No Smoking:** Critical for eye health",
            "👁️ **Regular Exams:** As per doctor's advice"
        ]
        for tip in tips:
            st.markdown(f"• {tip}")
    
    with next_steps_cols[2]:
        st.markdown("### 📅 Follow-up Schedule")
        follow_up_schedule = {
            0: "Annual comprehensive eye exam",
            1: "Follow-up in 6-12 months",
            2: "Follow-up in 4-6 months",
            3: "Consult ophthalmologist within 1 month",
            4: "Emergency referral needed"
        }
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Recommended Follow-up:</h4>
            <h3 style="color: {colors[prediction]};">{follow_up_schedule[prediction]}</h3>
            <hr>
            <p><strong>📞 Emergency Contact:</strong></p>
            <p>If experiencing sudden vision changes, contact emergency services immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional resources
    st.markdown("---")
    st.markdown("### 📚 Additional Resources")
    resource_cols = st.columns(4)
    
    with resource_cols[0]:
        st.markdown("**American Diabetes Association**\n\n[www.diabetes.org](https://www.diabetes.org)")
    with resource_cols[1]:
        st.markdown("**American Academy of Ophthalmology**\n\n[www.aao.org](https://www.aao.org)")
    with resource_cols[2]:
        st.markdown("**National Eye Institute**\n\n[www.nei.nih.gov](https://www.nei.nih.gov)")
    with resource_cols[3]:
        st.markdown("**Diabetes Eye Health Guide**\n\nFree educational materials available")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; margin-top: 20px;">
    ⚠️ **Important Medical Disclaimer:** 
    
    This AI-powered tool is designed for **preliminary screening and educational purposes only**. 
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
    
    - Always seek the advice of qualified ophthalmologists or healthcare providers
    - Do not disregard professional medical advice based on this tool's results
    - In case of eye emergencies, contact emergency services immediately
    
    The developers are not responsible for any medical decisions made based on this tool's output.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; font-size: 0.9rem;">
    <p>👁️ <strong>Diabetic Retinopathy Detector Pro v2.1</strong> | Powered by Arghyadeep Deb | For educational and research purposes only</p>
    <p style="font-size: 0.8rem; margin-top: 10px;">⚠️ This tool should be used under medical supervision</p>
</div>
""", unsafe_allow_html=True)

# If no file uploaded, show instructions
if uploaded_file is None:
    st.markdown("---")
    with st.expander("📖 How to Use This Tool", expanded=True):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **📁 Upload Image:** Click 'Browse files' to upload a retinal fundus image
        2. **🚀 Start Analysis:** Click the 'Start Analysis' button
        3. **📊 Review Results:** Examine the detailed diagnosis and recommendations
        4. **🩺 Take Action:** Follow the recommended next steps
        
        ### 📸 Image Requirements:
        - **Format:** JPG, PNG, or JPEG
        - **Quality:** Clear, well-focused retinal images
        - **Content:** Retinal fundus photographs
        - **Size:** At least 224×224 pixels recommended
        
        ### ⚠️ Important Notes:
        - Ensure proper lighting in the original image
        - Avoid blurry or out-of-focus images
        - The tool works best with high-quality retinal scans
        """)