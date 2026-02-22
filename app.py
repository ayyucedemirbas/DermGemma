import streamlit as st
import os
import torch
from PIL import Image
import time

st.set_page_config(
    page_title="DermGemma AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }

    .agent-card {
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .agent-card-vision {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .agent-card-nurse {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .agent-card-diagnostician {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }

    .result-box {
        background: white;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .info-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_step = 0
    st.session_state.visual_findings = None
    st.session_state.patient_history = []
    st.session_state.diagnosis = None
    st.session_state.clinic = None

# Header
st.markdown("""
<div class="main-header">
    <h1>DermGemma AI Assistant</h1>
    <p>Advanced AI-Powered Dermatological Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### System Status")

    if not st.session_state.initialized:
        st.warning("System not initialized")
        if st.button("Initialize AI Models", use_container_width=True):
            with st.spinner("Loading AI models... This may take a few minutes."):
                try:
                    from main_code import AgenticDermatologist
                    st.session_state.clinic = AgenticDermatologist()
                    st.session_state.initialized = True
                    st.success("Models loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.success("System Online")

    st.markdown("---")
    st.markdown("### Analysis Progress")

    steps = ["Upload Image", "Visual Analysis", "Patient Interview", "Diagnosis"]
    for i, step in enumerate(steps):
        if i < st.session_state.current_step:
            st.markdown(f"**{step}**")
        elif i == st.session_state.current_step:
            st.markdown(f"**{step}** (Current)")
        else:
            st.markdown(f"{step}")

    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **DermGemma** combines:
    -  Vision AI for lesion analysis
    - Conversational patient intake
    - RAG-based clinical guidelines
    - Advanced diagnostic reasoning
    """)

    if st.button("Reset Analysis", use_container_width=True):
        st.session_state.current_step = 0
        st.session_state.visual_findings = None
        st.session_state.patient_history = []
        st.session_state.diagnosis = None
        st.rerun()

# Main content
if not st.session_state.initialized:
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to DermGemma!</h3>
        <p>Please initialize the AI models using the sidebar to begin your dermatological analysis.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Advanced lesion image analysis</li>
            <li>Interactive patient interview</li>
            <li>Evidence-based diagnostic recommendations</li>
            <li>Multi-agent AI collaboration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Step 1: Image Upload
    if st.session_state.current_step == 0:
        st.markdown("## Step 1: Upload Image")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload a dermatological image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of the skin"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("Analyze Image", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        temp_path = "temp_upload.jpg"
                        image.save(temp_path)
                        st.session_state.visual_findings = st.session_state.clinic.agent_vision_describe(temp_path)
                        st.session_state.current_step = 1
                        os.remove(temp_path)
                        st.rerun()

        with col2:
            st.markdown("""
            <div class="agent-card agent-card-vision">
                <h3>Vision Agent</h3>
                <p><strong>Specialty:</strong> Semeiotics Analysis</p>
                <p><strong>Function:</strong> Analyzes visual features</p>
            </div>
            """, unsafe_allow_html=True)

    # Step 2: Visual Findings
    elif st.session_state.current_step == 1:
        st.markdown("## Step 2: Visual Analysis Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div class="result-box">
                <h3>Visual Findings</h3>
                <p>{st.session_state.visual_findings}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Continue to Interview", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()

        with col2:
            st.markdown("""
            <div class="agent-card agent-card-vision">
                <h3>Analysis Complete</h3>
                <p>Visual characteristics analyzed successfully.</p>
            </div>
            """, unsafe_allow_html=True)

    # Step 3: Patient Interview
    elif st.session_state.current_step == 2:
        st.markdown("## Step 3: Patient Interview")

        col1, col2 = st.columns([2, 1])

        with col1:
            patient_input = st.text_area(
                "Describe your symptoms:",
                placeholder="e.g., It's been itching for 2 weeks...",
                height=100
            )

            if patient_input and st.button("Submit", use_container_width=True):
                with st.spinner("Processing..."):
                    follow_up = st.session_state.clinic.agent_intake_interview(patient_input)
                    st.session_state.patient_history.append({
                        'patient': patient_input,
                        'nurse': follow_up
                    })
                    st.rerun()

            if st.session_state.patient_history:
                st.markdown("### Interview History")
                for exchange in st.session_state.patient_history:
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>You:</strong> {exchange['patient']}<br>
                        <strong>Nurse:</strong> {exchange['nurse']}
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("Generate Diagnosis", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()

        with col2:
            st.markdown("""
            <div class="agent-card agent-card-nurse">
                <h3> Intake Nurse</h3>
                <p><strong>Function:</strong> Medical interview</p>
            </div>
            """, unsafe_allow_html=True)

    # Step 4: Diagnosis
    elif st.session_state.current_step == 3:
        st.markdown("## Step 4: Diagnostic Analysis")

        if st.session_state.diagnosis is None:
            with st.spinner("Analyzing..."):
                history_text = " ".join([f"Patient: {h['patient']}. Nurse: {h['nurse']}."
                                        for h in st.session_state.patient_history])
                st.session_state.diagnosis = st.session_state.clinic.agent_diagnostician(
                    st.session_state.visual_findings, history_text
                )
                st.rerun()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div class="result-box">
                <h3>Diagnostic Report</h3>
                <h4>Visual Analysis:</h4>
                <p>{st.session_state.visual_findings}</p>
                <h4>Diagnosis:</h4>
                <p>{st.session_state.diagnosis}</p>
            </div>
            """, unsafe_allow_html=True)

            st.warning("**Disclaimer**: AI-assisted analysis. Consult a licensed dermatologist.")

            if st.button(" New Analysis", use_container_width=True):
                st.session_state.current_step = 0
                st.session_state.visual_findings = None
                st.session_state.patient_history = []
                st.session_state.diagnosis = None
                st.rerun()

        with col2:
            st.markdown("""
            <div class="agent-card agent-card-diagnostician">
                <h3>Diagnostician</h3>
                <p>Analysis complete!</p>
            </div>
            """, unsafe_allow_html=True)
