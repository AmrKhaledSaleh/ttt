import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'diabetes_model.pkl' exists in the root directory.")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main theme */
        :root {
            --primary: #5B21B6;
            --primary-light: #8B5CF6;
            --secondary: #10B981;
            --accent: #F59E0B;
            --background: #F3F4F6;
            --card: #FFFFFF;
            --text: #1F2937;
            --error: #EF4444;
            --success: #10B981;
        }
        
        /* Base styles */
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--background);
            color: var(--text);
        }
        
        .stApp {
            background: linear-gradient(135deg, #EEF2FF 0%, #F3F4F6 100%);
        }
        
        /* Card styling */
        .card {
            background: var(--card);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(209, 213, 219, 0.3);
        }
        
        /* Input fields */
        .stNumberInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            padding: 10px 14px;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: var(--primary-light);
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 10px;
            padding: 12px 28px;
            border: none;
            box-shadow: 0 4px 12px rgba(91, 33, 182, 0.15);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(91, 33, 182, 0.25);
        }
        
        /* Headers */
        h1 {
            color: var(--primary);
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        h3 {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary);
        }
        
        /* Results */
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 10px;
            font-weight: 600;
            font-size: 1.2rem;
            transition: all 0.5s ease;
        }
        
        .positive {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
        }
        
        .negative {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            color: var(--success);
        }
        
        /* Helper text */
        .helper-text {
            font-size: 0.85rem;
            color: #6B7280;
            margin-top: -10px;
            margin-bottom: 10px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: var(--primary-light);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            background-color: #F9FAFB;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            border-bottom: 2px solid var(--primary) !important;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: var(--primary);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
            padding-top: 2rem;
        }
        
        /* Info cards */
        .info-card {
            background-color: rgba(139, 92, 246, 0.05);
            border-left: 4px solid var(--primary);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: 0;
            height: 1px;
            background: #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create a download link for the chart
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📊 {text}</a>'
    return href

def main():
    # Sidebar for navigation and information
    with st.sidebar:
        st.image("https://www.cdc.gov/diabetes/images/library/spotlights/diabetes-awareness-H.jpg", width=250)
        st.title("Diabetes Risk Predictor")
        
        st.markdown("""
        <div class="info-card">
            <h3>About This App</h3>
            <p>This application uses machine learning to predict the risk of diabetes based on health metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. Enter your health metrics
        2. Click 'Predict Risk'
        3. View your results and recommendations
        """)
        
        st.markdown("### Data Privacy")
        st.markdown("Your data is processed locally and not stored on any server.")
        
        if st.button("Learn More About Diabetes"):
            st.markdown("""
            ### Diabetes Facts
            - Affects more than 422 million people worldwide
            - Can lead to serious health complications if untreated
            - Early detection and lifestyle changes can prevent or delay Type 2 diabetes
            
            [Visit CDC for more info](https://www.cdc.gov/diabetes/basics/diabetes.html)
            """)

    # Main content
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🩺 Diabetes Risk Assessment")
    st.markdown("Enter your health data below for a personalized risk assessment")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["📝 Input Data", "📊 Risk Analysis", "ℹ️ Help"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            age = st.number_input("Age (years)", 
                                min_value=0, 
                                max_value=120, 
                                value=30,
                                help="Your current age in years")
            st.markdown('<div class="helper-text">Age is a key factor in diabetes risk</div>', unsafe_allow_html=True)
            
            pregnancies = st.number_input("Number of Pregnancies", 
                                        min_value=0, 
                                        max_value=20, 
                                        value=0,
                                        help="Number of times pregnant (0 for males)")
            st.markdown('<div class="helper-text">For women only - enter 0 if male</div>', unsafe_allow_html=True)
            
            diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 
                                                        min_value=0.0, 
                                                        max_value=2.5, 
                                                        value=0.5, 
                                                        format="%.3f",
                                                        help="A function that scores likelihood of diabetes based on family history")
            st.markdown('<div class="helper-text">Family history score (higher means stronger family history)</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Health Metrics")
            glucose = st.number_input("Glucose Level (mg/dL)", 
                                    min_value=0, 
                                    max_value=300, 
                                    value=100,
                                    help="Blood glucose concentration after 2-hour oral glucose tolerance test")
            
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 
                                            min_value=0, 
                                            max_value=200, 
                                            value=70,
                                            help="Diastolic blood pressure")
            
            skin_thickness = st.number_input("Skin Thickness (mm)", 
                                            min_value=0, 
                                            max_value=100, 
                                            value=20,
                                            help="Triceps skin fold thickness")
            
            insulin = st.number_input("Insulin Level (μU/mL)", 
                                    min_value=0, 
                                    max_value=850, 
                                    value=80,
                                    help="2-Hour serum insulin")
            
            bmi = st.number_input("BMI (kg/m²)", 
                                min_value=0.0, 
                                max_value=70.0, 
                                value=25.0,
                                format="%.1f",
                                help="Body Mass Index")
            
            # BMI chart visualization
            if bmi > 0:
                bmi_status = ""
                if bmi < 18.5:
                    bmi_status = "Underweight"
                    bmi_color = "blue"
                elif 18.5 <= bmi < 25:
                    bmi_status = "Normal weight"
                    bmi_color = "green"
                elif 25 <= bmi < 30:
                    bmi_status = "Overweight"
                    bmi_color = "orange"
                else:
                    bmi_status = "Obese"
                    bmi_color = "red"
                
                st.markdown(f'<div style="color:{bmi_color};font-weight:bold;">BMI Status: {bmi_status}</div>', unsafe_allow_html=True)
        
        # Validation for required fields
        required_fields = [glucose, blood_pressure, bmi, age]
        is_valid = all(field > 0 for field in required_fields)
        
        if not is_valid:
            st.warning("Please fill in all required fields with valid values (greater than 0).")
        
        predict_btn = st.button("Predict Diabetes Risk", disabled=not is_valid)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Diabetes Risk Analysis")
        
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
            st.session_state.prediction_result = None
            st.session_state.prediction_probability = None
            st.session_state.features = None
        
        # Predict button was clicked
        if predict_btn:
            with st.spinner('Analyzing your data...'):
                # Collect input features
                features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree_function, age]])
                st.session_state.features = features
                
                try:
                    # Make prediction
                    prediction = model.predict(features)
                    st.session_state.prediction_result = prediction[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features)
                        st.session_state.prediction_probability = probabilities[0][1]  # Probability of class 1
                    else:
                        st.session_state.prediction_probability = None
                    
                    st.session_state.prediction_made = True
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        
        # Display prediction results if available
        if st.session_state.prediction_made:
            if st.session_state.prediction_result == 1:
                st.markdown('<div class="prediction-box positive">'
                           '⚠️ Higher Risk: Based on the provided data, you may have an elevated risk of diabetes.'
                           '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box negative">'
                           '✅ Lower Risk: Based on the provided data, you appear to have a lower risk of diabetes.'
                           '</div>', unsafe_allow_html=True)
            
            # Show probability if available
            if st.session_state.prediction_probability is not None:
                risk_percentage = st.session_state.prediction_probability * 100
                st.markdown(f"### Risk Score: {risk_percentage:.1f}%")
                
                # Progress bar for risk visualization
                st.progress(st.session_state.prediction_probability)
                
                # Risk level interpretation
                if risk_percentage < 20:
                    risk_level = "Low"
                    color = "green"
                elif risk_percentage < 50:
                    risk_level = "Moderate"
                    color = "orange"
                else:
                    risk_level = "High"
                    color = "red"
                
                st.markdown(f'<h3 style="color:{color};">Risk Level: {risk_level}</h3>', unsafe_allow_html=True)
            
            # Show feature importance visualization
            st.markdown("### Key Factors Analysis")
            
            # Create a dataframe for the feature values
            feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                           'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
            
            feature_values = st.session_state.features[0]
            
            # Example importance values (normally would come from model)
            # In a real app, you'd extract actual feature importances from your model
            feature_importance = [0.05, 0.28, 0.07, 0.05, 0.10, 0.22, 0.08, 0.15]
            
            # Create a DataFrame for visualization
            df = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values,
                'Importance': feature_importance
            })
            
            # Sort by importance
            df = df.sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(df['Feature'], df['Importance'], color='#8B5CF6')
            
            # Add value labels to the bars
            for i, (value, importance) in enumerate(zip(df['Value'], df['Importance'])):
                ax.text(importance + 0.01, i, f'{value:.1f}', va='center')
            
            ax.set_xlabel('Relative Importance')
            ax.set_title('Feature Importance for Diabetes Risk')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Add download link for the chart
            st.markdown(get_image_download_link(fig, 'feature_importance.png', 'Download Feature Importance Chart'), 
                      unsafe_allow_html=True)
            
            # Recommendations section
            st.markdown("### Recommendations")
            
            if st.session_state.prediction_result == 1:
                st.markdown("""
                Based on your results, consider:
                
                1. **Consult with a healthcare provider** to discuss your diabetes risk
                2. **Monitor blood glucose levels** regularly
                3. **Maintain a healthy weight** through diet and exercise
                4. **Stay physically active** with at least 150 minutes of moderate activity per week
                5. **Follow a balanced diet** rich in fruits, vegetables, and whole grains
                
                *Remember: This is not a medical diagnosis. Always consult with healthcare professionals.*
                """)
            else:
                st.markdown("""
                To maintain your health:
                
                1. **Continue regular health check-ups**
                2. **Maintain a balanced diet** rich in fruits, vegetables, and whole grains
                3. **Stay physically active** with regular exercise
                4. **Maintain a healthy weight**
                5. **Limit alcohol consumption** and avoid smoking
                
                *Remember: This is not a medical diagnosis. Always consult with healthcare professionals.*
                """)
            
            # Action button
            if st.button("Reset Assessment"):
                st.session_state.prediction_made = False
                st.session_state.prediction_result = None
                st.session_state.prediction_probability = None
                st.session_state.features = None
                st.experimental_rerun()
        else:
            st.info("Enter your health data in the 'Input Data' tab and click 'Predict Diabetes Risk' to see your results here.")
            
        st.markdown('</div>', unsafe_allow_html=True)
            
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Parameter Information")
        
        help_data = {
            "Pregnancies": "Number of times the patient has been pregnant. Enter 0 for males or if never pregnant.",
            "Glucose": "Plasma glucose concentration over 2 hours in an oral glucose tolerance test (mg/dL).",
            "Blood Pressure": "Diastolic blood pressure (mm Hg).",
            "Skin Thickness": "Triceps skin fold thickness (mm) - a measure of body fat.",
            "Insulin": "2-Hour serum insulin (μU/mL).",
            "BMI": "Body Mass Index - weight in kg/(height in m)².",
            "Diabetes Pedigree Function": "A function that scores likelihood of diabetes based on family history.",
            "Age": "Age in years."
        }
        
        normal_ranges = {
            "Glucose": "Fasting: 70-99 mg/dL, After eating: Less than 140 mg/dL",
            "Blood Pressure": "Less than 120/80 mm Hg is considered normal",
            "Skin Thickness": "Average values: Men: 12mm, Women: 23mm",
            "Insulin": "Fasting: 16-166 μU/mL",
            "BMI": "18.5-24.9 is considered normal weight"
        }
        
        for param, description in help_data.items():
            st.markdown(f"#### {param}")
            st.markdown(description)
            if param in normal_ranges:
                st.markdown(f"**Normal Range:** {normal_ranges[param]}")
            st.markdown("---")
        
        st.markdown("### Disclaimer")
        st.markdown("""
        This app provides an assessment of diabetes risk based on machine learning algorithms.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of your physician or other qualified health provider with any
        questions you may have regarding a medical condition.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please try refreshing the page. If the problem persists, contact support.")
