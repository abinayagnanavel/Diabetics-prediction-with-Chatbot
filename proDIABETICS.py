import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Prediction Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
        background-color: #F8F9FA;
        color:black;4    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #FFE6E6;
        color: #D32F2F;
        border: 2px solid #D32F2F;
    }
    .low-risk {
        background-color: #E8F5E8;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    .moderate-risk {
        background-color: #FFF3E0;
        color: #F57C00;
        border: 2px solid #F57C00;
    }
</style>
""", unsafe_allow_html=True)

# ✅ GLOBAL function with caching
@st.cache_resource
def load_models():
    model = joblib.load("best_diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("feature_selector.pkl")
    with open("selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f.readlines()]
    return model, scaler, selector, selected_features


class DiabetesChatbot:
    def __init__(self):
        try:
            self.model, self.scaler, self.selector, self.selected_features = load_models()
            self.model_loaded = True
        except FileNotFoundError:
            st.error("⚠️ Model files not found. Please train the model first.")
            self.model_loaded = False
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False

    def get_user_input_form(self):
        st.sidebar.header("📊 Health Information")
        age = st.sidebar.selectbox("Age Group", [
            "18-24", "25-29", "30-34", "35-39", "40-44",
            "45-49", "50-54", "55-59", "60-64", "65-69",
            "70-74", "75-79", "80+"
        ])
        sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
        bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0, 0.1)
        general_health = st.sidebar.selectbox("General Health", [
            "Excellent", "Very Good", "Good", "Fair", "Poor"
        ])
        high_bp = st.sidebar.checkbox("High Blood Pressure")
        high_chol = st.sidebar.checkbox("High Cholesterol")
        chol_check = st.sidebar.checkbox("Cholesterol Check (in past 5 years)")
        stroke = st.sidebar.checkbox("History of Stroke")
        heart_disease = st.sidebar.checkbox("Heart Disease or Heart Attack")
        smoker = st.sidebar.checkbox("Smoker")
        heavy_alcohol = st.sidebar.checkbox("Heavy Alcohol Consumption")
        physical_activity = st.sidebar.checkbox("Physical Activity (past 30 days)")
        fruits = st.sidebar.checkbox("Consume Fruits Daily")
        veggies = st.sidebar.checkbox("Consume Vegetables Daily")
        healthcare_access = st.sidebar.checkbox("Healthcare Access")
        no_doc_cost = st.sidebar.checkbox("Couldn't see doctor due to cost")
        mental_health = st.sidebar.slider("Mental Health (bad days)", 0, 30, 0)
        physical_health = st.sidebar.slider("Physical Health (bad days)", 0, 30, 0)
        diff_walk = st.sidebar.checkbox("Difficulty Walking")

        data = self.convert_inputs_to_model_format(
            age, sex, bmi, general_health, high_bp, high_chol, chol_check,
            stroke, heart_disease, smoker, heavy_alcohol, physical_activity,
            fruits, veggies, healthcare_access, no_doc_cost, mental_health,
            physical_health, diff_walk
        )
        return data

    def convert_inputs_to_model_format(self, age, sex, bmi, general_health, 
                                       high_bp, high_chol, chol_check, stroke,
                                       heart_disease, smoker, heavy_alcohol,
                                       physical_activity, fruits, veggies,
                                       healthcare_access, no_doc_cost,
                                       mental_health, physical_health, diff_walk):
        age_map = {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
            "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
            "70-74": 11, "75-79": 12, "80+": 13
        }
        health_map = {
            "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
        }
        d = {
            'HighBP': int(high_bp),
            'HighChol': int(high_chol),
            'CholCheck': int(chol_check),
            'BMI': bmi,
            'Smoker': int(smoker),
            'Stroke': int(stroke),
            'HeartDiseaseorAttack': int(heart_disease),
            'PhysActivity': int(physical_activity),
            'Fruits': int(fruits),
            'Veggies': int(veggies),
            'HvyAlcoholConsump': int(heavy_alcohol),
            'AnyHealthcare': int(healthcare_access),
            'NoDocbcCost': int(no_doc_cost),
            'GenHlth': health_map[general_health],
            'MentHlth': mental_health,
            'PhysHlth': physical_health,
            'DiffWalk': int(diff_walk),
            'Sex': int(sex == "Male"),
            'Age': age_map[age],
            'Education': 6,
            'Income': 8,
            'BMI_category': self.get_bmi_category(bmi),
            'Age_BMI_interaction': age_map[age]*bmi,
            'Health_Risk_Score': (
                int(high_bp)*2 + int(high_chol)*2 + bmi/10 + int(smoker) + 
                int(heart_disease)*3 + health_map[general_health] + age_map[age]/10
            ),
            'PhysActivity_Health': int(physical_activity)*(6 - health_map[general_health]),
            'Lifestyle_Risk': (
                int(smoker) + int(heavy_alcohol) + (1 - int(physical_activity)) + 
                (1 - int(fruits)) + (1 - int(veggies))
            ),
            'Age_Group': self.get_age_group(age_map[age]),
            'Mental_Physical_Health': mental_health + physical_health,
            'Sex_Age_interaction': int(sex == "Male") * age_map[age],
            'Cardio_Risk': (
                int(high_bp) + int(high_chol) + int(heart_disease) + int(stroke)
            ),
            'Healthcare_Risk': (
                (1 - int(healthcare_access)) + int(no_doc_cost) + (1 - int(chol_check))
            )
        }
        return d

    def get_bmi_category(self, bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        else:
            return 3

    def get_age_group(self, age):
        if age <= 3:
            return 0
        elif age <= 7:
            return 1
        elif age <= 11:
            return 2
        else:
            return 3

    def make_prediction(self, user_data):
        if not self.model_loaded:
            return None, None, "Model not loaded"
        df = pd.DataFrame([user_data])[self.selected_features]
        try:
            X_scaled = self.scaler.transform(df)
            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0,1]
            return pred, prob, "Success"
        except Exception as e:
            return None, None, str(e)

    def get_risk_level(self, prob):
        if prob < 0.3:
            return "Low Risk", "low-risk"
        elif prob < 0.7:
            return "Moderate Risk", "moderate-risk"
        else:
            return "High Risk", "high-risk"

    def create_risk_visualization(self, prob):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Diabetes Risk (%)"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'darkblue'},
                'steps':[
                    {'range':[0,30],'color':'lightgreen'},
                    {'range':[30,70],'color':'yellow'},
                    {'range':[70,100],'color':'red'}
                ],
                'threshold':{
                    'line':{'color':'red','width':4},
                    'value':70
                }
            }
        ))
        fig.update_layout(height=300)
        return fig

    def run_chatbot(self):
        st.markdown("<h1 class='main-header'>🩺 Diabetes Risk Prediction Chatbot</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("""
            <div class='chat-message'>
                <h3>👋 Hello! I'm your Diabetes Risk Assistant</h3>
                <p>Fill the sidebar and click Analyze My Risk.</p>
            </div>""", unsafe_allow_html=True)
            user_data = self.get_user_input_form()
            if st.button("🔍 Analyze My Risk"):
                pred, prob, status = self.make_prediction(user_data)
                if status == "Success":
                    risk_level, risk_class = self.get_risk_level(prob)
                    st.markdown(
                        f"<div class='prediction-box {risk_class}'>Risk Level: {risk_level}<br>Risk Score: {prob:.1%}</div>",
                        unsafe_allow_html=True
                    )
                    fig = self.create_risk_visualization(prob)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {status}")

# Run
tab1, tab2 = st.tabs(["🩺 Diabetes Prediction", "💬 Chatbot"])

with tab1:
    if __name__ == "__main__":
        chatbot = DiabetesChatbot()
        chatbot.run_chatbot()


import streamlit as st
import requests
import json
# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"

# Streamlit UI
#st.set_page_config(page_title="Chat with Me", page_icon="💬", layout="centered")
#st.title("💬 Chat with ME")

# Keep conversation history
with tab2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask something..."):
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare payload for Ollama
        payload = {
            "model": "mistral",  #  mistral is faster
            "messages": st.session_state.messages+[
            {
                "role": "system",
                "content": "You are a diabetes health assistant. Only answer questions related to diabetes risk, prevention, and lifestyle."
            }
        ],
            "options": {
                "num_predict": 100  # limit response length for speed
            }
        }

        # Call Ollama API
        response = requests.post(OLLAMA_URL, json=payload, stream=True)

        # Collect assistant reply
        full_reply = ""
        with st.chat_message("diabetics-assistant"):
            msg_placeholder = st.empty()
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    if data.startswith("{"):
                        content = json.loads(data).get("message", {}).get("content", "")
                        if content:
                            full_reply += content
                            msg_placeholder.markdown(full_reply + "▌")
            msg_placeholder.markdown(full_reply)

        # Save assistant reply
        st.session_state.messages.append({"role": "diabetics-assistant", "content": full_reply})


