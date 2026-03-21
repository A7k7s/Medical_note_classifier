# Medical Note Classifier
# Installation: pip install streamlit scikit-learn pandas
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import io


# Custom CSS for Premium Look
def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .category-title {
            color: #007bff;
            font-size: 24px;
            font-weight: bold;
        }
        .confidence-score {
            font-size: 18px;
            color: #6c757d;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Data Generation ---
@st.cache_data
def generate_synthetic_data():
    """Generates a small synthetic dataset for medical note classification."""
    data = []
    
    # Respiratory Keywords
    respiratory_notes = [
        "Patient presents with persistent cough and shortness of breath.",
        "Diagnosis of acute bronchitis after showing signs of wheezing.",
        "Chronic asthma patient reporting difficulty breathing during exercise.",
        "Chest X-ray shows signs of pneumonia in the lower left lobe.",
        "Severe congestion and productive cough for three days.",
        "Symptoms include dry cough, mild fever, and respiratory distress.",
        "COPD exacerbation noted in a long-term smoker.",
        "Patient complains of tight chest and rapid shallow breathing.",
        "Evaluated for suspected pulmonary embolism after sudden dyspnea.",
        "Treatment plan involves nebulizer and inhaled corticosteroids."
    ] * 20 # 200 samples
    data.extend([(note, "Respiratory") for note in respiratory_notes])
    
    # Cardiovascular Keywords
    cardio_notes = [
        "Patient exhibits hypertension and elevated heart rate.",
        "Reporting chronic chest pain radiating to the left arm.",
        "History of myocardial infarction and coronary artery disease.",
        "Palpitations and dizziness reported by the patient during rest.",
        "Blood pressure readings consistently above 150/90 mmHg.",
        "EKG shows irregular heart rhythm suggestive of atrial fibrillation.",
        "Peripheral vascular disease diagnosed in the lower extremities.",
        "Congestive heart failure management includes diuretics.",
        "Suspected angina pectoris after exertion-induced chest tightness.",
        "Patient being monitored for heart valve replacement recovery."
    ] * 20 # 200 samples
    data.extend([(note, "Cardiovascular") for note in cardio_notes])
    
    # Neurological Keywords
    neuro_notes = [
        "Patient describes sharp migrainous headaches and light sensitivity.",
        "Frequent episodes of dizziness and loss of balance (vertigo).",
        "Signs of ischemic stroke including facial drooping and numbness.",
        "History of generalized seizures and tonic-clonic activity.",
        "Resting tremor and bradykinesia noted in physical exam.",
        "Nerve conduction study indicates peripheral neuropathy.",
        "Patient reports memory loss and confusion over several months.",
        "Evaluated for multiple sclerosis after reporting limb weakness.",
        "Sleep study suggests narcolepsy with cataplexy episodes.",
        "Encephalitis suspected following high fever and altered mental state."
    ] * 20 # 200 samples
    data.extend([(note, "Neurological") for note in neuro_notes])
    
    # Gastrointestinal Keywords
    gi_notes = [
        "Abdominal pain and bloating reported after meals.",
        "Chronic heartburn and acid reflux for several weeks.",
        "Symptoms include frequent nausea, vomiting, and diarrhea.",
        "Suspected gastric ulcer based on localized epigastric pain.",
        "Inflammatory bowel disease (IBD) flare-up with bloody stools.",
        "Liver function tests indicate possible hepatic dysfunction.",
        "Patient reports persistent constipation and weight loss.",
        "Endoscopy shows inflammation of the esophageal lining.",
        "Irritable bowel syndrome (IBS) symptoms worsening under stress.",
        "Gastroenteritis diagnosed following sudden onset of stomach cramps."
    ] * 20 # 200 samples
    data.extend([(note, "Gastrointestinal") for note in gi_notes])
    
    df = pd.DataFrame(data, columns=['note', 'category'])
    # Add some noise/unseen words
    df['note'] = df['note'].apply(lambda x: x + " The visit was routine.")
    return df

# --- Text Preprocessing ---
def preprocess_text(text):
    """Clean and tokenize text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# --- ML Model Functions ---
@st.cache_resource
def train_model(df):
    """Train the TF-IDF + Logistic Regression model."""
    tfidf = TfidfVectorizer(preprocessor=preprocess_text, max_features=1000, stop_words='english')
    X = tfidf.fit_transform(df['note'])
    y = df['category']
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)
    
    return tfidf, model

def classify_note(text, tfidf, model):
    """Predict category and confidence score for a given note."""
    if not text.strip():
        return None
    processed_text = preprocess_text(text)
    vectorized_text = tfidf.transform([processed_text])
    
    # Predict category and probabilities
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Get top 3
    cat_probs = list(zip(model.classes_, probabilities))
    cat_probs = sorted(cat_probs, key=lambda x: x[1], reverse=True)
    
    return {
        'prediction': prediction,
        'confidence': probabilities[list(model.classes_).index(prediction)],
        'top_3': cat_probs[:3]
    }

# --- Application Layout ---
def main():
    st.set_page_config(page_title="Medical Note Classifier", page_icon="🏥", layout="wide")
    apply_custom_css()
    st.title("🏥 Medical Note Classifier")
    st.markdown("---")
    
    # Initialize session state for the model
    if 'data_loaded' not in st.session_state:
        with st.spinner("Generating data and training model..."):
            df = generate_synthetic_data()
            st.session_state.tfidf, st.session_state.model = train_model(df)
            st.session_state.data_loaded = True

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("Classify clinical notes into disease categories using TF-IDF and Logistic Regression.")
        st.header("Categories")
        st.markdown("""
        - **Respiratory**: Lungs, cough, asthma, COPD
        - **Cardiovascular**: Heart, hypertension, arteries
        - **Neurological**: Brain, headaches, seizures
        - **Gastrointestinal**: Stomach, digestion, liver
        """)
        st.header("Settings")
        show_top_3 = st.checkbox("Show Top 3 Predictions", value=True)

    # Main Input Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Medical Note")
        note_input = st.text_area("Paste the clinical note here:", height=200, placeholder="e.g., Patient complains of worsening cough and short breath...")
        
        uploaded_file = st.file_uploader("Or upload a .txt file", type=['txt'])
        if uploaded_file is not None:
            note_input = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.success("File uploaded successfully!")

    with col2:
        st.subheader("Actions")
        classify_btn = st.button("Classify Note")
        clear_btn = st.button("Clear Input")
        
        if clear_btn:
            st.rerun()

    # Classification Logic
    if classify_btn:
        if not note_input.strip():
            st.warning("Please enter a medical note or upload a file.")
        else:
            result = classify_note(note_input, st.session_state.tfidf, st.session_state.model)
            
            if result:
                st.markdown("### Result")
                
                # Display Prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="category-title">{result['prediction']}</div>
                    <div class="confidence-score">Confidence Score: <b>{result['confidence']:.2%}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 Predictions
                if show_top_3:
                    st.subheader("Top 3 Predictions")
                    for cat, prob in result['top_3']:
                        st.write(f"**{cat}**: {prob:.2%}")
                        st.progress(prob)
                
                # Feature Importance (Optional/Simple)
                st.subheader("Word Analysis")
                processed = preprocess_text(note_input).split()
                important_words = []
                for word in processed:
                    if word in st.session_state.tfidf.vocabulary_:
                        important_words.append(word)
                
                if important_words:
                    st.write(f"Recognized terms: {', '.join(set(important_words))}")
                
                # Download Result
                result_text = f"Medical Note Classification Result\n"
                result_text += f"---------------------------------\n"
                result_text += f"Input Note: {note_input[:100]}...\n\n"
                result_text += f"Predicted Category: {result['prediction']}\n"
                result_text += f"Confidence Score: {result['confidence']:.2%}\n"
                
                st.download_button(
                    label="Download Prediction as .txt",
                    data=result_text,
                    file_name="medical_note_prediction.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
