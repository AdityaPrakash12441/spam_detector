import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import lime
import lime.lime_text
import streamlit.components.v1 as components

# Load the TF-IDF vectorizer
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load all models
models = {
    'Naive Bayes': pickle.load(open('model.pkl', 'rb')),
    'Logistic Regression': None,
    'Random Forest': None,
    'SVM': None
}

# Load other models
models['Logistic Regression'] = pickle.load(open('logistic_regression_model.pkl', 'rb'))
models['Random Forest'] = pickle.load(open('random_forest_model.pkl', 'rb'))
models['SVM'] = pickle.load(open('svm_model.pkl', 'rb'))

def preprocess_text(text):
    """
    Preprocesses the text data.
    """
    stop_words = set(stopwords.words('english'))
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)

def predict_spam(text, model):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    confidence = model.predict_proba(vectorized_text).max()
    return ("Spam" if prediction[0] == 1 else "Safe"), confidence

# Streamlit app
st.title("SMS Spam Detector")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Dashboard
st.header("Performance Dashboard")
col1, col2, col3 = st.columns(3)

# Calculate metrics
if st.session_state.history:
    correct_predictions = sum(1 for item in st.session_state.history if item['correct'])
    total_predictions = len(st.session_state.history)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    true_positives = sum(1 for item in st.session_state.history if item['prediction'] == 'Spam' and item['correct'])
    false_positives = sum(1 for item in st.session_state.history if item['prediction'] == 'Spam' and not item['correct'])
    
    actual_spam = sum(1 for item in st.session_state.history if (item['prediction'] == 'Spam' and item['correct']) or (item['prediction'] == 'Safe' and not item['correct']))
    
    spam_detection_rate = true_positives / actual_spam if actual_spam > 0 else 0.0
    
    actual_ham = sum(1 for item in st.session_state.history if (item['prediction'] == 'Safe' and item['correct']) or (item['prediction'] == 'Spam' and not item['correct']))
    
    false_positive_rate = false_positives / actual_ham if actual_ham > 0 else 0.0
    
else:
    accuracy = 0.0
    spam_detection_rate = 0.0
    false_positive_rate = 0.0

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Spam Detection Rate", f"{spam_detection_rate:.2%}")
col3.metric("False Positive Rate", f"{false_positive_rate:.2%}")

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

st.header("Single Message Prediction")
st.write("Enter a message to check if it's spam or safe.")

message = st.text_area("Message")

if st.button("Predict"):
    if message:
        prediction, confidence = predict_spam(message, model)
        st.session_state.last_prediction = {'message': message, 'prediction': prediction, 'confidence': confidence}
        st.rerun()

if st.session_state.last_prediction:
    prediction = st.session_state.last_prediction['prediction']
    confidence = st.session_state.last_prediction['confidence']
    
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    # LIME Explanation
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Safe', 'Spam'])
    
    # Create a prediction function for LIME
    def lime_predict(texts):
        processed_texts = [preprocess_text(text) for text in texts]
        vectorized_texts = tfidf.transform(processed_texts).toarray()
        return model.predict_proba(vectorized_texts)

    explanation = explainer.explain_instance(message, lime_predict, num_features=10)
    
    st.subheader("Explanation")
    components.html(explanation.as_html(), height=800)
    
    if prediction == 'Spam':
        if confidence > 0.9:
            st.error("High-confidence spam")
        elif confidence > 0.7:
            st.warning("Likely spam")
        elif confidence > 0.3:
            st.info("Potential spam")
    else:
        st.success("Legitimate message")

    st.write("Was this prediction correct?")
    col1, col2 = st.columns(2)
    if col1.button("Yes"):
        st.session_state.history.append({'message': st.session_state.last_prediction['message'], 'prediction': st.session_state.last_prediction['prediction'], 'correct': True})
        st.session_state.last_prediction = None
        st.success("Thank you for your feedback!")
        st.rerun()
    if col2.button("No"):
        st.session_state.history.append({'message': st.session_state.last_prediction['message'], 'prediction': st.session_state.last_prediction['prediction'], 'correct': False})
        
        # Save feedback to a CSV file
        feedback_df = pd.DataFrame([st.session_state.last_prediction])
        feedback_df['correct_label'] = 'Spam' if st.session_state.last_prediction['prediction'] == 'Safe' else 'Safe'
        
        try:
            existing_feedback = pd.read_csv('feedback.csv')
            feedback_df = pd.concat([existing_feedback, feedback_df], ignore_index=True)
        except FileNotFoundError:
            pass
            
        feedback_df.to_csv('feedback.csv', index=False)
        
        st.session_state.last_prediction = None
        st.error("Thank you for your feedback! The model will be improved.")
        st.rerun()

st.header("Batch Processing")
uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'message' in df.columns:
            predictions = df['message'].apply(lambda x: predict_spam(x, model))
            df['prediction'] = [p[0] for p in predictions]
            df['confidence'] = [p[1] for p in predictions]
            
            st.write("Batch Predictions:")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='batch_predictions.csv',
                mime='text/csv',
            )
        else:
            st.error("The CSV file must have a 'message' column.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
