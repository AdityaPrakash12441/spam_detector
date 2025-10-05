import requests
import zipfile
import io
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

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

def download_and_load_data():
    """
    Downloads and loads the SMS Spam Collection dataset.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    # The zip file contains a single file named 'SMSSpamCollection'
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='	', header=None, names=['label', 'message'])
    
    return df

if __name__ == '__main__':
    # Fix for SSL issue with NLTK downloader
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download stopwords if not already downloaded
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    df = download_and_load_data()
    
    # Preprocess the messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    print("Dataset loaded and preprocessed successfully:")
    print(df.head())
    
    # Feature Engineering
    X = df['processed_message']
    y = df['label']
    
    # Convert labels to binary
    y = y.map({'ham': 0, 'spam': 1})
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(X).toarray()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nData split into training and testing sets:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    # Model Training and Saving
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        # Save the model
        with open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    # Save the TF-IDF vectorizer
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    print("\nModels and TF-IDF vectorizer saved to disk.")
    
    # Test the prediction function with the Naive Bayes model
    nb_model = models['Naive Bayes']
    def predict_spam(text, model):
        processed_text = preprocess_text(text)
        vectorized_text = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_text)
        return "Spam" if prediction[0] == 1 else "Safe"

    test_message = "Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize."
    print(f"\nTest message: '{test_message}'")
    print(f"Prediction: {predict_spam(test_message, nb_model)}")
    
    test_message = "Hey, are we still on for dinner tonight?"
    print(f"\nTest message: '{test_message}'")
    print(f"Prediction: {predict_spam(test_message, nb_model)}")
    
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    df['message_len'] = df['message'].apply(len)
    print("\nMessage length statistics:")
    print(df.groupby('label')['message_len'].describe())
