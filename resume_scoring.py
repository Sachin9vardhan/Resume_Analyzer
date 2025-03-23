import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Function to clean and preprocess text."""
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Load dataset (Assume CSV with 'resume_text' and 'score' columns)
df = pd.read_csv('Synthetic_Resumes.csv')
df['cleaned_text'] = df['resume_text'].apply(preprocess_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['score']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model
import joblib
joblib.dump(model, 'resume_scoring_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict resume score
def predict_resume_score(resume_text):
    cleaned_text = preprocess_text(resume_text)
    features = vectorizer.transform([cleaned_text]).toarray()
    return model.predict(features)[0]

# Example usage
resume_example = "Experienced software engineer with Python, ML, and NLP skills."
score = predict_resume_score(resume_example)
print(f'Predicted Score: {score}')
