import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
file_path = 'sightcom-2-prompt-classifier-data.csv'
data = pd.read_csv(file_path)

# Preprocess the text data and split the data
X_train, X_test, y_train, y_test = train_test_split(data['Query'], data['Category'], test_size=0.2, random_state=42)

# Build and train the text classification model
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

print('Accuracy:', accuracy)
print('Classification Report:\n', classification_report_str)

# Save the model
model_file_path = 'language_classifier.joblib'
joblib.dump(model, model_file_path)
