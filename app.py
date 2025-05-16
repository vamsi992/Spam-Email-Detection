import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Load dataset
data = pd.read_csv("email-spam/emails.csv")  # Update with the correct path to your CSV file
print(data.head())

# Prepare data
x = data['text']
y = data['spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Initialize algorithms
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Store accuracy results
accuracy_results = {}

# Train and evaluate each model
svm_pipeline = None  # To store the SVM pipeline
for model_name, model in models.items():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy
    
    # Save the SVM pipeline if it's the current model
    if model_name == "SVM":
        svm_pipeline = pipeline  # Store the SVM pipeline

    print(f"{model_name} Accuracy: {accuracy:.2f}")

# Save SVM pipeline as a .pkl file
if svm_pipeline:
    joblib.dump(svm_pipeline, 'svm_email_classifier.pkl')
    print("SVM model saved as 'svm_email_classifier.pkl'")

# Plotting accuracy
plt.figure(figsize=(10, 6))
plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['blue', 'orange', 'green', 'red'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Algorithms')
plt.ylim(0, 1)
plt.show()

# Test with sample emails
emails = [
    'the stock trading gunslinger  fanny is merrill but muzo not colza attainder and penultimate like esmark perspicuous ramble is segovia not group try slung kansas tanzania yes chameleon or continuant clothesman no  libretto is chesapeake but tight not waterway herald and hawthorn like chisel morristown superior is deoxyribonucleic not clockwork try hall incredible mcdougall yes hepburn or einsteinian earmark no  sapling is boar but duane not plain palfrey and inflexible like huzzah pepperoni bedtime is nameable not attire try edt chronography optima yes pirogue or diffusion albeit no ',
    'alp presentation  on behalf of enron corp . i would like to invite you to an alp project  presentation by a group of students  of jesse h . jones graduate school of management , rice university .  the students will present the results of a research project regarding  electronic trading  platforms in the energy industry .  the presentation will be held on may 7 , at 4 : 00 p . m . at enron , 1400 smith .  we would also like to invite you to dinner , following the presentation .  vince kaminski  vincent kaminski  managing director - research  enron corp .  1400 smith street  room ebl 962  houston , tx 77002 - 7361  phone : ( 713 ) 853 3848  ( 713 ) 410 5396 ( cell )  fax : ( 713 ) 646 2503  e - mail : vkamins @ enron . com'
]

results_summary = {}
for model_name, model in models.items():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions = pipeline.predict(emails)
    accuracy = accuracy_score(y_test, y_pred)
    
    results_summary[model_name] = {
        "Accuracy": accuracy,
        "Predictions": predictions
    }

# Display results
for model_name, result in results_summary.items():
    print(f"{model_name}:\n  Accuracy: {result['Accuracy']:.2f}\n  Predictions: {result['Predictions']}")
