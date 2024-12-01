from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from data_preprocessing import load_data, preprocess_data, vectorize_data, split_data
from utils import evaluate_model
import os

# Load and preprocess the data
data = load_data('Sentiment Analysis of Amazon Product Reviews/data/amazon_reviews.csv')
data = preprocess_data(data)
X, vectorizer = vectorize_data(data)
X_train, X_test, y_train, y_test = split_data(data, X)

# Train and evaluate the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
evaluate_model(y_test, nb_predictions, 'Naive_Bayes')
# joblib.dump(nb_model, '../models/saved_models/Naive_Bayes.pkl')
# Ensure the 'models/saved_models/' directory exists
model_dir = '../models/saved_models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the Naive Bayes model
joblib.dump(nb_model, f'{model_dir}/Naive_Bayes.pkl')

# Train and evaluate the Logistic Regression model
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
evaluate_model(y_test, lr_predictions, 'Logistic_Regression')
joblib.dump(lr_model, '../models/saved_models/Logistic_Regression.pkl')

# Train and evaluate the Support Vector Machines model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
evaluate_model(y_test, svm_predictions, 'SVM')
joblib.dump(svm_model, '../models/saved_models/SVM.pkl')

# Train and evaluate the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
evaluate_model(y_test, rf_predictions, 'Random_Forest')
joblib.dump(rf_model, '../models/saved_models/Random_Forest.pkl')
