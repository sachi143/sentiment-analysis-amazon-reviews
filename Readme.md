
---

# **Sentiment Analysis of Amazon Product Reviews**

This project performs sentiment analysis on Amazon product reviews, classifying reviews into **positive**, **neutral**, or **negative** sentiments using various machine learning models.

---

## **Project Structure**

```plaintext
.
├── data/                         # Dataset
│   └── amazon_reviews.csv         # Amazon product reviews dataset
│
├── notebooks/                    # Jupyter Notebooks (Optional)
│   └── sentiment_analysis.ipynb   # Interactive notebook for the analysis
│
├── scripts/                      # Python Scripts
│   ├── data_preprocessing.py      # Data loading & preprocessing
│   ├── model_training.py          # Model training & evaluation
│   └── utils.py                   # Utility functions (e.g., evaluation metrics)
│
├── models/                       # Saved models
│   └── saved_models/              # Serialized trained models (Optional)
│
├── outputs/                      # Confusion matrix & evaluation results
│   ├── confusion_matrix_Naive_Bayes.png
│   ├── confusion_matrix_Logistic_Regression.png
│   ├── confusion_matrix_SVM.png
│   └── confusion_matrix_Random_Forest.png
│
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies for the project
```

---

## **How to Run**

### **1. Clone the repository**

Open a terminal or command prompt and run the following commands to clone the repository:

```bash
git clone https://github.com/sachi143/sentiment-analysis-amazon-reviews.git
cd sentiment-analysis-amazon-reviews
```

---

### **2. Install dependencies**

Make sure you have Python 3.x installed. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

---

### **3. Run the sentiment analysis pipeline**

To execute the entire sentiment analysis pipeline, run:

```bash
python scripts/model_training.py
```

---

## **Output**

- **Classification reports** for all models will be displayed in the terminal.
- **Confusion matrices** for each model will be saved in the `outputs/` directory:
  - `confusion_matrix_Naive_Bayes.png`
  - `confusion_matrix_Logistic_Regression.png`
  - `confusion_matrix_SVM.png`
  - `confusion_matrix_Random_Forest.png`

---

## **Models Used**

1. **Naive Bayes Classifier**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**
4. **Random Forest Classifier**

---

## **Next Steps**

1. **Hyperparameter Tuning:** Optimize model parameters to improve performance.
2. **Try Advanced Vectorization:** Experiment with Word2Vec, GloVe, or BERT embeddings.
3. **Deploy the Model:** Use Flask or FastAPI to deploy the best-performing model as an API.
4. **Visualization:** Add more data visualizations for insights into the dataset and model performance.

---

## **Requirements**

List of dependencies:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

You can install these dependencies with:

```bash
pip install -r requirements.txt
```

---

## **License**

This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## **Contributing**

Contributions are welcome! If you find any bugs or want to add features, feel free to open an issue or submit a pull request.

---

## **Contact**

For any questions or issues, please contact:

**Sairam Chennaka**  
Email: [sachi777@outlook.in](mailto:sachi777@outlook.in)  
GitHub: [https://github.com/sachi143](https://github.com/sachi143)

---
