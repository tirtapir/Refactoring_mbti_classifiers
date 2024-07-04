import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Contoh fungsi untuk memuat dataset (Anda perlu menyesuaikan sesuai dengan format dataset Anda)
def load_dataset(file_path):
    # Misalkan dataset adalah file CSV dengan kolom 'text' dan 'label'
    data = pd.read_csv(file_path)
    return data['Indikator'], data['Tipe']

# Daftar model yang akan diuji
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Naive Bayes', MultinomialNB()),
    ('Support Vector Machine', SVC())
]

# Pipeline untuk preprocessing dan model
def create_pipeline(model):
    return Pipeline([
        ('tfidf', TfidfVectorizer()),  # Preprocessing teks dengan TF-IDF
        ('model', model)               # Model yang akan diuji
    ])

# Hyperparameter grid untuk grid search
param_grid = [
    {
        'model': [LogisticRegression()],
        'model__C': [0.1, 1, 10]
    },
    {
        'model': [MultinomialNB()],
        'model__alpha': [0.01, 0.1, 1]
    },
    {
        'model': [SVC()],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }
]

# Daftar file path dataset
dataset_files = ['dataset1.csv', 'dataset2.csv', ..., 'dataset12.csv']

# Menyimpan hasil model terbaik untuk setiap dataset
best_models = {}

for file_path in dataset_files:
    # Memuat dataset
    texts, labels = load_dataset(file_path)
    
    # Membagi data menjadi training dan test set
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
    
    # Melakukan grid search dengan cross-validation
    grid_search = GridSearchCV(create_pipeline(LogisticRegression()), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Menyimpan model terbaik
    best_model = grid_search.best_estimator_
    best_models[file_path] = best_model
    
    # Evaluasi model terbaik pada test set
    y_pred = best_model.predict(X_test)
    print(f"Dataset: {file_path}")
    print(classification_report(y_test, y_pred))
    print(f"Best Model: {grid_search.best_params_}")
    print("\n")

