# Importaciones necesarias
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Función para limpiar el texto
def clean_text(text):
    text = re.sub('\W+', ' ', text)
    text = text.lower()
    return text

# Cargar y preparar los datos
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Resume_str": "Resume"}, errors="raise")
    df["Cleaned Resume"] = df["Resume"].apply(lambda x: clean_text(x))
    encoder = LabelEncoder()
    df['Labels'] = encoder.fit_transform(df['Category'])
    return df, encoder

# Vectorización
def vectorize_text(df):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    X = word_vectorizer.fit_transform(df['Cleaned Resume'])
    y = df['Labels']
    return X, y, word_vectorizer

# Entrenar modelos
def train_models(X_train, y_train):
    models = {
        "MultinomialNB": MultinomialNB(),
        "KNeighborsClassifier": KNeighborsClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model
    return models

# Predicciones
def make_prediction(models, vectorizer, text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    predictions = {name: model.predict(vectorized_text)[0] for name, model in models.items()}
    return predictions

# Ejemplo de uso
if __name__ == "__main__":
    # Asumiendo que el CSV está en la misma ubicación con el nombre 'resume_dataset.csv'
    df, encoder = load_and_prepare_data('../Resume/Resume.csv')
    X, y, word_vectorizer = vectorize_text(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)

    # Prueba de predicción con texto de ejemplo
    sample_text = "Experience driving planes in diferent airlines"
    predictions = make_prediction(models, word_vectorizer, sample_text)
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {encoder.inverse_transform([prediction])[0]}")
