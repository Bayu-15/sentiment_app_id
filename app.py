
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text

stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    tokens = clean_text(text).split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_data():
    df = pd.read_csv("data/data_sentimen_tiktok.csv")
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text_clean'] = df['text'].apply(preprocess_text)
    return df

def train_model(df):
    X = df['text_clean']
    y = df['label']
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='positif'),
        'recall': recall_score(y_test, y_pred, pos_label='positif'),
        'f1': f1_score(y_test, y_pred, pos_label='positif'),
        'specificity': recall_score(y_test, y_pred, pos_label='negatif'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    with open("model/naive_bayes_model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    return metrics

def load_model():
    with open("model/naive_bayes_model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

st.title("📊 Aplikasi Klasifikasi Sentimen Relokasi IKN - Naïve Bayes")

menu = st.sidebar.selectbox("Pilih Menu", ["📁 Data & Training", "🧪 Evaluasi Model", "📝 Klasifikasi Teks"])

if menu == "📁 Data & Training":
    st.header("Dataset Komentar TikTok")
    df = load_data()
    st.dataframe(df[['text', 'label']].head())

    if st.button("🔁 Latih Model"):
        metrics = train_model(df)
        st.success("Model berhasil dilatih dan disimpan.")
        st.write("Akurasi:", round(metrics['accuracy'], 2))
        st.write("Presisi:", round(metrics['precision'], 2))
        st.write("Recall:", round(metrics['recall'], 2))
        st.write("F1 Score:", round(metrics['f1'], 2))
        st.write("Specificity:", round(metrics['specificity'], 2))
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

elif menu == "🧪 Evaluasi Model":
    st.header("Evaluasi Model")
    try:
        model, vectorizer = load_model()
        df = load_data()
        X = vectorizer.transform(df['text_clean'])
        y = df['label']
        y_pred = model.predict(X)
        st.text("Laporan Klasifikasi:")
        st.text(classification_report(y, y_pred))
    except FileNotFoundError:
        st.error("Model belum dilatih. Silakan latih terlebih dahulu.")

elif menu == "📝 Klasifikasi Teks":
    st.header("Klasifikasi Sentimen Komentar TikTok")
    input_text = st.text_area("Masukkan komentar TikTok:")
    if st.button("🔍 Klasifikasi"):
        if not input_text:
            st.warning("Tolong masukkan teks terlebih dahulu.")
        else:
            try:
                model, vectorizer = load_model()
                cleaned = preprocess_text(input_text)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                st.success(f"Prediksi Sentimen: **{prediction.upper()}**")
            except FileNotFoundError:
                st.error("Model belum dilatih. Silakan ke menu 'Data & Training'.")
