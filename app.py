import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import emoji
import contractions
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image

# تحميل الموارد الخاصة بـ NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# تحميل الملفات
model = load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# إعدادات
max_len = 1000
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# دالة تنظيف النصوص
def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ========== CSS Styling ==========

st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        background-image: linear-gradient(to right, #1e1e1e, #232323);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 45px;
        font-weight: bold;
        color: #00e676;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #aaa;
        margin-bottom: 40px;
    }
    .footer {
        margin-top: 60px;
        font-size: 12px;
        color: #888;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #00e676;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 2em;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Title ==========

st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter news text below and let the model decide if it\'s Real or Fake</div>', unsafe_allow_html=True)

# ========== Optional Logo ==========

# logo = Image.open("logo.png")
# st.image(logo, width=120)

# ========== Text Input ==========

headline = st.text_area("Enter Headline")
body = st.text_area("Enter Article Body")

# ========== Predict Button ==========

if st.button("Predict"):
    if not headline or not body:
        st.warning("Please enter both Headline and Article Body")
    else:
        text = clean_text(headline + " " + body)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        pred = model.predict(padded)
        label = label_encoder.inverse_transform([np.argmax(pred)])
        st.success(f"Predicted Stance: **{label[0].upper()}**")

# ========== Footer ==========

st.markdown('<div class="footer">Made with ❤️ by Menna Halim</div>', unsafe_allow_html=True)
