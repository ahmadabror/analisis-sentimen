# =========================
# streamlit_app.py (FINAL)
# =========================

import os
import re
import pickle

import numpy as np
import streamlit as st
import emoji
import nltk

import gensim
from gensim import corpora

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# =========================================================
# 0) ENV (harus di atas import transformers)
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Kunci untuk menghindari error Keras 3 pada Transformers:
# - Jangan izinkan Transformers menggunakan TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Cache HuggingFace di folder writable Streamlit Cloud
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

# Import transformers pipeline secara aman (kalau gagal, app tetap hidup)
try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = str(e)
else:
    TRANSFORMERS_IMPORT_ERROR = ""


# =========================================================
# 1) NLTK (Streamlit Cloud safe): download punkt + punkt_tab
# =========================================================
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)

def ensure_nltk():
    # NLTK 3.9.x sering butuh punkt_tab juga
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab/english")
        return
    except LookupError:
        pass

    nltk.download("punkt", download_dir=NLTK_DIR, quiet=True, raise_on_error=False)
    nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True, raise_on_error=False)

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError as e:
        st.error(
            "NLTK tokenizer belum tersedia dan gagal diunduh pada environment deploy.\n\n"
            "Solusi:\n"
            "- Redeploy\n"
            "- Pastikan environment mengizinkan download NLTK\n\n"
            f"Detail: {e}"
        )
        st.stop()

ensure_nltk()


# =========================================================
# 2) STOPWORDS & STEMMER
# =========================================================
STOPWORD_PATH = "stopwordbahasa.txt"

additional_stopwords = []
if os.path.exists(STOPWORD_PATH):
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        additional_stopwords = [line.strip() for line in f if line.strip()]
else:
    st.warning(f"Warning: {STOPWORD_PATH} not found. Continuing without additional stopwords.")

stop_factory = StopWordRemoverFactory()
more_stopword = ["dengan", "ia", "bahwa", "oleh", "nya", "dana"]

stop_words = set(stop_factory.get_stop_words())
stop_words.update(more_stopword)
stop_words.update(additional_stopwords)

stemmer = StemmerFactory().create_stemmer()


# =========================================================
# 3) NORMALIZATION DICT
# =========================================================
normalization_dict = {
    "ae": "saja", "aja": "saja", "ajah": "saja", "aj": "saja", "jha": "saja", "sj": "saja",
    "g": "tidak", "ga": "tidak", "gak": "tidak", "gk": "tidak", "kaga": "tidak", "kagak": "tidak",
    "kg": "tidak", "ngga": "tidak", "nggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "lgi": "lagi", "lg": "lagi", "donlod": "download", "pdhl": "padahal", "pdhal": "padahal",
    "tpi": "tapi", "tp": "tapi",
    "gliran": "giliran", "kl": "kalau", "klo": "kalau", "gatau": "tidak tau", "bgt": "banget",
    "hrs": "harus", "dll": "dan lain-lain", "dsb": "dan sebagainya", "trs": "terus", "trus": "terus",
    "sangan": "sangat", "bs": "bisa", "bsa": "bisa", "gabisa": "tidak bisa", "gbsa": "tidak bisa",
    "gada": "tidak ada", "gaada": "tidak ada", "gausah": "tidak usah", "bkn": "bukan",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "ribet": "ruwet", "ribed": "ruwet", "sdangkan": "sedangkan", "lemot": "lambat",
    "ngelag": "gangguan", "yg": "yang", "dipakek": "di pakai", "pake": "pakai",
    "kya": "seperti", "kyk": "seperti", "ngurus": "mengurus", "jls": "jelas",
    "burik": "buruk", "payah": "buruk", "krna": "karena", "dr": "dari", "smpe": "sampai",
    "slalu": "selalu", "mulu": "melulu", "d": "di", "konek": "terhubung", "suruh": "disuruh",
    "apk": "aplikasi", "app": "aplikasi", "apps": "aplikasi", "apl": "aplikasi",
    "bapuk": "jelek", "bukak": "buka",
    "uodate": "update", "ato": "atau", "onlen": "online", "cmn": "cuman", "jele": "jelek",
    "angel": "susah", "jg": "juga", "knp": "kenapa", "hbis": "setelah", "ny": "nya",
    "skck": "skck", "stnk": "stnk", "sim": "sim", "sp2hp": "sp2hp", "propam": "propam", "dumas": "dumas",
    "tilang": "tilang", "e-tilang": "tilang", "etilang": "tilang", "surat kehilangan": "kehilangan",
}


# =========================================================
# 4) PREPROCESSING (FIX UTAMA re.sub)
# =========================================================
def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def preprocess_text(text: str) -> str:
    text = str(text)
    text = normalize_repeated_characters(text)

    text = emoji.demojize(text)
    text = re.sub(r":[a-z_]+:", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\@\w+|#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)

    text = text.lower()

    for slang, standard in normalization_dict.items():
        text = re.sub(rf"\b{re.escape(slang.lower())}\b", standard.lower(), text)

    # ✅ FIX: harus ada argumen ke-3 (string)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_lda(text: str) -> str:
    text = stemmer.stem(text)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess_single_text(text: str) -> str:
    cleaned = preprocess_text(text)
    return preprocess_text_lda(cleaned)


# =========================================================
# 5) TOPIC MAPS
# =========================================================
topic_name_map_lda = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat",
}


# =========================================================
# 6) LOAD MODELS (robust)
# =========================================================
maxlen = 20
maxlen_sentiment = 20

def _exists_all(paths):
    return all(os.path.exists(p) for p in paths)

@st.cache_resource
def load_all_models():
    # ---------- LDA ----------
    lda_model = None
    dictionary = None

    # Gensim LDA bisa butuh file pendamping .npy
    lda_files_must = [
        "lda_model.gensim",
        "lda_model.gensim.expElogbeta.npy",
        "lda_dictionary.gensim",
    ]

    if _exists_all(lda_files_must):
        try:
            lda_model = gensim.models.LdaMulticore.load("lda_model.gensim")
            dictionary = corpora.Dictionary.load("lda_dictionary.gensim")
        except Exception as e:
            lda_model = None
            dictionary = None
            st.warning(f"LDA gagal diload, fitur LDA dimatikan. Detail: {e}")
    else:
        st.warning(
            "File LDA tidak lengkap. Fitur LDA dimatikan.\n"
            "Pastikan file berikut ada di repo:\n- " + "\n- ".join(lda_files_must)
        )

    # ---------- IndoBERT Hugging Face (PyTorch only) ----------
    indobert_pipe = None
    HF_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"

    if pipeline is None:
        st.warning(
            "Transformers/pipeline gagal di-import. IndoBERT dimatikan.\n"
            f"Detail import error: {TRANSFORMERS_IMPORT_ERROR}"
        )
    else:
        try:
            indobert_pipe = pipeline(
                "sentiment-analysis",
                model=HF_MODEL_ID,
                tokenizer=HF_MODEL_ID,
                framework="pt",
                device=-1,
            )
        except Exception as e:
            indobert_pipe = None
            st.warning(f"Gagal load IndoBERT dari Hugging Face. IndoBERT dimatikan. Detail: {e}")

    # ---------- LSTM Topic ----------
    lstm_topic_model = load_model("lstm_topic_model.h5")
    with open("tokenizer_topic.pkl", "rb") as f:
        tokenizer_topic = pickle.load(f)
    with open("label_encoder_topic.pkl", "rb") as f:
        label_encoder_topic = pickle.load(f)

    # ---------- LSTM Sentiment ----------
    lstm_sentiment_model = load_model("lstm_sentiment_model.h5")
    with open("tokenizer_sentiment.pkl", "rb") as f:
        tokenizer_sentiment = pickle.load(f)
    with open("label_encoder_sentiment.pkl", "rb") as f:
        label_encoder_sentiment = pickle.load(f)

    return (
        lda_model, dictionary,
        indobert_pipe,
        lstm_topic_model, tokenizer_topic, label_encoder_topic,
        lstm_sentiment_model, tokenizer_sentiment, label_encoder_sentiment,
    )

(
    lda_model_loaded, dictionary_loaded,
    indobert_sentiment_pipeline_loaded,
    lstm_topic_model_loaded, tokenizer_topic_loaded, label_encoder_topic_loaded,
    lstm_sentiment_model_loaded, tokenizer_sentiment_loaded, label_encoder_sentiment_loaded,
) = load_all_models()


# =========================================================
# 7) PREDICTIONS
# =========================================================
def predict_topic_lda(preprocessed_text_lda: str):
    if lda_model_loaded is None or dictionary_loaded is None:
        return -1, "LDA dimatikan (file tidak lengkap / gagal load)"
    if not preprocessed_text_lda.strip():
        return -1, "No content to classify"
    bow = dictionary_loaded.doc2bow(preprocessed_text_lda.split())
    topic_distribution = lda_model_loaded.get_document_topics(bow)
    if not topic_distribution:
        return -1, "No topic found"
    dominant_topic_id = max(topic_distribution, key=lambda x: x[1])[0]
    dominant_topic_name = topic_name_map_lda.get(dominant_topic_id, "Unknown Topic")
    return dominant_topic_id, dominant_topic_name

def predict_topic_lstm(preprocessed_text_lda: str):
    if not preprocessed_text_lda.strip():
        return -1, "No content to classify"
    seq = tokenizer_topic_loaded.texts_to_sequences([preprocessed_text_lda])
    pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    preds = lstm_topic_model_loaded.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    name = label_encoder_topic_loaded.inverse_transform([idx])[0]
    return idx, name

def predict_sentiment_lstm(preprocessed_text_lda: str):
    if not preprocessed_text_lda.strip():
        return "neutral"
    seq = tokenizer_sentiment_loaded.texts_to_sequences([preprocessed_text_lda])
    pad = pad_sequences(seq, maxlen=maxlen_sentiment, padding="post", truncating="post")
    preds = lstm_sentiment_model_loaded.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    name = label_encoder_sentiment_loaded.inverse_transform([idx])[0]
    return name

def predict_sentiment_indobert(cleaned_text: str):
    if not cleaned_text.strip():
        return "neutral"
    if indobert_sentiment_pipeline_loaded is None:
        return "neutral"

    out = indobert_sentiment_pipeline_loaded(cleaned_text)[0]
    label = out.get("label", "")

    mapping = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
    if label in mapping:
        return mapping[label]

    low = label.lower()
    if "pos" in low:
        return "positive"
    if "neg" in low:
        return "negative"
    return "neutral"


# =========================================================
# 8) UI
# =========================================================
st.title("Aplikasi Analisis Sentimen dan Topik Ulasan Pengguna")
st.write("Masukkan ulasan pengguna aplikasi Polri Presisi untuk menganalisis topik dan sentimennya.")

user_input = st.text_area("Masukkan ulasan Anda di sini:", "")

if st.button("Analisis Ulasan"):
    if not user_input.strip():
        st.warning("Silakan masukkan ulasan terlebih dahulu.")
        st.stop()

    lda_ready_text = preprocess_single_text(user_input)
    indobert_ready_text = preprocess_text(user_input)

    st.subheader("Hasil Analisis:")

    # Topic & LSTM sentiment
    if not lda_ready_text.strip():
        st.warning("Ulasan setelah preprocessing menjadi kosong. Tidak dapat menganalisis topik dan LSTM sentimen.")
    else:
        lda_topic_id, lda_topic_name = predict_topic_lda(lda_ready_text)
        st.write(f"**LDA Topik:** {lda_topic_name} (ID: {lda_topic_id})")

        lstm_topic_id, lstm_topic_name = predict_topic_lstm(lda_ready_text)
        st.write(f"**LSTM Topik:** {lstm_topic_name} (ID: {lstm_topic_id})")

        lstm_sent = predict_sentiment_lstm(lda_ready_text)
        st.write(f"**LSTM Sentimen:** {lstm_sent}")

    # IndoBERT
    if not indobert_ready_text.strip():
        st.warning("Ulasan setelah preprocessing untuk IndoBERT menjadi kosong. Tidak dapat menganalisis sentimen dengan IndoBERT.")
    else:
        if indobert_sentiment_pipeline_loaded is None:
            st.info("IndoBERT (Hugging Face) tidak tersedia. Menampilkan hasil LSTM saja.")
        else:
            indobert_sent = predict_sentiment_indobert(indobert_ready_text)
            st.write(f"**IndoBERT Sentimen:** {indobert_sent}")
