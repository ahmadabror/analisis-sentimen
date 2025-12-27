# =========================
# streamlit_app.py (FULL)
# =========================

import os
import re
import pickle

import numpy as np
import pandas as pd
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
# 0) ENV SETUP (penting untuk deploy Streamlit Cloud)
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Hindari Transformers menyentuh TensorFlow/Keras 3
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Cache HuggingFace di folder writable
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

# Import transformers pipeline dengan aman (kalau gagal, app tetap jalan)
try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = str(e)
else:
    TRANSFORMERS_IMPORT_ERROR = ""


# =========================================================
# 1) NLTK SETUP (download punkt + punkt_tab)
# =========================================================
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)

def ensure_nltk():
    """
    NLTK 3.9.x pada beberapa environment butuh `punkt_tab` (tokenizers/punkt_tab/english).
    Jadi kita pastikan download `punkt` dan `punkt_tab`.
    """
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
# 4) PREPROCESSING
# =========================================================
def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def preprocess_text(text: str) -> str:
    """
    Cleaning untuk IndoBERT (tanpa stemming/stopwords).
    """
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

    # FIX UTAMA: wajib ada argumen 'text' pada re.sub
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_lda(text: str) -> str:
    """
    Untuk LDA/LSTM: stemming + tokenisasi + stopwords removal.
    """
    text = stemmer.stem(text)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess_single_text(text: str) -> str:
    cleaned = preprocess_text(text)
    return preprocess_text_lda(cleaned)


# =========================================================
# 5) TOPIC MAPS (ubah sesuai label kamu)
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
MAXLEN_TOPIC = 20
MAXLEN_SENTIMENT = 20

def _exists_all(paths):
    return all(os.path.exists(p) for p in paths)

@st.cache_resource
def load_all_models():
    # ---- LDA (optional) ----
    lda_model = None
    dictionary = None
    lda_files_must = [
        "lda_model.gensim",
        "lda_model.gensim.expElogbeta.npy",  # sering hilang, bikin crash kalau tidak ada
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

    # ---- IndoBERT Hugging Face (optional) ----
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
                framework="pt",  # paksa PyTorch
                device=-1,       # CPU
            )
        except Exception as e:
            indobert_pipe = None
            st.warning(f"Gagal load IndoBERT dari Hugging Face. IndoBERT dimatikan. Detail: {e}")

    # ---- LSTM models (required) ----
    lstm_topic_model = load_model("lstm_topic_model.h5")
    with open("tokenizer_topic.pkl", "rb") as f:
        tokenizer_topic = pickle.load(f)
    with open("label_encoder_topic.pkl", "rb") as f:
        label_encoder_topic = pickle.load(f)

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
# 7) PREDICTION HELPERS
# =========================================================
def predict_topic_lda(preprocessed_text_lda: str):
    if lda_model_loaded is None or dictionary_loaded is None:
        return -1, "LDA dimatikan (file tidak lengkap / gagal load)"
    if not preprocessed_text_lda.strip():
        return -1, "No content to classify"
    bow = dictionary_loaded.doc2bow(preprocessed_text_lda.split())
    dist = lda_model_loaded.get_document_topics(bow)
    if not dist:
        return -1, "No topic found"
    topic_id = max(dist, key=lambda x: x[1])[0]
    topic_name = topic_name_map_lda.get(topic_id, "Unknown Topic")
    return topic_id, topic_name

def predict_topic_lstm(preprocessed_text_lda: str):
    if not preprocessed_text_lda.strip():
        return -1, "No content to classify"
    seq = tokenizer_topic_loaded.texts_to_sequences([preprocessed_text_lda])
    pad = pad_sequences(seq, maxlen=MAXLEN_TOPIC, padding="post", truncating="post")
    preds = lstm_topic_model_loaded.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    name = label_encoder_topic_loaded.inverse_transform([idx])[0]
    return idx, name

def predict_sentiment_lstm(preprocessed_text_lda: str):
    if not preprocessed_text_lda.strip():
        return "neutral"
    seq = tokenizer_sentiment_loaded.texts_to_sequences([preprocessed_text_lda])
    pad = pad_sequences(seq, maxlen=MAXLEN_SENTIMENT, padding="post", truncating="post")
    preds = lstm_sentiment_model_loaded.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    name = label_encoder_sentiment_loaded.inverse_transform([idx])[0]
    return name

def predict_sentiment_indobert(cleaned_text: str):
    """
    Output: 'positive' / 'neutral' / 'negative'
    """
    if not cleaned_text.strip():
        return "neutral"
    if indobert_sentiment_pipeline_loaded is None:
        return "neutral"

    out = indobert_sentiment_pipeline_loaded(cleaned_text)[0]
    label = out.get("label", "")

    # mapping umum model 3 kelas
    mapping = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
    if label in mapping:
        return mapping[label]

    # fallback
    low = str(label).lower()
    if "pos" in low:
        return "positive"
    if "neg" in low:
        return "negative"
    return "neutral"


# =========================================================
# 8) NET REPUTABLE SCORE
# =========================================================
def sentiment_to_score(label: str) -> int:
    """
    +1 positive, 0 neutral, -1 negative
    """
    if not label:
        return 0
    low = str(label).lower()
    if "pos" in low:
        return 1
    if "neg" in low:
        return -1
    return 0


# =========================================================
# 9) ANALYZE SINGLE TEXT (untuk input teks & batch file)
# =========================================================
def analyze_review(review: str, use_indobert: bool = True) -> dict:
    review = "" if review is None else str(review)

    lda_ready = preprocess_single_text(review)   # untuk LDA/LSTM
    indobert_ready = preprocess_text(review)     # untuk IndoBERT

    lda_topic = None
    lstm_topic = None
    lstm_sent = None
    indobert_sent = None
    notes = ""

    # LDA + LSTM
    if lda_ready.strip():
        _, lda_topic = predict_topic_lda(lda_ready)
        _, lstm_topic = predict_topic_lstm(lda_ready)
        lstm_sent = predict_sentiment_lstm(lda_ready)
    else:
        notes += "Preprocess LDA/LSTM kosong. "

    # IndoBERT
    if use_indobert:
        if indobert_ready.strip():
            if indobert_sentiment_pipeline_loaded is None:
                notes += "IndoBERT tidak tersedia. "
            else:
                indobert_sent = predict_sentiment_indobert(indobert_ready)
        else:
            notes += "Preprocess IndoBERT kosong. "
    else:
        notes += "IndoBERT dimatikan oleh user. "

    # Net Reputable Score:
    # pakai IndoBERT jika ada, fallback ke LSTM sentiment
    base_sent = indobert_sent if indobert_sent is not None else lstm_sent
    net_score = sentiment_to_score(base_sent)

    return {
        "Topik LDA": lda_topic,
        "Sentimen IndoBERT": indobert_sent,
        "Net Reputable Score": net_score,
        "Topik LSTM": lstm_topic,
        "Sentimen LSTM": lstm_sent,
        "Notes": notes.strip(),
    }


# =========================================================
# 10) UI
# =========================================================
st.title("Aplikasi Analisis Sentimen dan Topik Ulasan Pengguna")
st.write("Hasil yang ditampilkan: **Topik LDA**, **Sentimen IndoBERT**, **Net Reputable Score**, dan **hasil model LSTM**.")

mode = st.radio("Pilih metode input:", ["Input Teks", "Upload File (CSV/Excel)"], horizontal=True)

# ---------- Mode 1: Input Teks ----------
if mode == "Input Teks":
    text = st.text_area("Masukkan ulasan Anda di sini:", "")
    use_indobert = st.checkbox("Gunakan IndoBERT (butuh transformers+torch & akses model)", value=True)

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Silakan masukkan ulasan terlebih dahulu.")
        else:
            out = analyze_review(text, use_indobert=use_indobert)

            st.subheader("Hasil")
            st.write(f"**Topik LDA:** {out['Topik LDA']}")
            st.write(f"**Sentimen IndoBERT:** {out['Sentimen IndoBERT']}")
            st.write(f"**Net Reputable Score:** {out['Net Reputable Score']}")
            st.write(f"**Topik LSTM:** {out['Topik LSTM']}")
            st.write(f"**Sentimen LSTM:** {out['Sentimen LSTM']}")

            if out["Notes"]:
                st.caption(f"Catatan: {out['Notes']}")

# ---------- Mode 2: Upload File ----------
else:
    st.subheader("Upload CSV / Excel untuk Analisis Batch")

    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded:
        # Load file
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Pilih sheet:", xls.sheet_names)
            df = pd.read_excel(uploaded, sheet_name=sheet)

        st.write("Preview:")
        st.dataframe(df.head())

        # pilih kolom ulasan
        col = st.selectbox("Pilih kolom yang berisi ulasan:", df.columns)

        # kontrol performa
        max_rows = st.number_input(
            "Maksimal baris dianalisis (hindari timeout):",
            min_value=1,
            max_value=len(df),
            value=min(500, len(df)),
        )

        use_indobert = st.checkbox("Gunakan IndoBERT (lebih berat)", value=False)

        if st.button("Jalankan Analisis Batch"):
            work = df.copy().head(int(max_rows))
            work[col] = work[col].fillna("").astype(str)

            results = []
            with st.spinner("Menganalisis..."):
                for review in work[col].tolist():
                    results.append(analyze_review(review, use_indobert=use_indobert))

            res_df = pd.DataFrame(results)
            out_df = work.reset_index(drop=True).copy()

            # kolom output sesuai permintaan
            out_df["Topik LDA"] = res_df["Topik LDA"]
            out_df["Sentimen IndoBERT"] = res_df["Sentimen IndoBERT"]
            out_df["Net Reputable Score"] = res_df["Net Reputable Score"]
            out_df["Topik LSTM"] = res_df["Topik LSTM"]
            out_df["Sentimen LSTM"] = res_df["Sentimen LSTM"]
            out_df["Notes"] = res_df["Notes"]

            # ringkasan global reputasi
            total = len(out_df)
            pos = int((out_df["Net Reputable Score"] == 1).sum())
            neg = int((out_df["Net Reputable Score"] == -1).sum())
            neu = int((out_df["Net Reputable Score"] == 0).sum())
            net_percent = ((pos - neg) / total * 100) if total else 0.0

            st.subheader("Ringkasan")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", total)
            c2.metric("Positif", pos)
            c3.metric("Netral", neu)
            c4.metric("Negatif", neg)
            st.write(f"**Net Reputable Score (global)** = (Positif - Negatif) / Total × 100 = **{net_percent:.2f}%**")

            st.subheader("Hasil (preview)")
            st.dataframe(out_df.head(30))

            # download CSV hasil
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil (CSV)",
                data=csv_bytes,
                file_name="hasil_analisis.csv",
                mime="text/csv",
            )
