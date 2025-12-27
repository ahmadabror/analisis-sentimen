# =========================
# streamlit_app.py (UI FULL + LAZY LOADING)
# =========================

import os
import re
import pickle
from typing import Optional, Tuple, Dict

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
# 0) ENV (safe for deploy)
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"  # cegah Transformers pakai TF/Keras 3
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

# Safe import transformers pipeline (tidak boleh bikin app crash)
try:
    from transformers import pipeline
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception as e:
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = str(e)


# =========================================================
# 1) NLTK setup (punkt + punkt_tab)
# =========================================================
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)

def ensure_nltk() -> Tuple[bool, str]:
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab/english")
        return True, "OK"
    except LookupError:
        pass

    nltk.download("punkt", download_dir=NLTK_DIR, quiet=True, raise_on_error=False)
    nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True, raise_on_error=False)

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab/english")
        return True, "OK"
    except LookupError as e:
        return False, f"{e}"


# =========================================================
# 2) Text resources
# =========================================================
STOPWORD_PATH = "stopwordbahasa.txt"

stop_factory = StopWordRemoverFactory()
more_stopword = ["dengan", "ia", "bahwa", "oleh", "nya", "dana"]

stemmer = StemmerFactory().create_stemmer()

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
# 3) Preprocess
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

    # FIX: wajib ada argumen string
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_stopwords() -> set:
    additional_stopwords = []
    if os.path.exists(STOPWORD_PATH):
        with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
            additional_stopwords = [line.strip() for line in f if line.strip()]
    stop_words = set(stop_factory.get_stop_words())
    stop_words.update(more_stopword)
    stop_words.update(additional_stopwords)
    return stop_words

def preprocess_text_lda(text: str, stop_words: set) -> str:
    text = stemmer.stem(text)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess_single_text(text: str, stop_words: set) -> str:
    cleaned = preprocess_text(text)
    return preprocess_text_lda(cleaned, stop_words)


# =========================================================
# 4) Maps & scoring
# =========================================================
topic_name_map_lda = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat",
}

def sentiment_to_score(label: Optional[str]) -> int:
    if not label:
        return 0
    low = str(label).lower()
    if "pos" in low:
        return 1
    if "neg" in low:
        return -1
    return 0


# =========================================================
# 5) LAZY LOAD MODELS (dipanggil saat tombol klik)
# =========================================================
MAXLEN_TOPIC = 20
MAXLEN_SENTIMENT = 20
HF_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"

def load_pickled(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_resources(load_indobert: bool):
    # NLTK
    ok, msg = ensure_nltk()
    if not ok:
        raise RuntimeError(f"NLTK download gagal: {msg}")

    stop_words = build_stopwords()

    # LDA optional
    lda_model = None
    dictionary = None
    lda_files = ["lda_model.gensim", "lda_model.gensim.expElogbeta.npy", "lda_dictionary.gensim"]
    if all(os.path.exists(p) for p in lda_files):
        try:
            lda_model = gensim.models.LdaMulticore.load("lda_model.gensim")
            dictionary = corpora.Dictionary.load("lda_dictionary.gensim")
        except Exception:
            lda_model, dictionary = None, None

    # LSTM required
    lstm_topic_model = load_model("lstm_topic_model.h5")
    tokenizer_topic = load_pickled("tokenizer_topic.pkl")
    label_encoder_topic = load_pickled("label_encoder_topic.pkl")

    lstm_sentiment_model = load_model("lstm_sentiment_model.h5")
    tokenizer_sentiment = load_pickled("tokenizer_sentiment.pkl")
    label_encoder_sentiment = load_pickled("label_encoder_sentiment.pkl")

    # IndoBERT optional
    indobert_pipe = None
    indobert_error = ""
    if load_indobert:
        if pipeline is None:
            indobert_error = f"Transformers/pipeline gagal import: {TRANSFORMERS_IMPORT_ERROR}"
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
                indobert_error = str(e)
                indobert_pipe = None

    return {
        "stop_words": stop_words,
        "lda_model": lda_model,
        "dictionary": dictionary,
        "lstm_topic_model": lstm_topic_model,
        "tokenizer_topic": tokenizer_topic,
        "label_encoder_topic": label_encoder_topic,
        "lstm_sentiment_model": lstm_sentiment_model,
        "tokenizer_sentiment": tokenizer_sentiment,
        "label_encoder_sentiment": label_encoder_sentiment,
        "indobert_pipe": indobert_pipe,
        "indobert_error": indobert_error,
    }


# =========================================================
# 6) Predictors
# =========================================================
def predict_topic_lda(text_lda: str, lda_model, dictionary) -> str:
    if lda_model is None or dictionary is None:
        return "LDA dimatikan (file tidak lengkap / gagal load)"
    if not text_lda.strip():
        return "No content"
    bow = dictionary.doc2bow(text_lda.split())
    dist = lda_model.get_document_topics(bow)
    if not dist:
        return "No topic found"
    topic_id = max(dist, key=lambda x: x[1])[0]
    return topic_name_map_lda.get(topic_id, "Unknown Topic")

def predict_topic_lstm(text_lda: str, model, tokenizer, label_encoder) -> str:
    if not text_lda.strip():
        return "No content"
    seq = tokenizer.texts_to_sequences([text_lda])
    pad = pad_sequences(seq, maxlen=MAXLEN_TOPIC, padding="post", truncating="post")
    preds = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    return label_encoder.inverse_transform([idx])[0]

def predict_sentiment_lstm(text_lda: str, model, tokenizer, label_encoder) -> str:
    if not text_lda.strip():
        return "neutral"
    seq = tokenizer.texts_to_sequences([text_lda])
    pad = pad_sequences(seq, maxlen=MAXLEN_SENTIMENT, padding="post", truncating="post")
    preds = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(preds))
    return label_encoder.inverse_transform([idx])[0]

def predict_sentiment_indobert(cleaned_text: str, indobert_pipe) -> Optional[str]:
    if indobert_pipe is None:
        return None
    if not cleaned_text.strip():
        return "neutral"
    out = indobert_pipe(cleaned_text)[0]
    label = out.get("label", "")
    mapping = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
    if label in mapping:
        return mapping[label]
    low = str(label).lower()
    if "pos" in low:
        return "positive"
    if "neg" in low:
        return "negative"
    return "neutral"


def analyze_one(review: str, res: Dict, use_indobert: bool) -> Dict:
    stop_words = res["stop_words"]

    lda_ready = preprocess_single_text(review, stop_words)
    indobert_ready = preprocess_text(review)

    lda_topic = predict_topic_lda(lda_ready, res["lda_model"], res["dictionary"])
    lstm_topic = predict_topic_lstm(lda_ready, res["lstm_topic_model"], res["tokenizer_topic"], res["label_encoder_topic"])
    lstm_sent = predict_sentiment_lstm(lda_ready, res["lstm_sentiment_model"], res["tokenizer_sentiment"], res["label_encoder_sentiment"])

    indobert_sent = None
    notes = ""
    if use_indobert:
        indobert_sent = predict_sentiment_indobert(indobert_ready, res["indobert_pipe"])
        if indobert_sent is None and res["indobert_error"]:
            notes = f"IndoBERT off: {res['indobert_error']}"
    else:
        notes = "IndoBERT dimatikan oleh user."

    base_sent = indobert_sent if indobert_sent is not None else lstm_sent
    net_score = sentiment_to_score(base_sent)

    return {
        "Topik LDA": lda_topic,
        "Sentimen IndoBERT": indobert_sent,
        "Net Reputable Score": net_score,
        "Topik LSTM": lstm_topic,
        "Sentimen LSTM": lstm_sent,
        "Notes": notes,
    }


# =========================================================
# 7) UI FULL
# =========================================================
tab_out, tab_eval = st.tabs(["📊 Output Analisis (LDA+IndoBERT+NRS)", "🧪 Evaluasi LSTM"])

with tab_out:
    st.subheader("Output Analisis")

    # jalankan pred per baris: Topik LDA + Sentimen IndoBERT
    results = []
    with st.spinner("Menganalisis baris..."):
        for review in work[text_col].tolist():
            out = analyze_one(review, res, use_indobert=use_indobert_file)  # dari versi kamu
            # pastikan kolom yang dipakai:
            # out["Topik LDA"], out["Sentimen IndoBERT"]
            # hitung score baris dari IndoBERT
            base_sent = out["Sentimen IndoBERT"] if out["Sentimen IndoBERT"] is not None else "neutral"
            out["Score"] = sentiment_to_score(base_sent)
            results.append(out)

    res_df = pd.DataFrame(results)
    out_df = work.reset_index(drop=True).copy()
    out_df["Topik LDA"] = res_df["Topik LDA"]
    out_df["Sentimen IndoBERT"] = res_df["Sentimen IndoBERT"].fillna("neutral")
    out_df["Score"] = res_df["Score"]

    st.write("Preview hasil per ulasan:")
    st.dataframe(out_df.head(30))

    # NRS per topik + ranking
    st.subheader("NRS per Topik (Ranking)")
    topic_nrs = build_topic_nrs_table(out_df, topic_col="Topik LDA", sent_col="Sentimen IndoBERT")
    st.dataframe(topic_nrs)

    # tampilkan koherensi LDA
    st.subheader("Koherensi LDA")
    saved = load_saved_lda_coherence()
    if saved is not None:
        st.success("Koherensi dibaca dari lda_coherence.json (recommended).")
        st.write(saved)
    else:
        # coba approx coherence dari sample upload (tokenized)
        stop_words = res["stop_words"]
        texts_tokens = []
        for t in work[text_col].fillna("").astype(str).tolist()[:200]:
            cleaned = preprocess_text(t)
            tokens = nltk.tokenize.word_tokenize(stemmer.stem(cleaned))
            tokens = [x for x in tokens if x not in stop_words and len(x) > 2]
            if tokens:
                texts_tokens.append(tokens)

        coh = compute_lda_coherence_from_texts(res["lda_model"], res["dictionary"], texts_tokens)
        if coh is None:
            st.warning("Koherensi tidak bisa dihitung (butuh data lebih banyak atau model/dictionary tidak tersedia).")
        else:
            st.write(f"**Coherence (c_v, approx dari sample upload): {coh:.4f}**")

    # download hasil per ulasan
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Hasil Per Ulasan (CSV)", data=csv_bytes, file_name="hasil_ulasan.csv", mime="text/csv")

    # download ranking NRS per topik
    csv2 = topic_nrs.to_csv(index=False).encode("utf-8")
    st.download_button("Download Ranking NRS per Topik (CSV)", data=csv2, file_name="ranking_nrs_topik.csv", mime="text/csv")


with tab_eval:
    st.subheader("Evaluasi LSTM (Confusion Matrix & Accuracy)")
    st.info("Agar evaluasi berjalan, file harus punya kolom ground-truth: label_topik dan label_sentimen (atau pilih kolomnya di bawah).")

    # pilih kolom label
    true_topic_col = st.selectbox("Pilih kolom label topik (ground truth):", df.columns)
    true_sent_col = st.selectbox("Pilih kolom label sentimen (ground truth):", df.columns)

    if st.button("Jalankan Evaluasi LSTM"):
        eval_df = work.copy()
        # buang baris yang labelnya kosong
        eval_df = eval_df.dropna(subset=[true_topic_col, true_sent_col])
        if len(eval_df) == 0:
            st.error("Tidak ada data evaluasi (label kosong semua).")
        else:
            with st.spinner("Menghitung evaluasi LSTM..."):
                ev = evaluate_lstm(eval_df, text_col, true_topic_col, true_sent_col, res)

            c1, c2 = st.columns(2)
            c1.metric("Akurasi Topik (LSTM)", f"{ev['acc_topic']:.4f}")
            c2.metric("Akurasi Sentimen (LSTM)", f"{ev['acc_sent']:.4f}")

            st.markdown("### Confusion Matrix - Topik")
            cm_topic_df = pd.DataFrame(ev["cm_topic"], index=ev["topic_labels"], columns=ev["topic_labels"])
            st.dataframe(cm_topic_df)

            st.markdown("### Confusion Matrix - Sentimen")
            cm_sent_df = pd.DataFrame(ev["cm_sent"], index=ev["sent_labels"], columns=ev["sent_labels"])
            st.dataframe(cm_sent_df)

            # download evaluasi
            st.download_button(
                "Download Confusion Matrix Topik (CSV)",
                data=cm_topic_df.to_csv().encode("utf-8"),
                file_name="cm_topik.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Confusion Matrix Sentimen (CSV)",
                data=cm_sent_df.to_csv().encode("utf-8"),
                file_name="cm_sentimen.csv",
                mime="text/csv",
            )
