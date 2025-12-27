import os
import re
import json
import pickle
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import emoji
import nltk
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.metrics import confusion_matrix, accuracy_score


# =========================================================
# ENV safe for deploy
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"  # cegah transformers pakai TF/Keras 3
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

try:
    from transformers import pipeline
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception as e:
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = str(e)

HF_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"

NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)


# =========================================================
# NLTK ensure (punkt + punkt_tab)
# =========================================================
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
        return False, str(e)


# =========================================================
# Stopwords, stemmer, normalization
# =========================================================
STOPWORD_PATH = "stopwordbahasa.txt"
LDA_COHERENCE_PATH = "lda_coherence.json"  # optional, recommended

stop_factory = StopWordRemoverFactory()
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

more_stopword = ["dengan", "ia", "bahwa", "oleh", "nya", "dana"]


def build_stopwords() -> set:
    additional = []
    if os.path.exists(STOPWORD_PATH):
        with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
            additional = [line.strip() for line in f if line.strip()]
    sw = set(stop_factory.get_stop_words())
    sw.update(more_stopword)
    sw.update(additional)
    return sw


# =========================================================
# Preprocess
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

    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_lda(cleaned_text: str, stop_words: set) -> str:
    t = stemmer.stem(cleaned_text)
    tokens = nltk.tokenize.word_tokenize(t)
    tokens = [x for x in tokens if x not in stop_words and len(x) > 2]
    return " ".join(tokens)

def tokenize_for_coherence(cleaned_text: str, stop_words: set) -> List[str]:
    t = stemmer.stem(cleaned_text)
    tokens = nltk.tokenize.word_tokenize(t)
    tokens = [x for x in tokens if x not in stop_words and len(x) > 2]
    return tokens


# =========================================================
# LDA topic naming
# =========================================================
topic_name_map_lda = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat",
}


# =========================================================
# Coherence
# =========================================================
def load_saved_lda_coherence() -> Optional[dict]:
    if os.path.exists(LDA_COHERENCE_PATH):
        try:
            with open(LDA_COHERENCE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def compute_lda_coherence_from_tokens(lda_model, dictionary, token_texts: List[List[str]]) -> Optional[float]:
    if lda_model is None or dictionary is None:
        return None
    if not token_texts or len(token_texts) < 5:
        return None
    try:
        cm = CoherenceModel(model=lda_model, texts=token_texts, dictionary=dictionary, coherence="c_v")
        return float(cm.get_coherence())
    except Exception:
        return None


# =========================================================
# NRS per topik
# =========================================================
def sentiment_bucket(label: str) -> str:
    s = str(label).lower()
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    return "neutral"

def nrs_percent(pos, neg, total):
    return ((pos - neg) / total * 100) if total else 0.0

def build_topic_nrs_table(df: pd.DataFrame, topic_col="Topik LDA", sent_col="Sentimen IndoBERT") -> pd.DataFrame:
    tmp = df.copy()
    tmp[sent_col] = tmp[sent_col].fillna("neutral").astype(str)
    tmp["_sent"] = tmp[sent_col].apply(sentiment_bucket)

    grp = tmp.groupby(topic_col)["_sent"].value_counts().unstack(fill_value=0)
    for c in ["positive", "neutral", "negative"]:
        if c not in grp.columns:
            grp[c] = 0

    grp["total"] = grp["positive"] + grp["neutral"] + grp["negative"]
    grp["NRS_%"] = grp.apply(lambda r: nrs_percent(r["positive"], r["negative"], r["total"]), axis=1)
    grp = grp.reset_index().sort_values("NRS_%", ascending=False)
    grp["Rank"] = range(1, len(grp) + 1)

    # format friendly
    grp["NRS_%"] = grp["NRS_%"].map(lambda x: round(float(x), 2))
    return grp[["Rank", topic_col, "positive", "neutral", "negative", "total", "NRS_%"]]


# =========================================================
# Resources (lazy, cached)
# =========================================================
MAXLEN_TOPIC = 20
MAXLEN_SENTIMENT = 20

def load_pickled(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_resources(load_indobert: bool):
    ok, msg = ensure_nltk()
    if not ok:
        raise RuntimeError(f"NLTK belum siap: {msg}")

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

    # LSTM required for evaluation
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
# Predictors for output (LDA + IndoBERT)
# =========================================================
def lda_topic_from_text(text_lda: str, lda_model, dictionary) -> str:
    if lda_model is None or dictionary is None:
        return "LDA dimatikan (file tidak lengkap / gagal load)"
    if not text_lda.strip():
        return "No content"
    bow = dictionary.doc2bow(text_lda.split())
    dist = lda_model.get_document_topics(bow)
    if not dist:
        return "No topic found"
    tid = max(dist, key=lambda x: x[1])[0]
    return topic_name_map_lda.get(tid, "Unknown Topic")

def indobert_sentiment(cleaned_text: str, indobert_pipe) -> Optional[str]:
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


# =========================================================
# LSTM evaluation only
# =========================================================
def evaluate_lstm(df: pd.DataFrame, text_col: str, true_topic_col: str, true_sent_col: str, res: Dict) -> Dict:
    stop_words = res["stop_words"]

    texts = df[text_col].fillna("").astype(str).tolist()
    texts_lda = [preprocess_text_lda(preprocess_text(t), stop_words) for t in texts]

    # topic pred
    seq_topic = res["tokenizer_topic"].texts_to_sequences(texts_lda)
    pad_topic = pad_sequences(seq_topic, maxlen=MAXLEN_TOPIC, padding="post", truncating="post")
    pred_topic_probs = res["lstm_topic_model"].predict(pad_topic, verbose=0)
    pred_topic_idx = np.argmax(pred_topic_probs, axis=1)
    pred_topic_lbl = res["label_encoder_topic"].inverse_transform(pred_topic_idx)

    # sentiment pred
    seq_sent = res["tokenizer_sentiment"].texts_to_sequences(texts_lda)
    pad_sent = pad_sequences(seq_sent, maxlen=MAXLEN_SENTIMENT, padding="post", truncating="post")
    pred_sent_probs = res["lstm_sentiment_model"].predict(pad_sent, verbose=0)
    pred_sent_idx = np.argmax(pred_sent_probs, axis=1)
    pred_sent_lbl = res["label_encoder_sentiment"].inverse_transform(pred_sent_idx)

    y_topic_true = df[true_topic_col].astype(str).values
    y_sent_true = df[true_sent_col].astype(str).values

    acc_topic = float(accuracy_score(y_topic_true, pred_topic_lbl))
    acc_sent = float(accuracy_score(y_sent_true, pred_sent_lbl))

    topic_labels = sorted(list(set(y_topic_true) | set(pred_topic_lbl)))
    sent_labels = sorted(list(set(y_sent_true) | set(pred_sent_lbl)))

    cm_topic = confusion_matrix(y_topic_true, pred_topic_lbl, labels=topic_labels)
    cm_sent = confusion_matrix(y_sent_true, pred_sent_lbl, labels=sent_labels)

    return {
        "acc_topic": acc_topic,
        "acc_sent": acc_sent,
        "topic_labels": topic_labels,
        "sent_labels": sent_labels,
        "cm_topic": cm_topic,
        "cm_sent": cm_sent,
    }


# =========================================================
# OUTPUT COMPACT (tanpa CSS)
# =========================================================
def render_result_compact(topik_lda: str, sent_ib: str, nrs_value, notes: str = ""):
    st.subheader("Hasil")

    # Baris ringkas (tanpa metric)
    c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
    with c1:
        st.caption("Topik LDA")
        st.write(f"**{topik_lda}**")
    with c2:
        st.caption("Sentimen IndoBERT")
        st.write(f"**{sent_ib}**")
    with c3:
        st.caption("Net Reputable Score")
        st.write(f"**{nrs_value}**")

    if notes:
        with st.expander("Catatan", expanded=False):
            st.write(notes)


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Analisis Topik & Sentimen", layout="wide")

st.title("Dashboard Analisis: LDA + IndoBERT + NRS (Per Topik) & Evaluasi LSTM")
st.caption(
    "Tab Upload: Topik LDA, Sentimen IndoBERT, Koherensi, Ranking NRS per topik. "
    "Tab Evaluasi: Confusion Matrix & Akurasi LSTM."
    "Ahmad Abror 2043221003."
)

tab_upload, tab_eval, tab_diag = st.tabs(["📄 Upload & Output", "🧪 Evaluasi LSTM", "🛠 Diagnostik"])


with tab_diag:
    st.subheader("Diagnostik")
    required = [
        "lstm_topic_model.h5",
        "tokenizer_topic.pkl",
        "label_encoder_topic.pkl",
        "lstm_sentiment_model.h5",
        "tokenizer_sentiment.pkl",
        "label_encoder_sentiment.pkl",
    ]
    optional = [
        "stopwordbahasa.txt",
        "lda_model.gensim",
        "lda_model.gensim.expElogbeta.npy",
        "lda_dictionary.gensim",
        "lda_coherence.json",
    ]

    colA, colB = st.columns(2)
    with colA:
        st.caption("File Wajib")
        for f in required:
            st.write(("✅" if os.path.exists(f) else "❌"), f)
    with colB:
        st.caption("File Opsional")
        for f in optional:
            st.write(("✅" if os.path.exists(f) else "⚠️"), f)

    st.divider()
    st.caption("Transformers Import (IndoBERT)")
    if pipeline is None:
        st.warning("Transformers pipeline tidak bisa diimport → IndoBERT akan mati.")
        st.code(TRANSFORMERS_IMPORT_ERROR)
    else:
        st.success("Transformers pipeline import OK.")

    ok, msg = ensure_nltk()
    st.caption("NLTK")
    if ok:
        st.success("NLTK punkt + punkt_tab OK.")
    else:
        st.error("NLTK belum siap.")
        st.code(msg)


with tab_upload:
    st.subheader("Upload Data (CSV/Excel) → Output LDA + IndoBERT + NRS per Topik")

    left, right = st.columns([2, 1], gap="large")
    with right:
        use_indobert = st.checkbox("Gunakan IndoBERT", value=True)
        max_rows = st.number_input("Maks baris (hindari timeout)", 1, 1000000, 500)
    with left:
        uploaded = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded:
        # Load
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Pilih sheet:", xls.sheet_names)
            df = pd.read_excel(uploaded, sheet_name=sheet)

        st.caption("Preview data")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        text_col = st.selectbox("Pilih kolom teks ulasan:", df.columns)

        if st.button("Generate Output", type="primary"):
            work = df.copy().head(int(max_rows))
            work[text_col] = work[text_col].fillna("").astype(str)

            with st.spinner("Loading resources..."):
                res = get_resources(load_indobert=use_indobert)

            out_rows = []
            token_texts = []
            sw = res["stop_words"]

            with st.spinner("Memproses baris..."):
                for review in work[text_col].tolist():
                    cleaned = preprocess_text(review)
                    text_lda = preprocess_text_lda(cleaned, sw)

                    topik_lda = lda_topic_from_text(text_lda, res["lda_model"], res["dictionary"])
                    sent_ib = indobert_sentiment(cleaned, res["indobert_pipe"]) if use_indobert else None
                    if sent_ib is None:
                        sent_ib = "neutral"

                    toks = tokenize_for_coherence(cleaned, sw)
                    if toks:
                        token_texts.append(toks)

                    out_rows.append({
                        "Ulasan": review,
                        "Topik LDA": topik_lda,
                        "Sentimen IndoBERT": sent_ib,
                    })

            out_df = pd.DataFrame(out_rows)

            # NRS per baris (score)
            out_df["Score"] = out_df["Sentimen IndoBERT"].apply(
                lambda s: 1 if "pos" in str(s).lower() else (-1 if "neg" in str(s).lower() else 0)
            )

            st.divider()

            # ---- COHERENCE ----
            st.subheader("Koherensi LDA")
            saved = load_saved_lda_coherence()
            if saved is not None:
                st.caption("Koherensi dibaca dari file lda_coherence.json (recommended).")
                st.json(saved)
            else:
                coh = compute_lda_coherence_from_tokens(res["lda_model"], res["dictionary"], token_texts[:200])
                if coh is None:
                    st.warning("Koherensi tidak bisa dihitung (model/dictionary tidak tersedia atau token kurang).")
                    st.caption("Saran: simpan koherensi dari notebook training ke lda_coherence.json.")
                else:
                    st.write(f"**Coherence (c_v, approx dari sample upload): {coh:.4f}**")

            st.divider()

            # ---- PREVIEW OUTPUT (COMPACT TABLE) ----
            st.subheader("Output per Ulasan (Ringkas)")
            default_cols = ["Ulasan", "Topik LDA", "Sentimen IndoBERT", "Score"]
            show_cols = st.multiselect(
                "Pilih kolom yang ingin ditampilkan",
                options=list(out_df.columns),
                default=default_cols
            )
            st.dataframe(out_df[show_cols].head(50), use_container_width=True, hide_index=True)

            st.divider()

            # ---- NRS PER TOPIK ----
            st.subheader("Ranking NRS per Topik")
            topic_nrs = build_topic_nrs_table(out_df, topic_col="Topik LDA", sent_col="Sentimen IndoBERT")
            st.dataframe(topic_nrs, use_container_width=True, hide_index=True)

            st.divider()

            # ---- QUICK SUMMARY (tanpa metric besar) ----
            st.subheader("Ringkasan Cepat")
            total = len(out_df)
            pos = int((out_df["Score"] == 1).sum())
            neu = int((out_df["Score"] == 0).sum())
            neg = int((out_df["Score"] == -1).sum())
            net_percent = ((pos - neg) / total * 100) if total else 0.0

            a, b = st.columns(2)
            with a:
                st.caption("Jumlah Ulasan")
                st.write(f"**{total}**")
                st.caption("Positif / Netral / Negatif")
                st.write(f"**{pos} / {neu} / {neg}**")
            with b:
                st.caption("Net Reputable Score (global)")
                st.write(f"**{net_percent:.2f}%**")
                st.caption("Rumus: (Positif - Negatif) / Total × 100")

            # download
            st.download_button(
                "Download Output Per Ulasan (CSV)",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="output_ulasan_lda_indobert.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Ranking NRS per Topik (CSV)",
                data=topic_nrs.to_csv(index=False).encode("utf-8"),
                file_name="ranking_nrs_per_topik.csv",
                mime="text/csv",
            )


with tab_eval:
    st.subheader("Evaluasi LSTM (Confusion Matrix & Akurasi) — Tanpa Prediksi di Output Utama")
    st.caption(
        "Upload file yang memiliki label ground-truth.\n"
        "Pilih kolom teks, kolom label topik, dan kolom label sentimen."
    )

    uploaded2 = st.file_uploader("Upload file evaluasi (CSV/Excel)", type=["csv", "xlsx"], key="eval_uploader")
    if uploaded2:
        if uploaded2.name.lower().endswith(".csv"):
            df2 = pd.read_csv(uploaded2)
        else:
            xls2 = pd.ExcelFile(uploaded2)
            sheet2 = st.selectbox("Pilih sheet evaluasi:", xls2.sheet_names)
            df2 = pd.read_excel(uploaded2, sheet_name=sheet2)

        st.caption("Preview data evaluasi")
        st.dataframe(df2.head(10), use_container_width=True, hide_index=True)

        text_col2 = st.selectbox("Kolom teks ulasan:", df2.columns, key="tcol_eval")
        true_topic_col = st.selectbox("Kolom label topik (ground truth):", df2.columns, key="topic_true")
        true_sent_col = st.selectbox("Kolom label sentimen (ground truth):", df2.columns, key="sent_true")
        max_rows_eval = st.number_input("Maks baris evaluasi", 1, 1000000, min(2000, len(df2)))

        if st.button("Jalankan Evaluasi", type="primary", key="run_eval"):
            eval_df = df2.copy().head(int(max_rows_eval))
            eval_df = eval_df.dropna(subset=[true_topic_col, true_sent_col])

            if len(eval_df) == 0:
                st.error("Data evaluasi kosong (label true banyak yang NaN).")
            else:
                with st.spinner("Loading model LSTM..."):
                    res = get_resources(load_indobert=False)

                with st.spinner("Menghitung evaluasi..."):
                    ev = evaluate_lstm(eval_df, text_col2, true_topic_col, true_sent_col, res)

                # output compact (tanpa metric besar)
                st.subheader("Hasil Evaluasi (Ringkas)")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Akurasi Topik (LSTM)")
                    st.write(f"**{ev['acc_topic']:.4f}**")
                with col2:
                    st.caption("Akurasi Sentimen (LSTM)")
                    st.write(f"**{ev['acc_sent']:.4f}**")

                st.divider()

                st.subheader("Confusion Matrix — Topik")
                cm_topic_df = pd.DataFrame(ev["cm_topic"], index=ev["topic_labels"], columns=ev["topic_labels"])
                st.dataframe(cm_topic_df, use_container_width=True)

                st.subheader("Confusion Matrix — Sentimen")
                cm_sent_df = pd.DataFrame(ev["cm_sent"], index=ev["sent_labels"], columns=ev["sent_labels"])
                st.dataframe(cm_sent_df, use_container_width=True)

                st.download_button(
                    "Download CM Topik (CSV)",
                    data=cm_topic_df.to_csv().encode("utf-8"),
                    file_name="cm_topik.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download CM Sentimen (CSV)",
                    data=cm_sent_df.to_csv().encode("utf-8"),
                    file_name="cm_sentimen.csv",
                    mime="text/csv",
                )
