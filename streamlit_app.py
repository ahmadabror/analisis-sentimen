import streamlit as st
import gensim
from gensim import corpora
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np
import re
import emoji
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# =============================================================================
# 1. Global Variables Re-definition (from notebook)
# =============================================================================
STOPWORD_PATH = "stopwordbahasa.txt" # Assuming this file is in the same directory

additional_stopwords = []
if os.path.exists(STOPWORD_PATH):
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        additional_stopwords = [line.strip() for line in f.readlines()]
    additional_stopwords = [sw for sw in additional_stopwords if sw]  # remove empty
else:
    st.warning(f"Warning: {STOPWORD_PATH} not found. Continuing without additional stopwords.")

stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'nya', 'dana']

stop_words = set(stop_factory.get_stop_words())
stop_words.update(more_stopword)
stop_words.update(additional_stopwords)

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

normalization_dict = {
    'ae': 'saja','aja': 'saja','ajah': 'saja','aj': 'saja','jha': 'saja','sj': 'saja',
    'g': 'tidak','ga': 'tidak','gak': 'tidak','gk': 'tidak','kaga': 'tidak','kagak': 'tidak',
    'kg': 'tidak','ngga': 'tidak','Nggak': 'tidak','tdk': 'tidak','tak': 'tidak',
    'lgi': 'lagi','lg': 'lagi','donlod': 'download','pdhl': 'padahal','pdhal': 'padahal',
    'Coba2': 'coba-coba','tpi': 'tapi','tp': 'tapi','betmanfaat': 'bermanfaat',
    'gliran': 'giliran','kl': 'kalau','klo': 'kalau','gatau': 'tidak tau','bgt': 'banget',
    'hrs': 'harus','dll': 'dan lain-lain','dsb': 'dan sebagainya','trs': 'terus','trus': 'terus',
    'sangan': 'sangat','bs': 'bisa','bsa': 'bisa','gabisa': 'tidak bisa','gbsa': 'tidak bisa',
    'gada': 'tidak ada','gaada': 'tidak ada','gausah': 'tidak usah','bkn': 'bukan',
    'udh': 'sudah','udah': 'sudah','sdh': 'sudah','pertngahn': 'pertengahan',
    'ribet': 'ruwet','ribed': 'ruwet','sdangkan': 'sedangkan','lemot': 'lambat',
    'lag': 'lambat','ngelag': 'gangguan','yg': 'yang','dipakek': 'di pakai','pake': 'pakai',
    'kya': 'seperti','kyk': 'seperti','ngurus': 'mengurus','jls': 'jelas',
    'burik': 'buruk','payah':'buruk','krna': 'karena','dr': 'dari','smpe': 'sampai',
    'slalu': 'selalu','mulu': 'melulu','d': 'di','konek': 'terhubung','suruh': 'disuruh',
    'apk': 'aplikasi','app': 'aplikasi','apps': 'aplikasi','apl': 'aplikasi',
    'bapuk': 'jelek','bukak': 'buka','nyolong': 'mencuri','pas': 'ketika',
    'uodate': 'update','ato': 'atau','onlen': 'online','cmn': 'cuman','jele': 'jelek',
    'angel': 'susah','jg': 'juga','knp': 'kenapa','hbis': 'setelah','tololl': 'tolol','ny': 'nya',
    'skck':'skck','stnk':'stnk','sim':'sim','sp2hp':'sp2hp','propam':'propam','dumas':'dumas',
    'tilang':'tilang','e-tilang':'tilang','etilang':'tilang','surat kehilangan':'kehilangan'
}

# =============================================================================
# 2. Preprocessing Functions Re-definition (from notebook)
# =============================================================================
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
    text = re.sub(r"\s+", " ").strip()
    return text

def preprocess_text_lda(text: str) -> str:
    text = stemmer.stem(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess_single_text(text: str) -> str:
    cleaned_text = preprocess_text(text)
    processed_text_lda = preprocess_text_lda(cleaned_text)
    return processed_text_lda

# =============================================================================
# 3. Topic Name Maps Re-definition
# =============================================================================
topic_name_map_lda = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat"
}

topic_name_map_lstm = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat"
}

# =============================================================================
# 4. Load Models and Processors
# =============================================================================

# Suppress TensorFlow warnings related to loading compiled models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warnings

@st.cache_resource
def load_all_models():
    # LDA Model
    lda_model = gensim.models.LdaMulticore.load("lda_model.gensim")
    dictionary = corpora.Dictionary.load("lda_dictionary.gensim")

    # IndoBERT Sentiment Pipeline
    sentiment_model_dir = "indobert_sentiment_model"
    indobert_sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_dir)

    # LSTM Topic Classification Model
    lstm_topic_model = load_model("lstm_topic_model.h5")
    with open('tokenizer_topic.pkl', 'rb') as handle:
        tokenizer_topic = pickle.load(handle)
    with open('label_encoder_topic.pkl', 'rb') as handle:
        label_encoder_topic = pickle.load(handle)

    # LSTM Sentiment Classification Model
    lstm_sentiment_model = load_model("lstm_sentiment_model.h5")
    with open('tokenizer_sentiment.pkl', 'rb') as handle:
        tokenizer_sentiment = pickle.load(handle)
    with open('label_encoder_sentiment.pkl', 'rb') as handle:
        label_encoder_sentiment = pickle.load(handle)

    return (
        lda_model, dictionary,
        indobert_sentiment_pipeline,
        lstm_topic_model, tokenizer_topic, label_encoder_topic,
        lstm_sentiment_model, tokenizer_sentiment, label_encoder_sentiment
    )

(lda_model_loaded, dictionary_loaded,
 indobert_sentiment_pipeline_loaded,
 lstm_topic_model_loaded, tokenizer_topic_loaded, label_encoder_topic_loaded,
 lstm_sentiment_model_loaded, tokenizer_sentiment_loaded, label_encoder_sentiment_loaded) = load_all_models()

# =============================================================================
# 5. Define maxlen variables
# =============================================================================
maxlen = 20  # From previous notebook output
maxlen_sentiment = 20 # From previous notebook output

# =============================================================================
# 6. Prediction Functions Re-definition
# =============================================================================
def predict_topic_lda(preprocessed_text_lda: str):
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
    sequence = tokenizer_topic_loaded.texts_to_sequences([preprocessed_text_lda])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
    predictions = lstm_topic_model_loaded.predict(padded_sequence, verbose=0)[0]
    dominant_topic_id = np.argmax(predictions)
    dominant_topic_name = label_encoder_topic_loaded.inverse_transform([dominant_topic_id])[0]
    return dominant_topic_id, dominant_topic_name

def predict_sentiment_indobert(cleaned_text: str):
    if not cleaned_text.strip():
        return "neutral"
    sentiment_result = indobert_sentiment_pipeline_loaded(cleaned_text)[0]
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    sentiment_category = label_index[sentiment_result['label']]
    return sentiment_category

def predict_sentiment_lstm(preprocessed_text_lda: str):
    if not preprocessed_text_lda.strip():
        return "neutral"
    sequence = tokenizer_sentiment_loaded.texts_to_sequences([preprocessed_text_lda])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen_sentiment, padding='post', truncating='post')
    predictions = lstm_sentiment_model_loaded.predict(padded_sequence, verbose=0)[0]
    dominant_sentiment_id = np.argmax(predictions)
    dominant_sentiment_name = label_encoder_sentiment_loaded.inverse_transform([dominant_sentiment_id])[0]
    return dominant_sentiment_name

# =============================================================================
# Streamlit Application Interface
# =============================================================================
st.title('Aplikasi Analisis Sentimen dan Topik Ulasan Pengguna')
st.write('Masukkan ulasan pengguna aplikasi Polri Presisi untuk menganalisis topik dan sentimennya.')

user_input = st.text_area('Masukkan ulasan Anda di sini:', '')

if st.button('Analisis Ulasan'):
    if user_input:
        # Preprocess text for models
        lda_ready_text = preprocess_single_text(user_input)
        indobert_ready_text = preprocess_text(user_input)

        st.subheader('Hasil Analisis:')

        if not lda_ready_text.strip():
            st.warning("Ulasan setelah preprocessing menjadi kosong. Tidak dapat menganalisis topik dan LSTM sentimen.")
        else:
            # LDA Topic Prediction
            lda_topic_id, lda_topic_name = predict_topic_lda(lda_ready_text)
            st.write(f"**LDA (Latent Dirichlet Allocation) Topik:** {lda_topic_name} (ID: {lda_topic_id})")

            # LSTM Topic Prediction
            lstm_topic_id, lstm_topic_name = predict_topic_lstm(lda_ready_text)
            st.write(f"**LSTM (Bidirectional LSTM) Topik:** {lstm_topic_name} (ID: {lstm_topic_id})")

            # LSTM Sentiment Prediction
            lstm_sentiment = predict_sentiment_lstm(lda_ready_text)
            st.write(f"**LSTM (Bidirectional LSTM) Sentimen:** {lstm_sentiment}")

        if not indobert_ready_text.strip():
            st.warning("Ulasan setelah preprocessing untuk IndoBERT menjadi kosong. Tidak dapat menganalisis sentimen dengan IndoBERT.")
        else:
            # IndoBERT Sentiment Prediction
            indobert_sentiment = predict_sentiment_indobert(indobert_ready_text)
            st.write(f"**IndoBERT Sentimen:** {indobert_sentiment}")

    else:
        st.warning('Silakan masukkan ulasan terlebih dahulu.')
