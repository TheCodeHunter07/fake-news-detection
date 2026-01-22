import streamlit as st
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model/fake_news_model.pkl", "rb"))

model = load_model()

# -------------------------------
# Custom CSS for Strong UI
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.title-text {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle-text {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 30px;
}

.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.result-real {
    color: #065f46;
    font-size: 26px;
    font-weight: 700;
}

.result-fake {
    color: #991b1b;
    font-size: 26px;
    font-weight: 700;
}

.confidence-text {
    font-size: 16px;
    margin-top: 10px;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="title-text">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">'
    'An NLP-based Machine Learning application to identify whether a news article is Real or Fake.'
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------
# Input Card
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

news_text = st.text_area(
    "Paste the news article text below",
    height=220,
    placeholder="Enter or paste a full news article here for analysis..."
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Analyze News", use_container_width=True):

    if not news_text.strip():
        st.warning("⚠️ Please enter some news text before analyzing.")
    else:
        prediction = model.predict([news_text])[0]
        confidence = model.predict_proba([news_text])[0]
        confidence_score = confidence[prediction] * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<div class="result-real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="confidence-text">Prediction Confidence: <b>{confidence_score:.2f}%</b></div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.progress(int(confidence_score))

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "🔬 Built using TF-IDF, Logistic Regression, and Streamlit | "
    "This tool is for educational and research purposes only."
)
