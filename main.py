import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import nltk
import re

# ------------------ NLTK DOWNLOAD (SAFE) ------------------
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk()

# ------------------ EXTRACT VIDEO ID ------------------
def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu.be/([^?]+)",
        r"youtube.com/embed/([^?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ------------------ FETCH TRANSCRIPT ------------------
def get_transcript(video_id):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return transcript_list.find_transcript(['en']).fetch()

# ------------------ SUMMARIZATION (CHUNKED) ------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = load_summarizer()
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []

    for chunk in chunks[:5]:  # limit for performance
        s = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(s[0]['summary_text'])

    return " ".join(summaries)

# ------------------ KEYWORD EXTRACTION ------------------
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words if word.isalnum()
    ]

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform([' '.join(filtered_words)])
    return vectorizer.get_feature_names_out()

# ------------------ TOPIC MODELING ------------------
def topic_modeling(text):
    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=1,
        stop_words='english'
    )
    tf = vectorizer.fit_transform([text])

    lda = LatentDirichletAllocation(
        n_components=3,
        random_state=42
    )
    lda.fit(tf)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic in lda.components_:
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])

    return topics

# ------------------ STREAMLIT APP ------------------
def main():
    st.title("🎥 YouTube Video Summarizer")

    video_url = st.text_input("Enter YouTube Video URL")

    if st.button("Summarize"):
        try:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL")
                return

            with st.spinner("Fetching transcript..."):
                transcript = get_transcript(video_id)

            text = " ".join([t['text'] for t in transcript])

            with st.spinner("Summarizing video..."):
                summary = summarize_text(text)

            keywords = extract_keywords(text)
            topics = topic_modeling(text)
            sentiment = TextBlob(text).sentiment

            st.subheader("📌 Summary")
            st.write(summary)

            st.subheader("🔑 Keywords")
            st.write(list(keywords))

            st.subheader("🧠 Topics")
            for i, topic in enumerate(topics, 1):
                st.write(f"Topic {i}: {', '.join(topic)}")

            st.subheader("😊 Sentiment Analysis")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("❌ Transcripts are disabled for this video")
        except NoTranscriptFound:
            st.error("❌ No transcript found")
        except Exception as e:
            st.error(f"⚠ Error: {str(e)}")

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    main()
