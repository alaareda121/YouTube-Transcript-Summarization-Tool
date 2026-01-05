import streamlit as st
from transformers import pipeline
import torch
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📝")

# ------------------ DARK MODE ------------------ #
dark_css = """
<style>
body {
    background-color: #0d0d0d;
    color: white;
}
span, label, .stTextInput, .stButton>button {
    color: white !important;
}
.stTextInput>div>div>input {
    background-color: #1e1e1e;
    color: white;
    border: 1px solid #555;
}
.stButton>button {
    background-color: #333 !important;
    color: white !important;
    border-radius: 10px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #555 !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ------------------ MODEL ------------------ #
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )

pipe = load_model()

# ------------------ FUNCTIONS ------------------ #

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    video_ids = qs.get('v')
    if not video_ids:
        raise ValueError("Invalid YouTube URL")
    return video_ids[0]


def get_transcript(url: str) -> str:
    video_id = extract_video_id(url)

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=['en'])

    text = " ".join(snippet.text for snippet in fetched)
    return text


def summarize_text(text, max_len, min_len):
    summary_list = pipe(
        text[:1024],
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=2,
        early_stopping=True
    )

    return summary_list[0]["summary_text"]

# ------------------ UI ------------------ #

st.title("🎥 YouTube Video Summarizer")

video_url = st.text_input("Enter YouTube Video URL")

summary_size = st.selectbox(
    "Select Summary Length",
    ["Short", "Medium", "Long"]
)

if summary_size == "Short":
    max_len, min_len = 80, 30
elif summary_size == "Medium":
    max_len, min_len = 150, 60
else:
    max_len, min_len = 300, 120


if st.button("Summarize"):
    if video_url.strip() == "":
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Fetching transcript..."):
             transcript = get_transcript(video_url)

             st.subheader("📄 Transcript")
             st.write(transcript)

            with st.spinner("Summarizing..."):
             summary = summarize_text(transcript, max_len, min_len)

            st.subheader("✅ Summary")
            st.write(summary)


        except Exception as e:
         st.error(f"Error: {str(e)}")
