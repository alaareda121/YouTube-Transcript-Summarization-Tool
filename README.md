# YouTube Video Summarizer

A Python application that extracts transcripts from YouTube videos and summarizes them using a Transformer-based NLP model.

## What This App Does
- Takes a YouTube video URL
- Extracts the video transcript
- Generates a summary (Short, Medium, or Long)
- Displays results in a simple Streamlit interface

## Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- YouTube Transcript API

## How to Run the Project
1. Install the required libraries:
   pip install -r requirements.txt

2. Run the application:
   streamlit run youtube_gui.py