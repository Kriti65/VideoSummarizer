import io
import requests
import yt_dlp
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Whisper Model (Float32 since Float16 isn't supported)
whisper_model = WhisperModel("small", compute_type="float32")

# Load DistilBART Model and its tokenizer
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def extract_audio_stream(video_url):
    """Extract audio stream URL from a YouTube video without downloading."""
    ydl_opts = {
        'format': 'bestaudio',
        'quiet': True,  # Set to False for debugging
        'noprogress': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        if 'url' not in info:
            raise ValueError("Failed to extract audio URL")
    return info['url']

def extract_text_from_audio(audio_url):
    """Transcribe streamed audio using Faster-Whisper without async issues."""
    response = requests.get(audio_url, stream=True)
    if response.status_code != 200:
        raise Exception("Failed to stream audio")
    audio_data = io.BytesIO(response.content)
    torch.set_num_threads(1)  # Prevent overloading system resources
    segments, _ = whisper_model.transcribe(audio_data, language=None)
    text = " ".join(segment.text for segment in segments)
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks for summarization."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def summarize_long_text(text):
    """Summarize long text using DistilBART with proper tokenization."""
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        # Encode the chunk into input IDs for the model.
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = summarization_model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=50, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)
