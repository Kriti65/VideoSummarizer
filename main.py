import streamlit as st
from utils import extract_audio_stream, extract_text_from_audio, summarize_long_text, chunk_text

def main():
    st.title("YouTube Video Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Generate Summary"):
        st.spinner("Extracting Audio...")
        try:
            audio_url = extract_audio_stream(video_url)
            st.write(f"Audio URL: {audio_url}")  # Debugging step

            st.spinner("Transcribing Audio...")
            transcript = extract_text_from_audio(audio_url)
            st.write(f"Transcript: {transcript[:500]}")  # Show only first 500 chars

            st.spinner("Summarizing Text...")
            summary = summarize_long_text(transcript)
            st.success(summary)
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
