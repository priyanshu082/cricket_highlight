# audio_processing/extract_audio.py

import os
from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, audio_output_path):
    """
    Extracts the audio from a video and saves it as a WAV file.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)
    video.close()

if __name__ == "__main__":
    # Specify the single video file directly
    video_path = r"cricket_highlight\Videos\10min.mp4"
    audio_dir = r"cricket_highlight\audioProcessing"
    os.makedirs(audio_dir, exist_ok=True)

    audio_output_path = os.path.join(audio_dir, "10min.wav")
    extract_audio_from_video(video_path, audio_output_path)
    print(f"Audio extracted for {video_path}")
