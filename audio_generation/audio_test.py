import moviepy.editor as mp
import librosa
import numpy as np
from scipy.signal import find_peaks

def extract_audio(video_path, audio_output_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)
    return audio_output_path

def preprocess_audio(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr

def compute_noise_level(y, frame_length=2048, hop_length=512):
    # Compute the short-time energy
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return energy.squeeze()

def detect_highlights(noise_level, threshold_factor=1.5, min_distance=10):
    # Compute the dynamic threshold
    threshold = np.mean(noise_level) + threshold_factor * np.std(noise_level)
    
    # Find peaks above the threshold
    peaks, _ = find_peaks(noise_level, height=threshold, distance=min_distance)
    
    return peaks

def generate_highlights(video_path, peak_frames, duration=10, hop_length=512, sr=22050):
    video = mp.VideoFileClip(video_path)
    highlights = []
    
    for peak in peak_frames:
        # Convert frame index to time
        peak_time = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
        
        start_time = max(0, peak_time - duration/2)
        end_time = min(video.duration, peak_time + duration/2)
        
        highlight = video.subclip(start_time, end_time)
        highlights.append(highlight)
    
    return highlights

def cricket_highlight_detection(video_path, threshold_factor=1.5, highlight_duration=10):
    # Extract audio
    audio_path = extract_audio(video_path, "temp_audio.wav")
    
    # Load and preprocess audio
    y, sr = preprocess_audio(audio_path)
    
    # Compute noise level
    noise_level = compute_noise_level(y)
    
    # Detect highlight moments
    highlight_frames = detect_highlights(noise_level, threshold_factor)
    
    # Generate video highlights
    highlights = generate_highlights(video_path, highlight_frames, duration=highlight_duration, sr=sr)
    
    return highlights

# Usage
video_path = "3_min.mp4"
highlights = cricket_highlight_detection(video_path)

if highlights:
    final_video = mp.concatenate_videoclips(highlights)
    final_video.write_videofile("output.mp4")
else:
    print("No highlights detected or error occurred during processing.")

# Optional: Visualization of noise levels and detected peaks
import matplotlib.pyplot as plt

def visualize_noise_levels(y, sr, noise_level, highlight_frames):
    plt.figure(figsize=(15, 5))
    times = librosa.times_like(noise_level, sr=sr, hop_length=512)
    plt.plot(times, noise_level)
    plt.plot(times[highlight_frames], noise_level[highlight_frames], "ro")
    plt.title("Noise Level and Detected Highlights")
    plt.xlabel("Time (s)")
    plt.ylabel("Noise Level")
    plt.show()

# Uncomment the following lines to visualize
# y, sr = preprocess_audio("temp_audio.wav")
# noise_level = compute_noise_level(y)
# highlight_frames = detect_highlights(noise_level)
# visualize_noise_levels(y, sr, noise_level, highlight_frames)