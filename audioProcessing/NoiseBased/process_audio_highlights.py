import os
import sys

# Add the parent directory of 'cricket_highlight' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from cricket_highlight.audioProcessing.NoiseBased.extract_audio import extract_audio_from_video
from cricket_highlight.audioProcessing.NoiseBased.noise_detection import detect_noise_peaks

def process_audio_highlights(video_path, audio_output_dir, highlight_times_output):
    audio_output_path = os.path.join(audio_output_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
    extract_audio_from_video(video_path, audio_output_path)
    
    # Detect noise peaks
    peak_times = detect_noise_peaks(audio_output_path)
    
    # Group times into clusters
    clustered_times = cluster_times(peak_times, threshold=1.0)  # Group times within 1 second of each other
    
    # Save detected times to a file or return them
    with open(highlight_times_output, 'w') as f:
        for cluster in clustered_times:
            f.write(f"{cluster}\n")
    
    return clustered_times

def cluster_times(times, threshold=1.0):
    """
    Group times that are within the threshold of each other.
    """
    if not times:
        return []
    
    clusters = []
    current_cluster = [times[0]]  # Start the first cluster with the first time
    
    for i in range(1, len(times)):
        if times[i] - current_cluster[-1] <= threshold:
            current_cluster.append(times[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [times[i]]
    
    # Append the last cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters

if __name__ == "__main__":
    # Use absolute paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    video_dir = os.path.join(base_dir, "cricket_highlight", "audioProcessing")
    audio_dir = os.path.join(base_dir, "cricket_highlight", "audioProcessing", "audio_output")
    highlights_dir = os.path.join(base_dir, "cricket_highlight", "audioProcessing","NoiseBased" , "highlights_output")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(highlights_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            highlight_times_output = os.path.join(highlights_dir, video_file.replace('.mp4', '_times.txt'))
            try:
                clustered_times = process_audio_highlights(video_path, audio_dir, highlight_times_output)
                print(f"Clustered highlight times for {video_file}: {clustered_times}")
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")