import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to compute the histogram of a frame
def compute_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])  # Hue
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Value (brightness)
    
    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()

    return np.concatenate((hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()))

# Function to extract frames from a video and compute their histograms
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_numbers = []
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hist = compute_histogram(frame)
        features.append(hist)
        frame_numbers.append(frame_num)
        frame_num += 1

    cap.release()
    return np.array(features), frame_numbers

# Function to perform K-means clustering and select central frames as keyframes
def select_keyframes_vsum(video_path, num_clusters=5):
    features, frame_numbers = extract_features(video_path)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    
    keyframe_indices = []
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        distances = np.linalg.norm(features[cluster_indices] - kmeans.cluster_centers_[i], axis=1)
        closest_frame_index = cluster_indices[np.argmin(distances)]
        keyframe_indices.append(closest_frame_index)

    keyframe_indices = sorted(keyframe_indices)

    # Get the fps of the video to calculate timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Calculate timestamps for the selected keyframes
    keyframe_timestamps = [(frame_numbers[i], frame_numbers[i] / fps) for i in keyframe_indices]
    
    # Print the keyframe indices along with their timestamps
    for frame_idx, timestamp in keyframe_timestamps:
        print(f"Keyframe at Frame {frame_idx}, Timestamp: {timestamp:.2f} seconds")

    return [frame_numbers[i] for i in keyframe_indices]

# Function to visualize the keyframes (optional)
def visualize_keyframes(video_path, keyframe_indices):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in keyframe_indices:
            keyframes.append(frame)
        
        frame_num += 1
    
    cap.release()

    # Display keyframes
    for i, keyframe in enumerate(keyframes):
        plt.subplot(1, len(keyframes), i+1)
        plt.imshow(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

# Example usage
video_path = r'D:\Git_main\cricket_highlight\Videos\10min.mp4'
keyframe_indices = select_keyframes_vsum(video_path, num_clusters=5)
print(f"Keyframes are selected from frames: {keyframe_indices}")

# Visualize keyframes (optional)
visualize_keyframes(video_path, keyframe_indices)
