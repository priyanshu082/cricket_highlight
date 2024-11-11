import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to compute the histogram of a frame
def compute_histogram(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Compute histograms for the Hue, Saturation, and Value channels
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])  # Hue
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Value (brightness)
    
    # Normalize the histograms
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

        # Compute the histogram for the current frame
        hist = compute_histogram(frame)
        features.append(hist)
        frame_numbers.append(frame_num)
        frame_num += 1

    cap.release()
    return np.array(features), frame_numbers

# Function to perform K-means clustering and select keyframes
def select_keyframes(video_path, num_clusters=5):
    # Extract features from the video
    features, frame_numbers = extract_features(video_path)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    
    # Get the indices of the frames closest to the cluster centroids
    keyframe_indices = []
    for i in range(num_clusters):
        # Find the closest frame to the centroid of each cluster
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        distances = np.linalg.norm(features[cluster_indices] - kmeans.cluster_centers_[i], axis=1)
        closest_frame_index = cluster_indices[np.argmin(distances)]
        keyframe_indices.append(closest_frame_index)

    keyframe_indices = sorted(keyframe_indices)
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
video_path = 'your_video.mp4'
keyframe_indices = select_keyframes(video_path, num_clusters=5)
print(f"Keyframes are selected from frames: {keyframe_indices}")

# Visualize keyframes (optional)
visualize_keyframes(video_path, keyframe_indices)
