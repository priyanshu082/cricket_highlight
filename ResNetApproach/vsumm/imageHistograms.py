import cv2
import numpy as np

# Function to compute the histogram of a frame
def compute_histogram(frame):
    # Convert the frame to HSV color space (better for detecting color changes)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Compute histograms for the Hue, Saturation, and Value channels
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])  # Hue
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Value (brightness)
    return hist_hue, hist_saturation, hist_value

# Function to compare histograms using correlation
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to detect scene change based on histogram comparison
def detect_scene_changes(video_path, threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        return

    # Compute the histogram for the first frame
    hist_hue, hist_saturation, hist_value = compute_histogram(prev_frame)

    frame_num = 1
    scene_changes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute the histogram for the current frame
        curr_hist_hue, curr_hist_saturation, curr_hist_value = compute_histogram(frame)

        # Compare histograms (Hue, Saturation, and Value)
        hue_corr = compare_histograms(hist_hue, curr_hist_hue)
        sat_corr = compare_histograms(hist_saturation, curr_hist_saturation)
        val_corr = compare_histograms(hist_value, curr_hist_value)

        # Average correlation across all channels
        avg_corr = (hue_corr + sat_corr + val_corr) / 3

        # If correlation is below the threshold, it indicates a scene change
        if avg_corr < threshold:
            scene_changes.append(frame_num)
            print(f"Scene change detected at frame {frame_num}")

        # Update previous frame and histograms for the next iteration
        hist_hue, hist_saturation, hist_value = curr_hist_hue, curr_hist_saturation, curr_hist_value
        frame_num += 1

    cap.release()
    return scene_changes

# Example usage
video_path = r'D:\Git_main\cricket_highlight\Videos\10min.mp4'
scene_changes = detect_scene_changes(video_path)
print("Scene changes detected at frames:", scene_changes)
