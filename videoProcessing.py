import cv2
import numpy as np

def extract_scoreboard(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS: {fps}, Width: {width}, Height: {height}")
    
    # Define the scoreboard region
    top = int(height * 0.93)
    bottom = int(height * 0.97)
    left = int(width * 0.07)
    right = int(width * 0.30)
    scoreboard_region = (top, bottom, left, right)
    
    # Calculate the height of the cropped region
    crop_height = bottom - top
    crop_width = right - left
    # Create a VideoWriter object for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the scoreboard region
        scoreboard = frame[scoreboard_region[0]:scoreboard_region[1],
                           scoreboard_region[2]:scoreboard_region[3]]
        
        # Write the cropped frame to the output video
        out.write(scoreboard)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
extract_scoreboard('test.mp4', 'output1.mp4')