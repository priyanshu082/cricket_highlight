import cv2
import numpy as np
import pytesseract
import re
from collections import deque
import os
import subprocess

# Function to locate the scoreboard region in the video frame
def locate_scoreboard(frame):
    height, width = frame.shape[:2]
    top = int(height * 0.93)
    bottom = int(height * 0.97)
    left = int(width * 0.07)
    right = int(width * 0.30)
    return (top, bottom, left, right)

# Function to apply OCR to extract the score from the scoreboard region
def apply_ocr(frame):
    scoreboard_region = locate_scoreboard(frame)
    roi = frame[scoreboard_region[0]:scoreboard_region[1], scoreboard_region[2]:scoreboard_region[3]]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'
    text = pytesseract.image_to_string(gray, config=custom_config)
    
    score_match = re.search(r'\d{1,3}-\d{1,2}', text)
    return score_match.group(0) if score_match else None

# Function to extract batting score and wicket count from a score string (e.g., "31-1")
def parse_score(score):
    if score:
        runs, wickets = map(int, score.split('-'))
        return runs, wickets
    return None, None

# Function to extract highlight videos with audio based on score changes
def extract_highlight_videos(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    highlight_count = 0
    skip_frames = 30  # Process every 30th frame

    frame_buffer = deque(maxlen=fps * 8)  # Store 8 seconds of frames
    score_history = deque(maxlen=2)  # Store current and previous score

    timestamps = []  # To store the start and end time of each highlight

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_buffer.append(frame)

        if frame_count % skip_frames == 0:
            current_score = apply_ocr(frame)
            
            if current_score and len(score_history) == 2:
                prev_runs, prev_wickets = parse_score(score_history[-1])
                curr_runs, curr_wickets = parse_score(current_score)

                if prev_runs is not None and curr_runs is not None:
                    # Check for boundary conditions (4 or 6 runs) and wickets
                    runs_diff = curr_runs - prev_runs
                    wicket_diff = curr_wickets - prev_wickets

                    if runs_diff in {4, 6} or wicket_diff > 0:
                        # Calculate the start and end time for the highlight
                        start_time = max(0, frame_count - len(frame_buffer)) / fps
                        end_time = frame_count / fps + 8  # Add 8 seconds post event
                        timestamps.append((start_time - 4, end_time-4))
                        
                        highlight_type = "Boundary" if runs_diff in {4, 6} else "Wicket"
                        print(f"{highlight_type} detected: {score_history[-1]} to {current_score}. Highlight from {start_time-4} to {end_time-4} seconds.")
                        highlight_count += 1

            if current_score:
                score_history.append(current_score)

    cap.release()
    cv2.destroyAllWindows()

    # Use ffmpeg to extract clips with audio
    if timestamps:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for idx, (start, end) in enumerate(timestamps):
            output_file = os.path.join(output_folder, f"highlight_{idx + 1}.mp4")
            command = [
                'ffmpeg', '-y',  # Overwrite output files if exist
                '-i', video_path,  # Input video path
                '-ss', str(start),  # Start time
                '-to', str(end),  # End time
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                output_file  # Output file path
            ]
            subprocess.run(command)
            print(f"Highlight saved: {output_file}")
    else:
        print("No highlights detected.")

# Usage
video_path = 'cricket_match.mp4'  # Replace with your full video path
output_folder = 'extracted_highlights'  # Replace with your desired output folder
extract_highlight_videos(video_path, output_folder)
