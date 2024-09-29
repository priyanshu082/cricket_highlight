import cv2
import pytesseract
from PIL import Image
import re
from collections import deque

def apply_ocr(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to make OCR easier
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply a simple binary threshold for better contrast
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Define Region of Interest (ROI) for the score area
    x, y, w, h = 150, 0, 200, 100  # Adjust based on your video layout
    roi = gray[y:y+h, x:x+w]
    
    # Restrict OCR to recognize only numbers and the dash
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'
    text = pytesseract.image_to_string(roi, config=custom_config)
    
    # Use regex to find the pattern you're looking for (e.g., "31-0")
    score_match = re.search(r'\d{1,3}-\d{1,2}', text)
    
    if score_match:
        return score_match.group(0)
    else:
        return None

def extract_highlight_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_buffer = deque(maxlen=fps * 8)  # Store 8 seconds of frames
    score_history = deque(maxlen=2)  # Store current and previous score
    
    frame_count = 0
    highlight_count = 0
    saving_frames = False
    frames_to_save = 0
    skip_frames = 30  # Process every 30th frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_buffer.append(frame)
        
        if frame_count % skip_frames == 0:
            current_score = apply_ocr(frame)
            
            if current_score and len(score_history) == 2:
                if current_score != score_history[-1]:
                    highlight_count += 1
                    saving_frames = True
                    frames_to_save = fps * 16  # Save 8 seconds before and 8 seconds after
                    print(f"Score change detected: {score_history[-1]} to {current_score}")
            
            if current_score:
                score_history.append(current_score)
        
        if saving_frames:
            save_frame(frame, output_folder, highlight_count, frame_count)
            frames_to_save -= 1
            if frames_to_save == 0:
                saving_frames = False
    
    cap.release()
    cv2.destroyAllWindows()

def save_frame(frame, output_folder, highlight_count, frame_count):
    output_path = f"{output_folder}/highlight_{highlight_count}_frame_{frame_count:06d}.jpg"
    cv2.imwrite(output_path, frame)

# Usage
video_path = 'output.mp4'  # Replace with your video path
output_folder = 'extracted_frames'  # Replace with your desired output folder
extract_highlight_frames(video_path, output_folder)