import cv2
import pytesseract
import re
from collections import deque

def apply_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    x, y, w, h = 150, 0, 200, 100  # Adjust based on your video layout
    roi = gray[y:y+h, x:x+w]
    
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'
    text = pytesseract.image_to_string(roi, config=custom_config)
    
    score_match = re.search(r'\d{1,3}-\d{1,2}', text)
    
    return score_match.group(0) if score_match else None

def extract_highlight_videos(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_buffer = deque(maxlen=fps * 8)  # Store 8 seconds of frames
    score_history = deque(maxlen=2)  # Store current and previous score
    
    frame_count = 0
    highlight_count = 0
    skip_frames = 30  # Process every 30th frame
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
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
                    print(f"Score change detected: {score_history[-1]} to {current_score}")
                    
                    # Start a new video writer
                    out = cv2.VideoWriter(f"{output_folder}/highlight_{highlight_count}.mp4", 
                                          fourcc, fps, (width, height))
                    
                    # Write the buffered frames
                    for buffered_frame in frame_buffer:
                        out.write(buffered_frame)
                    
                    # Continue writing frames for the next 8 seconds
                    frames_to_write = fps * 8
                    while frames_to_write > 0 and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frames_to_write -= 1
                    
                    # Release the video writer
                    out.release()
            
            if current_score:
                score_history.append(current_score)
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Usage
video_path = 'output.mp4'  # Replace with your video path
output_folder = 'extracted_highlights'  # Replace with your desired output folder
extract_highlight_videos(video_path, output_folder)