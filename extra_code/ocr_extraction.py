import pytesseract
from PIL import Image
import cv2
import re

def apply_ocr(frame, frame_count):
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
        return "No score detected"

# Example of applying OCR on video frames
cap = cv2.VideoCapture('output.mp4')
frame_count = 0
skip_frames = 30  # Process every 30th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if there are no more frames

    if frame_count % skip_frames == 0:
        ocr_text = apply_ocr(frame, frame_count)
        print(f"Frame {frame_count}, Detected Text: {ocr_text}")

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows after the loop
cap.release()
cv2.destroyAllWindows()
