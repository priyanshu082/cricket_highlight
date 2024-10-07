import cv2
import numpy as np
import pytesseract
import re
import os
import subprocess
from datetime import datetime

class DebugScoreboardOCR:
    def __init__(self, debug_folder=None):
        self.last_valid_score = None
        self.debug_folder = debug_folder
        self.frame_count = 0
        
        # Create debug folder if specified
        if debug_folder:
            os.makedirs(debug_folder, exist_ok=True)
    
    def extract_score(self, frame):
        """Extract score with extensive debugging"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Define ROI
        top = int(height * 0.93)
        bottom = int(height * 0.97)
        left = int(width * 0.13)
        right = int(width * 0.195)
        
        # Extract ROI
        roi = frame[top:bottom, left:right]
        
        # Save original ROI for debugging
        # if self.debug_folder and self.frame_count % 30 == 0:
        #     cv2.imwrite(os.path.join(self.debug_folder, f'original_roi_{self.frame_count}.jpg'), roi)
        
        # Basic preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
        
        # Save preprocessed image for debugging
        # if self.debug_folder and self.frame_count % 30 == 0:
        #     cv2.imwrite(os.path.join(self.debug_folder, f'preprocessed_{self.frame_count}.jpg'), binary)
        
        # OCR with minimal configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, config=custom_config)
        
        # Log OCR output
        if self.debug_folder and self.frame_count % 30 == 0:
            with open(os.path.join(self.debug_folder, 'ocr_log.txt'), 'a') as f:
                f.write(f"\nFrame {self.frame_count}:\nRaw OCR output: {text}\n")
        
        # Try to find score pattern
        score_match = re.search(r'(\d{1,3})[^0-9a-zA-Z]*(\d{1,2})', text)
        
        if score_match:
            try:
                runs = int(score_match.group(1))
                wickets = int(score_match.group(2))
                
                if 0 <= runs <= 999 and 0 <= wickets <= 10:
                    score = f"{runs}-{wickets}"
                    
                    # Log valid score
                    if self.debug_folder:
                        with open(os.path.join(self.debug_folder, 'score_log.txt'), 'a') as f:
                            f.write(f"Frame {self.frame_count}: Valid score detected: {score}\n")
                    
                    self.last_valid_score = score
                    return score
            except ValueError:
                pass
        
        return self.last_valid_score

def extract_highlights(video_path, output_folder, debug=True):
    """Extract highlights with extensive debugging"""
    
    # Create debug folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_folder = os.path.join(output_folder, f"debug")
    if debug:
        os.makedirs(debug_folder, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps=30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize OCR with debug folder
    ocr = DebugScoreboardOCR(debug_folder if debug else None)
    
    frame_count = 0
    prev_score = None
    highlights = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 15th frame
            if frame_count % 15 == 0:
                # Get current score
                current_score_text = ocr.extract_score(frame)
                
                # Log progress
                if frame_count % 150 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Current Score: {current_score_text}")
                
                # Process score if detected
                if current_score_text and prev_score:
                    try:
                        curr_runs, curr_wickets = map(int, current_score_text.split('-'))
                        prev_runs, prev_wickets = map(int, prev_score.split('-'))
                        
                        # Detect significant events
                        runs_diff = curr_runs - prev_runs
                        wickets_diff = curr_wickets - prev_wickets
                        
                        if runs_diff in {4, 6} or wickets_diff > 0:
                            # Calculate actual event frame
                            event_frame = frame_count

                            # Calculate start and end frames, ensuring we don't go out of bounds
                            start_frame = max(1, event_frame - (fps * 11))  # 12 seconds before
                            end_frame = min(total_frames, event_frame + (fps * 6))  # 8 seconds after
                            
                            event_type = 'Wicket' if wickets_diff > 0 else f'Boundary ({runs_diff} runs)'
                            
                            highlights.append({
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'event_type': event_type,
                                'score_change': f'{prev_score} to {current_score_text}',
                                'event_frame': event_frame  # Store the actual event frame for verification
                            })
                            
                            # Debug output using print statements
                            if debug:
                                with open(os.path.join(debug_folder, 'highlights_log.txt'), 'a') as f:
                                    f.write(f"\nHighlight detected:\n")
                                    f.write(f"Event Frame: {event_frame} ({event_frame/fps:.1f}s)\n")
                                    f.write(f"Start Frame: {start_frame} ({start_frame/fps:.1f}s)\n")
                                    f.write(f"End Frame: {end_frame} ({end_frame/fps:.1f}s)\n")
                                    f.write(f"Score Change: {prev_score} to {current_score_text}\n")

                
                    except ValueError as e:
                        if debug:
                            with open(os.path.join(debug_folder, 'error_log.txt'), 'a') as f:
                                f.write(f"Error processing scores at frame {frame_count}: {str(e)}\n")
                
                prev_score = current_score_text
    
    finally:
        cap.release()
    
    # Extract highlight clips
    if highlights:
        highlight_folder = os.path.join(output_folder, f"highlights")
        os.makedirs(highlight_folder, exist_ok=True)
        
        for idx, highlight in enumerate(highlights, 1):
            output_file = os.path.join(highlight_folder, f"highlight_{idx}.mp4")
            start_time = highlight['start_frame'] / fps
            end_time = highlight['end_frame'] / fps
            
            command = [
                'ffmpeg', '-y',
                '-ss', str(highlight['start_frame'] / fps),  # Convert frames to seconds
                '-i', video_path,  # Input file
                '-t', str((highlight['end_frame'] - highlight['start_frame']) / fps),  # Duration instead of end time
                '-c:v', 'libx264',
                '-c:a', 'aac',
                output_file
            ]
            
            subprocess.run(command)
    
    return len(highlights)

# Usage
if __name__ == "__main__":
    video_path = 'cricket_match.mp4'  # Replace with your video path
    output_folder = 'extracted_details'
    
    print("Starting highlight extraction with debug mode...")
    num_highlights = extract_highlights(video_path, output_folder, debug=True)
    
    print(f"\nExtraction complete!")
    print(f"Number of highlights extracted: {num_highlights}")
    print(f"Check the debug folder for detailed logs and intermediate images")


import cv2
import numpy as np
import pytesseract
import re
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class ScoreboardOCR:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.last_valid_score = None
        self.confidence_threshold = 0.7
        
    def preprocess_scoreboard(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect and preprocess the scoreboard region"""
        height, width = frame.shape[:2]
        
        # Define multiple candidate regions
        regions = [
            (int(height * 0.93), int(height * 0.97), int(width * 0.13), int(width * 0.195)),  # Original
            (int(height * 0.92), int(height * 0.98), int(width * 0.12), int(width * 0.20)),   # Slightly larger
            (int(height * 0.94), int(height * 0.96), int(width * 0.14), int(width * 0.19))    # Slightly smaller
        ]
        
        best_region = None
        best_score = 0
        best_processed = None
        
        for region in regions:
            top, bottom, left, right = region
            roi = frame[top:bottom, left:right]
            
            # Skip if ROI is empty
            if roi.size == 0:
                continue
                
            # Process the region
            processed = self._apply_preprocessing(roi)
            
            # Calculate contrast and clarity metrics
            clarity_score = self._calculate_clarity_score(processed)
            
            if clarity_score > best_score:
                best_score = clarity_score
                best_region = region
                best_processed = processed
        
        if best_processed is None:
            raise ValueError("No valid scoreboard region found")
            
        return best_processed, best_region
    
    def _apply_preprocessing(self, roi: np.ndarray) -> np.ndarray:
        """Apply various preprocessing techniques to improve OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Increase resolution
        enhanced = cv2.resize(enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilate to make text thicker
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        if self.debug_mode:
            self._show_debug_images(roi, gray, enhanced, binary)
            
        return binary
    
    def _calculate_clarity_score(self, img: np.ndarray) -> float:
        """Calculate a clarity score for the image"""
        # Calculate image statistics
        mean = np.mean(img)
        std = np.std(img)
        
        # Calculate Laplacian variance (measure of focus)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        focus_measure = np.var(laplacian)
        
        # Combine metrics into a single score
        clarity_score = (std / mean) * np.log(1 + focus_measure)
        
        return clarity_score
    
    def _show_debug_images(self, original, gray, enhanced, binary):
        """Show debug images for preprocessing steps"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        
        plt.subplot(142)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        
        plt.subplot(143)
        plt.imshow(enhanced, cmap='gray')
        plt.title('Enhanced')
        
        plt.subplot(144)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary')
        
        plt.show()
    
    def extract_score(self, frame: np.ndarray) -> Optional[str]:
        """Extract score from the frame with validation"""
        try:
            # Preprocess the scoreboard region
            processed_roi, region = self.preprocess_scoreboard(frame)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789- tessedit_min_confidence=60'
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(processed_roi, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence scores
            texts = []
            confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf > -1:  # Filter out unrecognized text
                    text = ocr_data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)
            
            # Combine detected text
            combined_text = ''.join(texts)
            
            # Apply regex pattern matching
            score_match = re.search(r'(\d{1,3})-(\d{1,2})', combined_text)
            
            if score_match:
                runs = int(score_match.group(1))
                wickets = int(score_match.group(2))
                
                # Validate score
                if self._is_valid_score(runs, wickets):
                    score = f"{runs}-{wickets}"
                    self.last_valid_score = score
                    return score
            
            return self.last_valid_score
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in score extraction: {str(e)}")
            return self.last_valid_score
    
    def _is_valid_score(self, runs: int, wickets: int) -> bool:
        """Validate if the detected score is reasonable"""
        # Basic cricket score validation rules
        if not (0 <= runs <= 999):  # Maximum reasonable runs
            return False
        if not (0 <= wickets <= 10):  # Maximum wickets in cricket
            return False
            
        # Check for reasonable run-wicket relationship
        if wickets == 10 and runs == 0:  # Unlikely to be all out for 0
            return False
        
        # Compare with last valid score if available
        if self.last_valid_score:
            last_runs, last_wickets = map(int, self.last_valid_score.split('-'))
            
            # Score shouldn't decrease
            if runs < last_runs:
                return False
            
            # Wickets shouldn't decrease
            if wickets < last_wickets:
                return False
            
            # Runs shouldn't increase too dramatically
            if runs > last_runs + 6:
                return False
            
            # Wickets shouldn't increase by more than 1
            if wickets > last_wickets + 1:
                return False
        
        return True

# Modified highlight extraction function to use the enhanced OCR
def extract_highlight_videos(video_path: str, output_folder: str, debug_mode: bool = False):
    ocr_processor = ScoreboardOCR(debug_mode=debug_mode)
    cap = cv2.VideoCapture(video_path)
    
    # Rest of your highlight extraction code, but replace the OCR part with:
    # score = ocr_processor.extract_score(frame)
    
    # Example usage in your main loop:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        score = ocr_processor.extract_score(frame)
        if score:
            # Process the score as before
            pass

# Usage example
if __name__ == "__main__":
    # Enable debug mode to see preprocessing steps
    ocr = ScoreboardOCR(debug_mode=True)
    
    # Test with a single frame
    frame = cv2.imread('example_frame.jpg')
    score = ocr.extract_score(frame)
    print(f"Detected score: {score}")
    
    # Or use in video processing
    video_path = 'cricket_match.mp4'
    output_folder = 'highlights'
    extract_highlight_videos(video_path, output_folder, debug_mode=True) merge these two codes 