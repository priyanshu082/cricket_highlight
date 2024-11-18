import cv2
import numpy as np
import pytesseract
import re
import os
import subprocess
from datetime import datetime
from typing import Optional, Tuple

class OptimizedScoreboardOCR:
    def __init__(self, debug_folder=None, debug_mode=False):
        self.last_valid_score = None
        self.debug_folder = debug_folder
        self.debug_mode = debug_mode
        self.frame_count = 0
        
        # Create debug folder if specified
        if debug_folder:
            os.makedirs(debug_folder, exist_ok=True)
    
    def extract_score(self, frame: np.ndarray) -> Optional[str]:
        """Extract and validate score from frame using standard parameters"""
        self.frame_count += 1
        
        try:
            height, width = frame.shape[:2]
            
            # Use original ROI parameters
            top = int(height * 0.93)
            bottom = int(height * 0.97)
            left = int(width * 0.13)
            right = int(width * 0.195)
            
            # Extract ROI
            roi = frame[top:bottom, left:right]
            
            # Save original ROI for debugging
            if self.debug_folder and self.frame_count % 30 == 0:
                cv2.imwrite(os.path.join(self.debug_folder, f'original_roi_{self.frame_count}.jpg'), roi)
            
            # Basic preprocessing with standard parameters
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
            
            # Save preprocessed image for debugging
            if self.debug_folder and self.frame_count % 30 == 0:
                cv2.imwrite(os.path.join(self.debug_folder, f'preprocessed_{self.frame_count}.jpg'), binary)
            
            # OCR with standard configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            # Log OCR output if debug enabled
            if self.debug_folder and self.frame_count % 30 == 0:
                with open(os.path.join(self.debug_folder, 'ocr_log.txt'), 'a') as f:
                    f.write(f"\nFrame {self.frame_count}:\nRaw OCR output: {text}\n")
            
            # Match score pattern
            score_match = re.search(r'(\d{1,3})[^0-9a-zA-Z]*(\d{1,2})', text)
            
            if score_match:
                runs = int(score_match.group(1))
                wickets = int(score_match.group(2))
                
                if self._is_valid_score(runs, wickets):
                    score = f"{runs}-{wickets}"
                    
                    # Log valid score
                    if self.debug_folder:
                        with open(os.path.join(self.debug_folder, 'score_log.txt'), 'a') as f:
                            f.write(f"Frame {self.frame_count}: Valid score detected: {score}\n")
                    
                    self.last_valid_score = score
                    return score
            
            return self.last_valid_score
            
        except Exception as e:
            if self.debug_folder:
                with open(os.path.join(self.debug_folder, 'error_log.txt'), 'a') as f:
                    f.write(f"Error in frame {self.frame_count}: {str(e)}\n")
            return self.last_valid_score
    
    def _is_valid_score(self, runs: int, wickets: int) -> bool:
        """Validate cricket score"""
        if not (0 <= runs <= 999) or not (0 <= wickets <= 10):
            return False
            
        if wickets == 10 and runs == 0:
            return False
        
        if self.last_valid_score:
            last_runs, last_wickets = map(int, self.last_valid_score.split('-'))

            if runs == 0 and wickets == 0 and (last_runs > 0 or last_wickets > 0):
                return True
            
            if runs < last_runs or wickets < last_wickets:
                return False
            
            if runs > last_runs + 6:
                return False
            
            if wickets > last_wickets + 1:
                return False
        
        return True

def extract_highlights(video_path: str, output_folder: str, debug: bool = True):
    """Extract cricket highlights with standard OCR parameters"""
    
    # Setup debug folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_folder = os.path.join(output_folder, f"debug")
    if debug:
        os.makedirs(debug_folder, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = 30  # Fixed FPS for consistency
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize OCR
    ocr = OptimizedScoreboardOCR(debug_folder if debug else None, debug_mode=debug)
    
    frame_count = 0
    prev_score = None
    highlights = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 15th frame for efficiency
            if frame_count % 60 == 0:
                current_score_text = ocr.extract_score(frame)
                
                # Log progress
                if frame_count % 150 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Current Score: {current_score_text}")
                
                # Process score changes
                if current_score_text and prev_score:
                    try:
                        curr_runs, curr_wickets = map(int, current_score_text.split('-'))
                        prev_runs, prev_wickets = map(int, prev_score.split('-'))
                        
                        runs_diff = curr_runs - prev_runs
                        wickets_diff = curr_wickets - prev_wickets
                        
                        if runs_diff in {4, 6} or wickets_diff > 0:
                            event_frame = frame_count
                            before_time = 8  # 8 seconds before
                            after_time = 8   # 5 seconds after
                            
                            start_frame = max(1, event_frame - (fps * before_time))
                            end_frame = min(total_frames, event_frame + (fps * after_time))
                            
                            event_type = 'Wicket' if wickets_diff > 0 else f'Boundary ({runs_diff} runs)'
                            
                            highlights.append({
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'event_type': event_type,
                                'score_change': f'{prev_score} to {current_score_text}',
                                'event_frame': event_frame
                            })
                            
                            # Debug logging
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
            output_file = os.path.join(highlight_folder, f"highlight_{idx}_{highlight['event_type']}.mp4")
            
            command = [
                'ffmpeg', '-y',
                '-ss', str(highlight['start_frame'] / fps),
                '-i', video_path,
                '-t', str((highlight['end_frame'] - highlight['start_frame']) / fps),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                output_file
            ]
            
            subprocess.run(command)
    
    return len(highlights)

if __name__ == "__main__":
    video_path = '200_min.mp4'
    output_folder = 'extracted_details'
    
    print("Starting highlight extraction...")
    num_highlights = extract_highlights(video_path, output_folder, debug=True)
    
    print(f"\nExtraction complete!")
    print(f"Number of highlights extracted: {num_highlights}")
    print(f"Check the debug folder for detailed logs and intermediate images")