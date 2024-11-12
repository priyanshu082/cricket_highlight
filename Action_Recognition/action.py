import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image

class ActionHighlightExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        # Load pre-trained model
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cricket-specific action labels
        self.cricket_actions = {
            'batting': 0,
            'bowling': 1,
            'fielding': 2,
            'celebration': 3
        }

    def preprocess_frames(self, frames):
        """Preprocess frames for the model"""
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Apply transforms
            processed = self.transform(pil_image)
            processed_frames.append(processed)
            
        # Stack frames along the time dimension and permute to [channels, frames, height, width]
        processed_frames = torch.stack(processed_frames)
        return processed_frames.permute(1, 0, 2, 3)

    def detect_cricket_action(self, frames_tensor):
        """Detect cricket-specific actions in frames"""
        with torch.no_grad():
            # Add batch dimension and reorder to [batch_size, channels, frames, height, width]
            frames_tensor = frames_tensor.unsqueeze(0)
            
            # Get model predictions
            predictions = self.model(frames_tensor)
            
            # Convert to probabilities
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            
            return probabilities.squeeze()

    def detect_highlights(self, confidence_threshold=0.7, clip_length=16, stride=8):
        """Detect highlights based on action recognition"""
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        highlight_timestamps = []
        frames_buffer = []
        frame_count = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            frames_buffer.append(frame)
            
            # Process when we have enough frames
            if len(frames_buffer) == clip_length:
                # Get current timestamp
                timestamp = frame_count / fps
                
                # Preprocess frames
                frames_tensor = self.preprocess_frames(frames_buffer)
                
                # Get action predictions
                probabilities = self.detect_cricket_action(frames_tensor)
                
                # Check if any interesting action is detected
                max_prob, action_idx = torch.max(probabilities, dim=0)
                
                if max_prob > confidence_threshold:
                    highlight_timestamps.append(timestamp)
                
                # Remove old frames based on stride
                frames_buffer = frames_buffer[stride:]
            
            frame_count += 1
        
        video.release()
        return highlight_timestamps

    def create_highlight_video(self, output_path, buffer_seconds=5):
        """Create highlight video from detected moments"""
        # Get highlight timestamps
        highlights = self.detect_highlights()
        
        # Load video
        video = VideoFileClip(self.video_path)
        highlight_clips = []
        
        # Extract clips around each highlight
        for timestamp in highlights:
            start_time = max(0, timestamp - buffer_seconds)
            end_time = min(video.duration, timestamp + buffer_seconds)
            
            clip = video.subclip(start_time, end_time)
            highlight_clips.append(clip)
        
        if highlight_clips:
            # Concatenate all highlight clips
            final_video = concatenate_videoclips(highlight_clips)
            # Write to file
            final_video.write_videofile(output_path, codec='libx264')
            final_video.close()
        
        video.close()

    def analyze_action_distribution(self):
        """Analyze distribution of detected actions"""
        video = cv2.VideoCapture(self.video_path)
        action_counts = {action: 0 for action in self.cricket_actions.keys()}
        frames_buffer = []
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            frames_buffer.append(frame)
            
            if len(frames_buffer) == 16:
                frames_tensor = self.preprocess_frames(frames_buffer)
                probabilities = self.detect_cricket_action(frames_tensor)
                _, action_idx = torch.max(probabilities, dim=0)
                
                # Increment the count for detected action
                for action, index in self.cricket_actions.items():
                    if action_idx == index:
                        action_counts[action] += 1
                
                frames_buffer = frames_buffer[8:]  # Move forward by the stride

        video.release()
        return action_counts

if __name__ == "__main__":
    # Define paths
    video_path = "3_min.mp4"  # Replace with actual video file path
    output_path = "output_highlight.mp4"  # Replace with desired output path
    
    # Initialize extractor and create highlights
    extractor = ActionHighlightExtractor(video_path)
    extractor.create_highlight_video(output_path)
