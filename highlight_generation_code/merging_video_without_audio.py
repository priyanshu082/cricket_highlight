import cv2
import os
import numpy as np

def merge_videos(input_folder, output_file):
    # Get all video files from the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    video_files.sort()  # Sort the files to ensure consistent order

    if not video_files:
        print("No video files found in the input folder.")
        return

    # Read the first video to get the properties
    first_video = cv2.VideoCapture(os.path.join(input_folder, video_files[0]))
    frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(first_video.get(cv2.CAP_PROP_FPS))
    first_video.release()

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process each video file
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(input_folder, video_file))
        print(f"Processing {video_file}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()

    # Release the output video writer
    out.release()
    print(f"All videos merged into {output_file}")

# Usage
input_folder = 'extracted_highlights'  # Folder containing the highlight videos
output_file = 'merged_highlights.mp4'  # Name of the output merged video
merge_videos(input_folder, output_file)