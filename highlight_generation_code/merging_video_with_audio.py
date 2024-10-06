import os
import subprocess

def merge_videos(input_folder, output_file):
    # Get all video files from the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    video_files.sort()  # Sort the files to ensure consistent order

    if not video_files:
        print("No video files found in the input folder.")
        return

    # Create a temporary text file listing all videos to be merged
    with open("videos_to_merge.txt", "w") as file:
        for video in video_files:
            file.write(f"file '{os.path.join(input_folder, video)}'\n")

    # Use ffmpeg to merge the videos
    command = [
        'ffmpeg', '-y',  # Overwrite output if it exists
        '-f', 'concat',  # Concatenate the files
        '-safe', '0',  # Set to 0 to allow unsafe file paths
        '-i', 'videos_to_merge.txt',  # Input file list
        '-c', 'copy',  # Copy codec to preserve audio and video without re-encoding
        output_file  # Output merged file
    ]

    print(f"Merging videos into {output_file}...")
    subprocess.run(command)
    print(f"All videos merged into {output_file}")

    # Remove the temporary file
    os.remove("videos_to_merge.txt")

# Usage
input_folder = 'extracted_details/highlights'  # Folder containing the highlight videos
output_file = 'merged_highlights.mp4'  # Name of the output merged video
merge_videos(input_folder, output_file)
