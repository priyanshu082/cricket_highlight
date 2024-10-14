import moviepy.editor as mp
from moviepy.video.compositing.concatenate import concatenate_videoclips

def generate_highlights(input_video, output_video, peak_ranges, buffer_seconds=1):
    # Load the video file
    video = mp.VideoFileClip(input_video)
    
    # Function to extract a clip with a buffer
    def extract_clip_with_buffer(start, end):
        clip_start = max(0, start - buffer_seconds)
        clip_end = min(video.duration, end + buffer_seconds)
        return video.subclip(clip_start, clip_end)
    
    # Extract clips for each peak range
    highlight_clips = []
    for peak_range in peak_ranges:
        if len(peak_range) >= 2:
            start = peak_range[0]
            end = peak_range[-1]
            highlight_clips.append(extract_clip_with_buffer(start, end))
    
    # Concatenate all highlight clips
    final_video = concatenate_videoclips(highlight_clips)
    
    # Write the final video to file
    final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")
    
    # Close the video to free up resources
    video.close()
    
    print(f"Highlight video saved as {output_video}")

# Example usage
input_video = r"D:\Git_main\cricket_highlight\audioProcessing\10min.mp4"
output_video = r"D:\Git_main\cricket_highlight\audioProcessing\highlight.mp4"
peak_ranges = [
    [40.85551020408163, 40.867120181405895, 40.87873015873016, 40.89034013605442, 40.90195011337868, 40.91356009070295],
    [190.8912471655329, 190.90285714285713, 190.9144671201814, 190.92607709750567, 190.93768707482994, 190.9492970521542],
    [293.15192743764175, 293.16353741496596, 293.1751473922902],
    [407.2083446712018],
    [447.18149659863946],
    [477.64607709750567, 477.65768707482994, 477.6692970521542],
    [541.9421315192744, 541.9537414965987, 541.9653514739229],
    [550.4986848072563, 550.5102947845805, 550.5219047619048, 550.5335147392291]
]

generate_highlights(input_video, output_video, peak_ranges)