import os
import subprocess

# Directory containing the videos
video_dir = "../data/urfunny_orig_videos"
new_video_dir = "../data/urfunny_videos"

# Iterate over all files in the directory
# Iterate over all files in the directory
for filename in os.listdir(video_dir):
    # Check if the file is a video (you might need to adjust this)
    if filename.endswith(".mp4"):
        # Full path to the video file
        video_path = os.path.join(video_dir, filename)

        # Use ffprobe to get the start_time
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=start_time -of default=noprint_wrappers=1:nokey=1 {video_path}"
        start_time = float(subprocess.check_output(cmd, shell=True))

        # Use ffmpeg to adjust the timestamps
        output_path = os.path.join(new_video_dir, filename)
        cmd = f"ffmpeg -i {video_path} -c copy -ss {start_time} {output_path}"
        subprocess.call(cmd, shell=True)

# output_path = os.path.join(new_video_dir, filename)