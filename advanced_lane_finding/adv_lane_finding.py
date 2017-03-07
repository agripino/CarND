from lane_finder import LaneFinder
from moviepy.editor import VideoFileClip


# A LaneFinder object is callable
adv_lane_finder = LaneFinder(n_last=15)

input_path = "./videos/project_video.mp4"
output_path = "./videos/project_video_annotated_2.mp4"

clip = VideoFileClip(input_path)
output_video = clip.fl_image(adv_lane_finder)

# Write annotated video
output_video.write_videofile(output_path, audio=False)
