from utils import Line
from moviepy.editor import VideoFileClip

if __name__ == '__main__':
	line = Line()
	raw_clip = VideoFileClip('project_video.mp4')
	processed_clip = raw_clip.fl_image(line.process_pipeline)
	processed_clip.write_videofile('processed_project_video.mp4', audio=False)