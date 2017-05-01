from lane import *
import glob
from moviepy.editor import VideoFileClip

calib_images = glob.glob('camera_cal/calibration*.jpg')

test_img = cv2.imread(calib_images[0])
lane = Lane.from_calib_images(test_img.shape, calib_images)

output = "project_video_result.mp4"
clip = VideoFileClip("project_video.mp4")
clip.fl_image(lane.draw_lane).write_videofile(output, audio=None)

