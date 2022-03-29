import datetime
import ffmpeg
import pandas as pd
import os
import folium

video1 = ffmpeg.probe('/Users/Ambro/Desktop/video1.mp4')
video2 = ffmpeg.probe('/Users/Ambro/Desktop/PXL_20210716_101954152_0.mp4')
videos = [video1, video2]

fecha = os.stat('/Users/Ambro/Desktop/video1.mp4')

for i in videos:
    video_info = next(stream for stream in i['streams'] if stream['codec_type'] == 'video')
    #print(i)
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frame = int(video_info['nb_frames'])
    time = (video_info['duration'])
    dt = datetime.datetime.fromtimestamp(fecha.st_birthtime)
    
#print('width: ' + str(width) + ', height: ' + str(height) + ', fps: ' + str(round(num_frame/float(time), 2)) + ', duration: ' + str(time))
#print(video_info)

values = {
    "width [pixels]": width,
    "height [pixels]": height,
    "fps": round(num_frame/float(time), 2),
    "duration [seg]": round(float(time), 2),
    "creation_time:": dt
}
#table = pd.DataFrame(values, index = [str(video_info['id'])])
#print(table)

m = folium.Map(location = [41.38067355754285, 2.1916822645479432], zoom_start=12)
m.save("foto.html")