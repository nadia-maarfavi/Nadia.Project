import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import time
from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np
import subprocess
import cv2
import pandas as pd
from csv import writer

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Just to load the models
faces = RetinaFace.extract_faces("friends.jpg")
print("There are ", len(faces), " faces in the image")
for face in faces:
    obj = DeepFace.analyze(face, detector_backend='skip')
    print(obj)
    print(obj["age"], " years old ", obj["dominant_race"], " ", obj["dominant_emotion"], " ", obj["gender"],
          obj['emotion']['angry'],obj['emotion']['disgust'],obj['emotion']['fear'],obj['emotion']['happy'],obj['emotion']['sad'],obj['emotion']['surprise'],obj['emotion']['neutral'],
          obj['race']['asian'],obj['race']['indian'],obj['race']['black'],obj['race']['white'],obj['race']['middle eastern'],obj['race']['latino hispanic'],
          obj['region']['x'],obj['region']['y'],obj['region']['w'],obj['region']['h'])
    print("------------------------")


#This function access an entire video and return the frame type of all the frames
def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

#This function will call previous function and then for each keyframe it append it to the list
def save_i_keyframes(video_fn):
    alist = []
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    print("iframes len:  ", len(i_frames))
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        print("I frames are: ", i_frames)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            alist.append(frame)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

    arr = np.array(alist)
    return arr, i_frames

start = time.time()


#This function will iterate through each keyframe, call the DeepFace model to detect the face and facial attributes
#Also save the result of each video in a separate .csv file in a directory
#Each movie will have one file associate with it including the facial attributes of each keyframe in one row
def DetectFace(videopath, videoname,link):
    images, KeyframesNo = save_i_keyframes(videopath)

    filename = "....."+videoname +".csv"
    for i in range(images.shape[0]):
        keynumber = KeyframesNo[i]
        img = images[i]
        faces = RetinaFace.extract_faces(img)
        FaceNo = len(faces)
        for face in faces:
            OneRow = []
            obj = DeepFace.analyze(face, detector_backend='skip')
            OneRow.extend((videoname,link, i, keynumber, FaceNo, obj['age'],obj['gender'],obj["dominant_emotion"],obj["dominant_race"],
            obj['emotion']['angry'],obj['emotion']['disgust'],obj['emotion']['fear'],obj['emotion']['happy'],obj['emotion']['sad'],obj['emotion']['surprise'],obj['emotion']['neutral'],
            obj['race']['asian'],obj['race']['indian'],obj['race']['black'],obj['race']['white'],obj['race']['middle eastern'],obj['race']['latino hispanic'],
            obj['region']['x'],obj['region']['y'],obj['region']['w'],obj['region']['h']))
            #print("Len one row : ", len(OneRow))
            print("One Row  ", OneRow)
            with open(filename, 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(OneRow)
                f_object.close()



end = time.time()
#print("Time to detect face keyframes ", end-start)


# Read movies dataset and send each movie to the DetectFace function
videono = 1
dataset = pd.read_csv("/home/nmaarfavi/Face/SampleDataset.csv")
VideosName = dataset.VideoName.values
MoviesName = dataset.Movie.values
MovieLinks = dataset.Movie_Link.values
cur_dir = "/home/nmaarfavi/Face/DownloadVideo/Videos"

for i in range(len(VideosName)):
    starting_time = time.time()
    file_list = os.listdir(cur_dir)
    videopath = cur_dir + "/" + VideosName[i] +".mp4"
    print("Video Number: ", videono)
    print("Video Name: ", VideosName[i])
    if((os.path.isfile(videopath))==False):
        print("not found")
        videono += 1
        i += 1
    else:

        DetectFace(videopath,VideosName[i],MovieLinks[i])
        elapsed_time = time.time() - starting_time
        print("Time per video:  ", elapsed_time)
        videono +=1
        i += 1





