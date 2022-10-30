# Move to proper directory
import os 
ROOT_DIR = os.getcwd()
os.path.join(ROOT_DIR)
print(ROOT_DIR)
os.chdir(ROOT_DIR)
ROOT_DIR = {""}
os.system("cd {ROOT_DIR}")

import torch
import cv2
import numpy as np
import pandas as pd
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn

import cv2 

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename=f'{ROOT_DIR}/data/images/test.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread(f'{ROOT_DIR}/data/images/test.jpg', cv2.COLOR_BGR2RGB)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Image saved!")
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

path_to_file=ROOT_DIR + '/output/bbox_file.txt'
if not os.path.exists(ROOT_DIR + '/output'):
  os.mkdir(ROOT_DIR + '/output')
with open(path_to_file, 'w'):
    pass
big_image_list=[]
big_face_id=[]
big_bbox_list=[]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_size=128
test_transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((image_size,image_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5],
                                                          [0.5,0.5,0.5])])


##loading yolo weights
#options available : 5l-face.pt,5m-face,5s-face.pt
weight='face5.pt'

root_data_path=ROOT_DIR + '/data'
dataset_path=root_data_path+'/images' ## set path of to-be-tested images
temp_storage=root_data_path+'/temp'
cropped_faces_path=temp_storage+'/faces'
_,_,target_image_files=next(os.walk(dataset_path))

#create a text file to write bboxes
for image_file in target_image_files:#takes 1 image
    if not os.path.exists(cropped_faces_path):
      os.mkdir(temp_storage)
      os.mkdir(cropped_faces_path)
    original_image_file_path=dataset_path+'/'+image_file
    before_image=cv2.imread(original_image_file_path)
    before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
    
  # face detector
    os.system("cd {ROOT_DIR}")
    os.system("python detect.py --weights weights/{weight} --image {original_image_file_path}")
# os.remove(original_image_file_path)

import face_recognition

file = open(ROOT_DIR + '/dataset/Identities.txt','r')
image = []
face_encoding = []
names = []
i = 0
for name in file.read().splitlines():
    print("Identified:",name)
    names.append(name)
    image.append(face_recognition.load_image_file(f"dataset/{name}/{name}.jpg"))
    face_encoding.append(face_recognition.face_encodings(image[i])[0])
    i += 1
known_face_encodings = face_encoding
known_face_names = names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
crop_path = f'{ROOT_DIR}/data/temp/faces/test_1.jpg'
frame = cv2.imread(crop_path)

# Resize frame of video to 1/4 size for faster face recognition processing
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_frame = small_frame[:, :, ::-1]

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)


# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the resulting image

cv2.imshow('Identity', frame)
cv2.waitKey(5000)
cv2.destroyAllWindows()