import os
import time
import cv2
import numpy as np

def capture_samples(FRAMES):
    TIMEBETWEEN = 6
    frameCount = 0
    subject_no = len([folder for folder in os.listdir('data')])
    os.makedirs('data/s%s'%(str(subject_no)))
    while frameCount < FRAMES:
        imageNumber = str(frameCount).zfill(7)
        os.system("raspistill -o data/s%s/image%s.jpg"%(str(subject_no), imageNumber))
        frameCount += 1
        time.sleep(TIMEBETWEEN - 6) #Takes roughly 6 seconds to take a picture

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/lbpcascades/lbpcascade_frontalface.xml')
 
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
 
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
 
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
 
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
 
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
 
    #let's go through each directory and read images within it
    for dir_name in dirs:
        #print dir_name
 
    #our subject directories start with letter 's' so
    #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
 
    #------STEP-2--------
    #extract label number of subject from dir_name
    #format of dir name = slabel
    #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        #print label
 
        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

 
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        
 
    #------STEP-3--------
    #go through each image name, read image, 
    #detect face and add face to list of faces
        for image_name in subject_images_names:
            #print image_name
 
    #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
 
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
 
            #read image
            image = cv2.imread(image_path)
            #image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
 
            #display an image window to show the image
            #cv2.namedWindow("Training on image...")
            #cv2.moveWindow("Training on image...", 40, 30)
            #cv2.imshow("Training on image...", image)
            #cv2.waitKey(100)
 
            #detect face
            face, rect = detect_face(image)
            #print face, rect
 
#------STEP-4--------
    #for the purpose of this tutorial
    #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
 
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
 
    return faces, labels

def train():
    faces, labels = prepare_training_data('/home/pi/Hack/train/people')
    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save('model.xml')
    
