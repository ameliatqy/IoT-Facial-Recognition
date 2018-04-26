import cv2
import os

subjects = ['John', 'Amelia', 'George']
SDM_Lab = ['John', 'Amelia']

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

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

def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    
 
    #predict the image using our face recognizer
    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.load('model.xml')
    label = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = str(label)
    print label
 
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
 
    return img, label_text

def Access():
    #load test images
    os.system("raspistill -o test_image.jpg")
    test_img1 = cv2.imread("test_image.jpg")
    #test_img1 = cv2.imread("/home/pi/Hack/train/people/d0/image0000016.jpg")
    #cv2.imshow('GP', test_img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #perform a prediction
    predicted_img1, label_txt = predict(test_img1)
    filename = subjects[int(label_txt)] + '.jpg'

    if subjects[int(label_txt)] in SDM_Lab:
        print "True"
        return True
    else:
        print "False"
        return False

    #display both images
    predicted_img1 = cv2.resize(predicted_img1, (0,0), fx=0.2, fy=0.2)
    cv2.imwrite(filename, predicted_img1)
    cv2.imshow(label_txt, predicted_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

