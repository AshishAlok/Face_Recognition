import os
import numpy as np 
import cv2 as cv

IMG_SIZE = (500,500)
# resize frame
def resizeFrame(frame):
    dimension = IMG_SIZE
    return cv.resize(frame, dimension,interpolation = cv.INTER_AREA)
# rescale func if required
def rescaleFrame(frame,scale = 0.75):
    width = (int)(frame.shape[1]*scale)
    height = (int)(frame.shape[0]*scale)
    
    dimension = (width , height)
    return cv.resize(frame, dimension,interpolation = cv.INTER_AREA)

# this list will store the name of the characters 
# for whom the recgonizer is trained
people = []

DIR = r'charac'

# Append the name of the charcters to the list
for i in os.listdir(DIR):
    people.append(i)

# the characters scanned 
print(f'The list of people scanned {people}')

# creating face Detector
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# creating face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()


features = [] # contains images for training 
labels = []   # contains labels for image identification

# Function to train the recognizer
def create_training():
    for person in people:
        # path of images of every person in the list
        path = os.path.join(DIR, person)
        # label of each person is its index in the list
        label = people.index(person) 

        for img in os.listdir(path):
            
            # finding path of each images
            img_path = os.path.join(path,img)
            # reading each image
            img_array = cv.imread(img_path)
            # checking if the character has images or all images have been read
            if img_array is None:
                continue 
            # converting to gray scale 
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # list to store the detected face 
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12)
            
            # cropping and storing the face in features and its label in labels
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

# initialinzing the training

create_training()

print(f'total features = {len(features)}')
print(f'total labels = {len(labels)}')

print('Training done ---------------')

# converting features and labels to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# training face recognizer with features and labels
face_recognizer.train(features,labels)

# saving the training results in face_trained.yml
face_recognizer.save('face_trained.yml')

# also saving features and labels
np.save('features.npy', features)
np.save('labels.npy', labels)

# reading the training results from face_trained.yml
face_recognizer.read('face_trained.yml')

#--------------------------------------------------------------------
# function takes an image matrix or a frame and recognizes face
# a= 0 means iamge is passed and a== 1 means video frmaes are passed
def recognize_face(img,a):
    img = resizeFrame(img)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('gray',gray)
    # cv.imshow('img',img)


    face_rect = haar_cascade.detectMultiScale(gray,1.3,8)
    # print(f'num of face detected {len(face_rect)}')
    for (x,y,w,h) in face_rect:
        
        # slicing the detected face i.e region of intrest
        face_roi = gray[y:y+h, x:x+w]
        
        # making predictions
        label,confidence = face_recognizer.predict(face_roi)
        if(a==0):
            print(f'{people[label]} with a confidence of {confidence}')
        
        #putting text and marking the recognized face
        cv.putText(img,str(people[label]),(x,y+h),cv.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # displaying the recognized face
    cv.imshow('Recognized face',img)
#--------------------------------------------------------------------

a = 0
# initializing the test directory
test_dir = 'test'
# continue until right instruction is not received
while a!=1 and a!=2:
    print('For Testing an image enter 1 or for testing a video enter 2 : ')
    a =int(input())

    if(a==1):
        
        # list stores the characters to be tested
        test = []
        # finding and storing the characters from the test directory
        for i in os.listdir(test_dir):
            test.append(i)
            
        # for every character    
        for i in test:
            
            # creating the path to images of the character
            path = os.path.join(test_dir,i)
            
            # and for every image of that character 
            for imgs in os.listdir(path):
                # path of the image
                img_path = os.path.join(path,imgs)
                
                img = cv.imread(img_path)
                # pass the image to the recognoze face function
                recognize_face(img = img,a=0)
                print('Press any key board button to move to next image.')
                cv.waitKey(0)
                
        
    elif (a==2):
        # continue until right instruction is not received
        c=0   
        while c!=1 and c!=2:
            
            print('Enter 1 if you want to use the front camera or ')
            print('Enter 2 if you want to run the test on saved videos :')
        
            c = int (input())
            
            if (c==1):
                # capture the front camera
                cap = cv.VideoCapture(0)
                print('Press q to quit the video')
                while True:
                    isTrue,img = cap.read()
                    if not isTrue:
                        print('Video not read!! Exiting...')
                        break
                    
                    recognize_face(img= img,a=1)
                    
                    if cv.waitKey(1) & 0xFF==ord('q'):
                        cap.release()
                        break
            elif (c==2):
                # read every video in the videos_test directory
                videos = os.listdir('videos_test')
                for vid in videos:
                    vid_path = os.path.join('videos_test',vid)
                    
                    cap = cv.VideoCapture(vid_path)
                    
                    print('Press q to quit the video')
                    
                    while cap.isOpened():
                        isTrue, frame = cap.read()
                        
                        # if frame not read break out of the loop
                        if not isTrue:
                            print('Video ended! moving to next video..')
                            break
                        # pass the frame to recognize_face directory
                        recognize_face(img = frame,a=1)
                        
                        if cv.waitKey(20) & 0xFF==ord('q'):
                            break
                    
            else:
                print('Wrong choice!! \nEnter your choice again!')
            
    else:
        print('Wrong choice!! \nEnter your choice again!')
    

# destroy all the windows of OpenCV
cv.destroyAllWindows()
print('program Ended...')