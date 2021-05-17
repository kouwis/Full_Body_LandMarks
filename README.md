# Full_Body_LandMarks

### Explained

In this repo used [Mediapipe](https://google.github.io/mediapipe/solutions/solutions.html) solutions in sections:

- Face Mesh
- Hands
- Pose

Togther and achieved all coordinates every point


![alt text](Landmarks.gif)


**Divide [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) into :-**

Face Mesh images points in folder 

- Left Eye
- Right Eye
- Nose
- Mouth
- Head

**Divide [Hands](https://google.github.io/mediapipe/solutions/hands.html) into :-**

- Fingers

**Divide [Pose](https://google.github.io/mediapipe/solutions/pose.html) into :-**

- Chest
- Arms
- Legs

![alt text](Project.jpg)



### Installation
```bash
pip install landmark-python

```
### Usage

```python

#Imports
import landmark as LM
import cv2 as cv

cap = cv.VideoCapture(1)
face_detector = LM.Face() #Make instance from class

while 1:
    _, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #Any function of them need two parameters (image, imageRGB)
    face_detector.Drawing(img, imgRGB)
    face_detector.Find_Points(img, imgRGB)  #Find_points function returns a list of all points in the class

    #To detect any part of the body you should call (Find_points) function
    #Any function of Them does not need parameters
    face_detector.Left_Eye()  #Any part of the body returns a list of points for this part

    cv.imshow("img", img)
    cv.waitKey(1)
```
