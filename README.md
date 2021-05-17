# Full_Body_LandMarks

### Explained

In this repo used [Mediapipe](https://google.github.io/mediapipe/solutions/solutions.html) solutions in sections:

- Face Mesh
- Hands
- Pose

![alt text](Landmarks.gif)


 Together, this will extract all coordinates points for any part of the body. In any class of them, there are two functions: 
- Find Points
- Drawing

**Find_Points** function has some of sub functions those are body parts after that returns a list of class points such as: Face points

1. Divide Find_Points function in class [Face](https://google.github.io/mediapipe/solutions/face_mesh.html) into:


	- Left Eye
	- Right Eye
	- Nose
	- Mouth
	- Head


2. Divide Find_Points function in class [Hands](https://google.github.io/mediapipe/solutions/hands.html) into:


	- Fingers


3. Divide Find_Points function in class [Body](https://google.github.io/mediapipe/solutions/pose.html) into:


	- Chest
	- Arms
	- Legs


**Drawing** function returns an image with labeled points.


### Output


In the face parts, It returns a 2d list the first for all face points and the second for point coordinates **otherwise** 
it returns a 3d list  the first for all class points, the second for the beginning and end point in the part, and the third for point coordinates 

### Diagram

![alt text](Project.jpg)



### Installation
```bash
pip install landmark-detection

```
### Usage

```python

#Imports
import landmark as LM
import cv2 as cv

cap = cv.VideoCapture(0)  # 0 as a default webcam
face_detector = LM.Face()  # Make instance from class

while 1:
    _, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Any function of them need two parameters (image, imageRGB)
    # color, thickness and circle_radius are default parameters
    face_detector.Drawing(img, imgRGB, color=(0, 110, 100), thickness=3, circle_radius=3)
    face_detector.Find_Points(img, imgRGB)

    # To detect any part of the body you should call (Find_points) function
    # Any function of Them does not need parameters
    print(face_detector.Left_Eye())

    cv.imshow("img", img)
    cv.waitKey(1)

```
