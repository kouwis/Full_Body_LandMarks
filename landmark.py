import mediapipe as mp


class FullBody_LM:
    def __init__(self):
        self.mpDrawing = mp.solutions.drawing_utils

        self.mpFace = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                      max_num_faces=1,
                                                      min_detection_confidence=0.5,
                                                      min_tracking_confidence=0.5
                                                      )
        self.mpHands = mp.solutions.hands.Hands(static_image_mode=False,
                                                max_num_hands=2,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5
                                                )
        self.mpBody = mp.solutions.pose.Pose(static_image_mode=False,
                                             upper_body_only=False,
                                             smooth_landmarks=True,
                                             min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5
                                             )


class Face(FullBody_LM):

    def Drawing(self, image, imageRGB, color=(200, 200, 100), thickness=1, circle_radius=1):
        self.results = self.mpFace.process(imageRGB)
        if self.results.multi_face_landmarks:
            for face in self.results.multi_face_landmarks:
                self.mpDrawing.draw_landmarks(image, face,
                                              mp.solutions.face_mesh.FACE_CONNECTIONS,
                                              self.mpDrawing.DrawingSpec(color,
                                                                         thickness=thickness,
                                                                         circle_radius=circle_radius
                                                                         )
                                              )
        return image

    def Find_Points(self, image, imageRGB):
        Points = []
        self.results = self.mpFace.process(imageRGB)
        if self.results.multi_face_landmarks:
            face = self.results.multi_face_landmarks[0]
            for id, lm in enumerate(face.landmark):
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                Points.append([id, cx, cy])

        def Left_Eye():
            if len(Points) != 0:
                LeftEyeXL, LeftEyeYL = Points[130][1], Points[130][2]
                LeftEyeXR, LeftEyeYR = Points[234][1], Points[234][2]
                LeftEyeXU, LeftEyeYU = Points[27][1], Points[27][2]
                LeftEyeXD, LeftEyeYD = Points[23][1], Points[23][2]

                ROI = [[130, LeftEyeXL, LeftEyeYL],
                       [234, LeftEyeXR, LeftEyeYR],
                       [27, LeftEyeXU, LeftEyeYU],
                       [23, LeftEyeXD, LeftEyeYD]]

                return ROI

        self.Left_Eye = Left_Eye

        def Right_Eye():
            if len(Points) != 0:
                RightEyeXL, RightEyeYL = Points[463][1], Points[463][2]
                RightEyeXR, RightEyeYR = Points[359][1], Points[359][2]
                RightEyeXU, RightEyeYU = Points[257][1], Points[257][2]
                RightEyeXD, RightEyeYD = Points[253][1], Points[253][2]

                ROI = [[463, RightEyeXL, RightEyeYL],
                       [359, RightEyeXR, RightEyeYR],
                       [257, RightEyeXU, RightEyeYU],
                       [253, RightEyeXD, RightEyeYD]]

                return ROI

        self.Right_Eye = Right_Eye

        def Mouth():
            if len(Points) != 0:
                MouthXL, MouthYL = Points[61][1], Points[61][2]
                MouthXR, MouthYR = Points[306][1], Points[306][2]
                MouthXU, MouthYU = Points[11][1], Points[11][2]
                MouthXD, MouthYD = Points[16][1], Points[16][2]

                ROI = [[61, MouthXL, MouthYL],
                       [306, MouthXR, MouthYR],
                       [11, MouthXU, MouthYU],
                       [16, MouthXD, MouthYD]]

                return ROI

        self.Mouth = Mouth

        def Nose():
            if len(Points) != 0:
                NoseXL, NoseYL = Points[100][1], Points[100][2]
                NoseXR, NoseYR = Points[371][1], Points[371][2]
                NoseXU, NoseYU = Points[168][1], Points[168][2]
                NoseXD, NoseYD = Points[2][1], Points[2][2]

                ROI = [[100, NoseXL, NoseYL],
                       [371, NoseXR, NoseYR],
                       [168, NoseXU, NoseYU],
                       [2, NoseXD, NoseYD]]

                return ROI

        self.Nose = Nose

        def Head():
            if len(Points) != 0:
                HeadXL, HeadYL = Points[103][1], Points[103][2]
                HeadXR, HeadYR = Points[298][1], Points[298][2]
                HeadXU, HeadYU = Points[338][1], Points[338][2]
                HeadXD, HeadYD = Points[336][1], Points[336][2]

                ROI = [[103, HeadXL, HeadYL],
                       [298, HeadXR, HeadYR],
                       [338, HeadXU, HeadYU],
                       [336, HeadXD, HeadYD]]

                return ROI

        self.Head = Head
        return Points


class Hands(FullBody_LM):

    def Drawing(self, image, imageRGB, color=(200, 200, 100), thickness=5, circle_radius=2):
        self.results = self.mpHands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(image, hand,
                                              mp.solutions.hands.HAND_CONNECTIONS,
                                              self.mpDrawing.DrawingSpec(color,
                                                                         thickness=thickness,
                                                                         circle_radius=circle_radius
                                                                         )
                                              )

    def Find_Points(self, image, imageRGB):
        Points = []
        self.results = self.mpHands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                Points.append([id, cx, cy])

        def Fingers():
            ROI = []
            if len(Points) != 0:
                for finger in range(1, 21, 4):
                    FingerUX, FingerUY = Points[finger][1], Points[finger][2]
                    FingerDX, FingerDY = Points[finger + 3][1], Points[finger + 3][2]

                    ROI.append([[finger, FingerUX, FingerUY],
                                [finger + 3, FingerDX, FingerDY]])

                return ROI

        self.Fingers = Fingers
        return Points


class Body(FullBody_LM):

    def Drawing(self, image, imageRGB, color=(200, 200, 100), thickness=2, circle_radius=2):
        self.results = self.mpBody.process(imageRGB)
        if self.results.pose_landmarks:
            self.mpDrawing.draw_landmarks(image, self.results.pose_landmarks,
                                          mp.solutions.pose.POSE_CONNECTIONS,
                                          self.mpDrawing.DrawingSpec(color,
                                                                     thickness=thickness,
                                                                     circle_radius=circle_radius
                                                                     )
                                          )
        return image

    def Find_Points(self, image, imageRGB):
        Points = []
        self.results = self.mpBody.process(imageRGB)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                Points.append([id, cx, cy])

        def Chest():
            ROI = []
            if len(Points) != 0:
                for chest in range(11, 24, +12):
                    ChestLX, ChestLY = Points[chest][1], Points[chest][2]
                    ChestRX, ChestRY = Points[chest + 1][1], Points[chest + 1][2]

                    ROI.append([[chest, ChestLX, ChestLY], [chest + 1, ChestRX, ChestRY]])

                return ROI

        self.Chest = Chest

        def Arms():
            ROI = []
            if len(Points) != 0:
                for arm in range(11, 13):
                    ArmLX, ArmLY = Points[arm][1], Points[arm][2]
                    ArmRX, ArmRY = Points[arm + 4][1], Points[arm + 4][2]

                    ROI.append([[arm, ArmLX, ArmLY], [arm + 4, ArmRX, ArmRY]])

                return ROI

        self.Arms = Arms

        def Legs():
            ROI = []
            if len(Points) != 0:
                for leg in range(23, 25):
                    LegLX, LegLY = Points[leg][1], Points[leg][2]
                    LegRX, LegRY = Points[leg + 4][1], Points[leg + 4][2]

                    ROI.append([[leg, LegLX, LegLY], [leg + 4, LegRX, LegRY]])

                return ROI

        self.Legs = Legs
        return Points
