import cv2
import mediapipe as mp
import numpy as np


class Detector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmark = None
        self.width = 0
        self.height = 0

        self.prev_face_sp = None
        self.prev_face_ep = None

    def process(self, frame):
        # Set size
        self.height, self.width = frame.shape[:2]

        # process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return False

        # parsing
        self.landmark = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]
        return len(self.landmark) > 0

    def get_face_rect(self):
        if self.landmark is not None:
            cx, cy = self.landmark[0][0], self.landmark[0][1]
            br = self.landmark[7][0]
            bl = self.landmark[8][0]
            bu = self.landmark[1][1]
            bb = self.landmark[10][1]

            w = (max(br, bl) - min(br, bl)) * 1.5
            h = (max(bu, bb) - min(bu, bb)) * 2.5

            sx = int((cx - w / 2) * self.width)
            sy = int((cy - h / 2) * self.height)
            ex = sx + int(w * self.width)
            ey = sy + int(h * self.height)

            sp, ep = (max(sx, 0), max(sy, 0)), (min(ex, self.width), min(ey, self.height))

            return sp, ep
        return (0, 0), (1, 1)

    def get_torso_rect(self):
        if self.landmark is not None:
            sx = int(min(self.landmark[11][0], self.landmark[12][0]) * self.width)
            sy = int(self.height * (self.landmark[9][1] + self.landmark[10][1] + self.landmark[11][1] + self.landmark[12][1]) / 4)
            ex = int(max(self.landmark[11][0], self.landmark[12][0]) * self.width)
            ey = int(min(self.landmark[23][1], self.landmark[24][1]) * self.height)

            return (max(sx, 0), max(sy, 0)), (min(ex, self.width), min(ey, self.height))
        return (0, 0), (1, 1)
