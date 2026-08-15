# Driver Drowsiness Detection System
# File: face_detector.py
# Purpose: Detect facial landmarks using MediaPipe Tasks.

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import Camera


class FaceDetector:

    def __init__(self):

        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

        self.frame_timestamp = 0

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.frame_timestamp += 1

        results = self.detector.detect_for_video(
            mp_image,
            self.frame_timestamp
        )

        return results

    def draw_landmarks(self, frame, results):

        if not results.face_landmarks:
            return frame

        height, width, _ = frame.shape

        for face_landmarks in results.face_landmarks:

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        return frame

    def close(self):

        self.detector.close()


def main():

    camera = Camera()
    detector = FaceDetector()

    if not camera.open():
        detector.close()
        return

    print("[INFO] Face landmark detection started.")
    print("[INFO] Press Q to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            results = detector.detect(frame)

            frame = detector.draw_landmarks(
                frame,
                results
            )

            cv2.imshow(
                "Driver Drowsiness - Face Landmarks",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\n[INFO] Program interrupted by user.")

    finally:

        detector.close()
        camera.release()


if __name__ == "__main__":
    main()