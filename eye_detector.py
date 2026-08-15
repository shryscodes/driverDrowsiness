# Driver Drowsiness Detection System
# File: eye_detector.py
# Purpose: Extract and display left and right eye landmarks.

import cv2

from camera import Camera
from face_detector import FaceDetector


# MediaPipe Face Landmarker landmark indexes.
#
# These indexes identify points around the eyes.
# We will use these points later for EAR calculation.

LEFT_EYE_LANDMARKS = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE_LANDMARKS = [
    362,
    385,
    387,
    263,
    373,
    380
]


class EyeDetector:

    def __init__(self):
        self.face_detector = FaceDetector()

    def get_eye_points(self, results, frame):

        left_eye = []
        right_eye = []

        if not results.face_landmarks:
            return left_eye, right_eye

        height, width, _ = frame.shape

        # We are using only the first detected face.
        face_landmarks = results.face_landmarks[0]

        # Extract left eye points.
        for index in LEFT_EYE_LANDMARKS:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            left_eye.append((x, y))

        # Extract right eye points.
        for index in RIGHT_EYE_LANDMARKS:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            right_eye.append((x, y))

        return left_eye, right_eye

    def draw_eye_points(
        self,
        frame,
        left_eye,
        right_eye
    ):

        # Draw left eye points.
        for point in left_eye:

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )

        # Draw right eye points.
        for point in right_eye:

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )

        return frame

    def close(self):

        self.face_detector.close()


def main():

    camera = Camera()
    eye_detector = EyeDetector()

    if not camera.open():
        eye_detector.close()
        return

    print("[INFO] Eye landmark detection started.")
    print("[INFO] Red points indicate the eye landmarks.")
    print("[INFO] Press Q to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Detect the face.
            results = eye_detector.face_detector.detect(frame)

            # Extract eye landmarks.
            left_eye, right_eye = eye_detector.get_eye_points(
                results,
                frame
            )

            # Draw the eye landmarks.
            frame = eye_detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            cv2.imshow(
                "Driver Drowsiness - Eye Landmarks",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\n[INFO] Program interrupted by user.")

    finally:

        eye_detector.close()
        camera.release()

# Driver Drowsiness Detection System
# File: eye_detector.py
# Purpose: Detect eye and mouth landmarks using MediaPipe.

import cv2
import mediapipe as mp


class EyeDetector:

    def __init__(
        self,
        model_path="face_landmarker.task"
    ):

        self.model_path = model_path

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        self.mp_tasks = (
            mp.tasks
        )

        self.mp_vision = (
            self.mp_tasks.vision
        )

        # ----------------------------------------------------
        # Face Landmarker options
        # ----------------------------------------------------

        options = (
            self.mp_vision
            .FaceLandmarkerOptions(
                base_options=(
                    self.mp_tasks.BaseOptions(
                        model_asset_path=model_path
                    )
                ),
                running_mode=(
                    self.mp_vision
                    .RunningMode.IMAGE
                ),
                num_faces=1
            )
        )

        # ----------------------------------------------------
        # Create face landmarker
        # ----------------------------------------------------

        self.face_landmarker = (
            self.mp_vision
            .FaceLandmarker
            .create_from_options(options)
        )

        print(
            "[INFO] Face landmark detector initialized."
        )

    # ========================================================
    # DETECT FACE
    # ========================================================

    def detect(self, frame):

        # MediaPipe expects RGB images.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = (
            mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame
            )
        )

        results = (
            self.face_landmarker.detect(
                mp_image
            )
        )

        return results

    # ========================================================
    # GET LANDMARK
    # ========================================================

    @staticmethod
    def get_point(
        results,
        index,
        frame
    ):

        if results is None:
            return None

        if not results.face_landmarks:
            return None

        landmarks = (
            results.face_landmarks[0]
        )

        if index >= len(landmarks):
            return None

        landmark = landmarks[index]

        height, width = (
            frame.shape[:2]
        )

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        return (x, y)

    # ========================================================
    # GET EYE POINTS
    # ========================================================

    def get_eye_points(
        self,
        results,
        frame
    ):

        # ----------------------------------------------------
        # MediaPipe Face Landmarker eye landmarks
        # ----------------------------------------------------

        left_eye_indices = [
            362,
            385,
            387,
            263,
            373,
            380
        ]

        right_eye_indices = [
            33,
            160,
            158,
            133,
            153,
            144
        ]

        left_eye = []

        right_eye = []

        for index in left_eye_indices:

            point = self.get_point(
                results,
                index,
                frame
            )

            if point is not None:

                left_eye.append(point)

        for index in right_eye_indices:

            point = self.get_point(
                results,
                index,
                frame
            )

            if point is not None:

                right_eye.append(point)

        return (
            left_eye,
            right_eye
        )

    # ========================================================
    # GET MOUTH POINTS
    # ========================================================

    def get_mouth_points(
        self,
        results,
        frame
    ):

        # ----------------------------------------------------
        # Mouth landmarks
        #
        # These points describe the mouth opening and width.
        # ----------------------------------------------------

        mouth_indices = [

            # Left corner
            61,

            # Upper-left
            185,

            # Upper-middle
            13,

            # Upper-right
            409,

            # Lower-right
            324,

            # Lower-middle
            14,

            # Lower-left
            95,

            # Right corner
            291
        ]

        mouth_points = []

        for index in mouth_indices:

            point = self.get_point(
                results,
                index,
                frame
            )

            if point is not None:

                mouth_points.append(
                    point
                )

        return mouth_points

    # ========================================================
    # DRAW EYE POINTS
    # ========================================================

    def draw_eye_points(
        self,
        frame,
        left_eye,
        right_eye
    ):

        for point in left_eye:

            cv2.circle(
                frame,
                point,
                2,
                (0, 255, 0),
                -1
            )

        for point in right_eye:

            cv2.circle(
                frame,
                point,
                2,
                (0, 255, 0),
                -1
            )

        return frame

    # ========================================================
    # DRAW MOUTH POINTS
    # ========================================================

    def draw_mouth_points(
        self,
        frame,
        mouth_points
    ):

        for point in mouth_points:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 255),
                -1
            )

        return frame

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.face_landmarker is not None:

            self.face_landmarker.close()

            self.face_landmarker = None

        print(
            "[INFO] Face landmark detector closed."
        )


# ============================================================
# TEST
# ============================================================

def main():

    from camera import Camera

    camera = Camera()

    detector = EyeDetector()

    if not camera.open():

        detector.close()

        return

    print()
    print("=" * 60)
    print(" EYE + MOUTH LANDMARK TEST")
    print("=" * 60)
    print()
    print("GREEN  = eye landmarks")
    print("PURPLE = mouth landmarks")
    print()
    print("Open and close your mouth to check the points.")
    print()
    print("Press Q to exit.")
    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            results = detector.detect(
                frame
            )

            # ------------------------------------------------
            # Eyes
            # ------------------------------------------------

            left_eye, right_eye = (
                detector.get_eye_points(
                    results,
                    frame
                )
            )

            frame = detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            # ------------------------------------------------
            # Mouth
            # ------------------------------------------------

            mouth_points = (
                detector.get_mouth_points(
                    results,
                    frame
                )
            )

            frame = detector.draw_mouth_points(
                frame,
                mouth_points
            )

            cv2.imshow(
                "Eye + Mouth Landmark Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "[INFO] Test completed."
        )


if __name__ == "__main__":

    main()
if __name__ == "__main__":
    main()  