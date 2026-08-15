# Driver Drowsiness Detection System
# File: ear.py
# Purpose: Calculate Eye Aspect Ratio (EAR).

import cv2
import math

from camera import Camera
from eye_detector import EyeDetector


class EARCalculator:

    @staticmethod
    def distance(point1, point2):
        """Calculate Euclidean distance between two points."""

        x1, y1 = point1
        x2, y2 = point2

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    @staticmethod
    def calculate(eye_points):
        """
        Calculate Eye Aspect Ratio.

        Eye points:

        P1 -------- P4
         \          /
          P2      P3
          P6      P5
         /          \
        """

        if len(eye_points) != 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_distance_1 = EARCalculator.distance(
            p2,
            p6
        )

        vertical_distance_2 = EARCalculator.distance(
            p3,
            p5
        )

        horizontal_distance = EARCalculator.distance(
            p1,
            p4
        )

        if horizontal_distance == 0:
            return 0.0

        ear = (
            vertical_distance_1 +
            vertical_distance_2
        ) / (2.0 * horizontal_distance)

        return ear


def main():

    camera = Camera()
    eye_detector = EyeDetector()
    ear_calculator = EARCalculator()

    if not camera.open():
        eye_detector.close()
        return

    print("[INFO] EAR calculation started.")
    print("[INFO] Blink normally and watch the EAR values.")
    print("[INFO] Press Q to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Detect facial landmarks.
            results = eye_detector.face_detector.detect(frame)

            # Extract eye landmarks.
            left_eye, right_eye = eye_detector.get_eye_points(
                results,
                frame
            )

            # Calculate EAR.
            left_ear = ear_calculator.calculate(
                left_eye
            )

            right_ear = ear_calculator.calculate(
                right_eye
            )

            # Average the two eyes.
            if left_ear > 0 and right_ear > 0:

                average_ear = (
                    left_ear +
                    right_ear
                ) / 2.0

            else:

                average_ear = 0.0

            # Draw eye landmarks.
            frame = eye_detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            # Display EAR values.
            cv2.putText(
                frame,
                f"Left EAR: {left_ear:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right EAR: {right_ear:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Average EAR: {average_ear:.3f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Driver Drowsiness - EAR",
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


if __name__ == "__main__":
    main()