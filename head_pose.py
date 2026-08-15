# ============================================================
# DRIVER DROWSINESS DETECTION SYSTEM
# File: head_pose.py
# Purpose: Detect left/right head tilt using eye landmarks.
# ============================================================

import cv2
import math
import time

from eye_detector import EyeDetector


class HeadPoseDetector:

    def __init__(
        self,
        tilt_threshold=15.0,
        tilt_duration=1.5
    ):

        self.tilt_threshold = tilt_threshold
        self.tilt_duration = tilt_duration

        self.eye_detector = EyeDetector()

        self.tilt_start_time = None
        self.is_tilted = False
        self.tilt_direction = "NORMAL"

    # ========================================================
    # CALCULATE HEAD ROLL ANGLE
    # ========================================================

    @staticmethod
    def calculate_roll_angle(
        left_eye,
        right_eye
    ):

        if left_eye is None or right_eye is None:
            return 0.0

        if len(left_eye) == 0 or len(right_eye) == 0:
            return 0.0

        # ----------------------------------------------------
        # Calculate the center of each eye.
        # ----------------------------------------------------

        left_x = sum(
            point[0]
            for point in left_eye
        ) / len(left_eye)

        left_y = sum(
            point[1]
            for point in left_eye
        ) / len(left_eye)

        right_x = sum(
            point[0]
            for point in right_eye
        ) / len(right_eye)

        right_y = sum(
            point[1]
            for point in right_eye
        ) / len(right_eye)

        # ----------------------------------------------------
        # Difference between eye centers.
        # ----------------------------------------------------

        dx = right_x - left_x
        dy = right_y - left_y

        if abs(dx) < 1e-6:
            return 0.0

        # ----------------------------------------------------
        # Calculate angle of the line joining both eyes.
        # ----------------------------------------------------

        raw_angle = math.degrees(
            math.atan2(dy, dx)
        )

        # ----------------------------------------------------
        # Normalize the angle.
        #
        # A horizontal eye line can mathematically be represented
        # as either 0 degrees or 180 degrees depending on the
        # direction of the vector.
        #
        # We always want:
        #
        # Straight head -> approximately 0 degrees
        # ----------------------------------------------------

        if raw_angle > 90.0:
            raw_angle -= 180.0

        elif raw_angle < -90.0:
            raw_angle += 180.0

        return raw_angle

    # ========================================================
    # DETECT HEAD TILT
    # ========================================================

    def update(self, frame):

        results = self.eye_detector.detect(
            frame
        )

        left_eye, right_eye = (
            self.eye_detector.get_eye_points(
                results,
                frame
            )
        )

        roll_angle = self.calculate_roll_angle(
            left_eye,
            right_eye
        )

        current_time = time.time()

        absolute_angle = abs(
            roll_angle
        )

        # ----------------------------------------------------
        # Check whether tilt exceeds threshold.
        # ----------------------------------------------------

        if absolute_angle >= self.tilt_threshold:

            if self.tilt_start_time is None:

                self.tilt_start_time = current_time

            tilt_duration = (
                current_time -
                self.tilt_start_time
            )

            if tilt_duration >= self.tilt_duration:

                self.is_tilted = True

                if roll_angle > 0:

                    self.tilt_direction = "RIGHT"

                else:

                    self.tilt_direction = "LEFT"

        else:

            self.tilt_start_time = None

            self.is_tilted = False

            self.tilt_direction = "NORMAL"

            tilt_duration = 0.0

        return {
            "roll_angle": roll_angle,
            "tilted": self.is_tilted,
            "direction": self.tilt_direction,
            "tilt_duration": tilt_duration,
            "left_eye": left_eye,
            "right_eye": right_eye
        }

    # ========================================================
    # DRAW STATUS
    # ========================================================

    def draw_status(
        self,
        frame,
        data
    ):

        roll_angle = data["roll_angle"]
        tilted = data["tilted"]
        direction = data["direction"]
        tilt_duration = data["tilt_duration"]

        # ----------------------------------------------------
        # Draw eye landmarks.
        # ----------------------------------------------------

        frame = self.eye_detector.draw_eye_points(
            frame,
            data["left_eye"],
            data["right_eye"]
        )

        # ----------------------------------------------------
        # Display angle.
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Head Tilt: {roll_angle:.1f} deg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Display status.
        # ----------------------------------------------------

        if tilted:

            cv2.putText(
                frame,
                f"HEAD TILTED: {direction}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"Duration: {tilt_duration:.1f}s",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Head Status: NORMAL",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return frame

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.eye_detector.close()


# ============================================================
# TEST
# ============================================================

def main():

    from camera import Camera

    camera = Camera()

    detector = HeadPoseDetector(
        tilt_threshold=15.0,
        tilt_duration=1.5
    )

    if not camera.open():

        detector.close()

        return

    print()
    print("=" * 60)
    print(" HEAD TILT DETECTION TEST")
    print("=" * 60)
    print()
    print("Keep your head straight.")
    print()
    print("Straight head should be approximately 0 degrees.")
    print()
    print("Then slowly tilt your head LEFT.")
    print()
    print("Return to normal.")
    print()
    print("Then slowly tilt your head RIGHT.")
    print()
    print("A tilt must remain for about 1.5 seconds")
    print("before it is reported as a head-tilt warning.")
    print()
    print("Press Q to exit.")
    print("=" * 60)
    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            data = detector.update(
                frame
            )

            frame = detector.draw_status(
                frame,
                data
            )

            cv2.imshow(
                "Head Tilt Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Test interrupted."
        )

    finally:

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "[INFO] Head tilt test completed."
        )


if __name__ == "__main__":

    main()