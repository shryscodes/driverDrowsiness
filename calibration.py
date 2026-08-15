# ============================================================
# DRIVER DROWSINESS DETECTION SYSTEM
# File: calibration.py
# Purpose: Calibrate the driver's natural open-eye EAR.
# ============================================================

import cv2
import time
import statistics

from camera import Camera
from eye_detector import EyeDetector
from ear import EARCalculator


class Calibrator:

    def __init__(self, duration=5):
        self.duration = duration

    def calibrate(self):

        camera = Camera()
        eye_detector = EyeDetector()
        ear_calculator = EARCalculator()

        if not camera.open():

            print("[ERROR] Camera could not be opened.")

            eye_detector.close()

            return None

        ear_values = []

        start_time = time.time()

        print()
        print("=" * 55)
        print(" EYE CALIBRATION")
        print("=" * 55)
        print()
        print("Keep your eyes naturally OPEN.")
        print("Look straight at the camera.")
        print(f"Calibration will run for {self.duration} seconds.")
        print()

        try:

            while True:

                success, frame = camera.read()

                if not success:

                    print(
                        "[ERROR] Could not read camera frame."
                    )

                    break

                elapsed_time = (
                    time.time() - start_time
                )

                if elapsed_time >= self.duration:
                    break

                # ------------------------------------------------
                # Detect face landmarks.
                #
                # IMPORTANT:
                # EyeDetector now owns the MediaPipe detector.
                # ------------------------------------------------

                results = eye_detector.detect(
                    frame
                )

                # ------------------------------------------------
                # Extract eye landmarks.
                # ------------------------------------------------

                left_eye, right_eye = (
                    eye_detector.get_eye_points(
                        results,
                        frame
                    )
                )

                # ------------------------------------------------
                # Calculate EAR for both eyes.
                # ------------------------------------------------

                left_ear = (
                    ear_calculator.calculate(
                        left_eye
                    )
                )

                right_ear = (
                    ear_calculator.calculate(
                        right_eye
                    )
                )

                # ------------------------------------------------
                # Only use valid measurements.
                # ------------------------------------------------

                if left_ear > 0 and right_ear > 0:

                    average_ear = (
                        left_ear + right_ear
                    ) / 2.0

                    ear_values.append(
                        average_ear
                    )

                    # Draw eye landmarks.
                    frame = (
                        eye_detector.draw_eye_points(
                            frame,
                            left_eye,
                            right_eye
                        )
                    )

                    cv2.putText(
                        frame,
                        f"EAR: {average_ear:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                # ------------------------------------------------
                # Calibration countdown.
                # ------------------------------------------------

                remaining = max(
                    0,
                    self.duration - elapsed_time
                )

                cv2.putText(
                    frame,
                    f"Time remaining: {remaining:.1f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Keep eyes naturally OPEN",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Driver Drowsiness - Calibration",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print(
                        "[INFO] Calibration cancelled."
                    )

                    return None

        finally:

            eye_detector.close()

            camera.release()

            cv2.destroyAllWindows()

        # --------------------------------------------------------
        # Validate collected measurements.
        # --------------------------------------------------------

        if len(ear_values) < 10:

            print()
            print(
                "[ERROR] Not enough valid eye measurements."
            )

            print(
                "Please try calibration again."
            )

            return None

        # --------------------------------------------------------
        # Calculate personalized baseline.
        # --------------------------------------------------------

        baseline_ear = (
            statistics.median(
                ear_values
            )
        )

        standard_deviation = (
            statistics.pstdev(
                ear_values
            )
        )

        # --------------------------------------------------------
        # Initial personalized threshold.
        #
        # This is a prototype parameter and should later be
        # experimentally evaluated.
        # --------------------------------------------------------

        threshold = (
            baseline_ear * 0.70
        )

        print()
        print("=" * 55)
        print(" CALIBRATION COMPLETE")
        print("=" * 55)
        print()
        print(
            f"Measurements collected : "
            f"{len(ear_values)}"
        )

        print(
            f"Open-eye baseline EAR  : "
            f"{baseline_ear:.3f}"
        )

        print(
            f"EAR standard deviation : "
            f"{standard_deviation:.3f}"
        )

        print(
            f"Eye-closure threshold  : "
            f"{threshold:.3f}"
        )

        print()
        print("=" * 55)

        return {
            "baseline_ear": baseline_ear,
            "standard_deviation": standard_deviation,
            "threshold": threshold
        }


def main():

    calibrator = Calibrator(
        duration=5
    )

    result = calibrator.calibrate()

    if result is None:
        return

    print()
    print(
        "[SUCCESS] Calibration data generated."
    )

    print()
    print(
        "The next module will use this threshold"
    )

    print(
        "to determine whether your eyes are OPEN or CLOSED."
    )


if __name__ == "__main__":
    main()