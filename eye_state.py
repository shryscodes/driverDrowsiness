# Driver Drowsiness Detection System
# File: eye_state.py
# Purpose: Determine whether the eyes are OPEN or CLOSED.

from eye_detector import EyeDetector
from ear import EARCalculator


class EyeStateDetector:

    def __init__(
        self,
        threshold=0.205
    ):

        self.threshold = threshold

        self.eye_detector = EyeDetector()

        self.ear_calculator = EARCalculator()

        self.last_results = None

    # ========================================================
    # GET EYE STATE
    # ========================================================

    def get_state(self, frame):

        results = self.eye_detector.detect(
            frame
        )

        self.last_results = results

        left_eye, right_eye = (
            self.eye_detector.get_eye_points(
                results,
                frame
            )
        )

        left_ear = (
            self.ear_calculator.calculate(
                left_eye
            )
        )

        right_ear = (
            self.ear_calculator.calculate(
                right_eye
            )
        )

        # ----------------------------------------------------
        # Calculate average EAR
        # ----------------------------------------------------

        if left_ear > 0 and right_ear > 0:

            average_ear = (
                left_ear +
                right_ear
            ) / 2.0

        else:

            average_ear = 0.0

        # ----------------------------------------------------
        # Determine eye state
        # ----------------------------------------------------

        if average_ear == 0:

            state = "UNKNOWN"

        elif average_ear < self.threshold:

            state = "CLOSED"

        else:

            state = "OPEN"

        return {

            "state": state,

            "average_ear": average_ear,

            "left_ear": left_ear,

            "right_ear": right_ear,

            "left_eye": left_eye,

            "right_eye": right_eye

        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.eye_detector.close()