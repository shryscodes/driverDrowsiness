# Driver Drowsiness Detection System
# File: mar.py
# Purpose: Calculate Mouth Aspect Ratio (MAR)
#          for yawning detection.

import math


class MARCalculator:

    def __init__(
        self,
        yawning_threshold=0.60
    ):

        self.yawning_threshold = yawning_threshold

    # ========================================================
    # DISTANCE BETWEEN TWO LANDMARKS
    # ========================================================

    @staticmethod
    def distance(point1, point2):

        x1, y1 = point1
        x2, y2 = point2

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    # ========================================================
    # CALCULATE MAR
    # ========================================================

    def calculate(self, mouth_points):

        if mouth_points is None:
            return 0.0

        if len(mouth_points) < 6:
            return 0.0

        # ----------------------------------------------------
        # Expected mouth points:
        #
        # 0 → left corner
        # 1 → upper-left inner point
        # 2 → upper-middle point
        # 3 → upper-right inner point
        # 4 → lower-right inner point
        # 5 → lower-middle point
        # 6 → lower-left inner point
        # 7 → right corner
        #
        # The calculation uses vertical mouth opening
        # divided by mouth width.
        # ----------------------------------------------------

        left_corner = mouth_points[0]

        upper_middle = mouth_points[2]

        right_corner = mouth_points[7]

        lower_middle = mouth_points[5]

        upper_left = mouth_points[1]

        lower_left = mouth_points[6]

        upper_right = mouth_points[3]

        lower_right = mouth_points[4]

        # ----------------------------------------------------
        # Vertical distances
        # ----------------------------------------------------

        vertical_center = self.distance(
            upper_middle,
            lower_middle
        )

        vertical_left = self.distance(
            upper_left,
            lower_left
        )

        vertical_right = self.distance(
            upper_right,
            lower_right
        )

        # ----------------------------------------------------
        # Horizontal mouth width
        # ----------------------------------------------------

        horizontal_width = self.distance(
            left_corner,
            right_corner
        )

        if horizontal_width == 0:

            return 0.0

        # ----------------------------------------------------
        # MAR formula
        # ----------------------------------------------------

        mar = (
            vertical_center +
            vertical_left +
            vertical_right
        ) / (
            3.0 * horizontal_width
        )

        return mar

    # ========================================================
    # CHECK WHETHER MOUTH IS OPEN
    # ========================================================

    def is_yawning(self, mar):

        return mar >= self.yawning_threshold


# ============================================================
# SIMPLE MAR TEST
# ============================================================

def main():

    print()
    print("=" * 55)
    print(" MOUTH ASPECT RATIO TEST")
    print("=" * 55)
    print()

    calculator = MARCalculator()

    print(
        "MAR calculator created successfully."
    )

    print(
        f"Yawning threshold: "
        f"{calculator.yawning_threshold:.2f}"
    )

    print()
    print(
        "This module will be connected to the"
    )
    print(
        "MediaPipe face landmarks in the next step."
    )

    print()
    print(
        "[SUCCESS] MAR module is ready."
    )


if __name__ == "__main__":

    main()