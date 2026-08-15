# Driver Drowsiness Detection System
# File: drowsiness_detector.py
# Purpose:
#   1. Detect prolonged eye closure
#   2. Detect yawning
#   3. Detect prolonged head tilt
#   4. Count drowsiness events

import cv2
import time

from eye_state import EyeStateDetector
from mar import MARCalculator
from head_pose import HeadPoseDetector


class DrowsinessDetector:

    def __init__(
        self,
        eye_threshold=0.205,
        closure_duration=2.0,
        yawning_threshold=0.42,
        yawning_duration=1.0,
        head_tilt_threshold=15.0,
        head_tilt_duration=1.5
    ):

        # ====================================================
        # CONFIGURATION
        # ====================================================

        self.eye_threshold = eye_threshold
        self.closure_duration = closure_duration

        self.yawning_threshold = yawning_threshold
        self.yawning_duration = yawning_duration

        self.head_tilt_threshold = head_tilt_threshold
        self.head_tilt_duration = head_tilt_duration

        # ====================================================
        # EYE DETECTION
        # ====================================================

        self.eye_state_detector = EyeStateDetector(
            threshold=self.eye_threshold
        )

        # ====================================================
        # MOUTH DETECTION
        # ====================================================

        self.mar_calculator = MARCalculator(
            yawning_threshold=self.yawning_threshold
        )

        # ====================================================
        # HEAD DETECTION
        # ====================================================

        self.head_pose_detector = HeadPoseDetector(
            tilt_threshold=self.head_tilt_threshold,
            tilt_duration=self.head_tilt_duration
        )

        # ====================================================
        # EYE CLOSURE STATE
        # ====================================================

        self.closed_start_time = None

        self.is_drowsy = False

        self.drowsiness_events = 0

        self.event_counted = False

        # ====================================================
        # YAWN STATE
        # ====================================================

        self.yawn_start_time = None

        self.is_yawning = False

        self.yawn_count = 0

        self.yawn_counted = False

        # ====================================================
        # HEAD TILT STATE
        # ====================================================

        self.is_head_tilted = False

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, frame):

        current_time = time.time()

        # ====================================================
        # 1. EYE DETECTION
        # ====================================================

        eye_data = (
            self.eye_state_detector.get_state(
                frame
            )
        )

        state = eye_data["state"]

        # ====================================================
        # EYE CLOSURE
        # ====================================================

        closed_duration = 0.0

        if state == "CLOSED":

            if self.closed_start_time is None:

                self.closed_start_time = current_time

                self.event_counted = False

            closed_duration = (
                current_time -
                self.closed_start_time
            )

            if closed_duration >= self.closure_duration:

                self.is_drowsy = True

                if not self.event_counted:

                    self.drowsiness_events += 1

                    self.event_counted = True

                    print(
                        "[DROWSINESS] Event #"
                        f"{self.drowsiness_events}"
                    )

        elif state == "OPEN":

            self.closed_start_time = None

            self.is_drowsy = False

            self.event_counted = False

        # ====================================================
        # 2. MOUTH / YAWN DETECTION
        # ====================================================

        mouth_points = (
            self.eye_state_detector
            .eye_detector
            .get_mouth_points(
                self.eye_state_detector.last_results,
                frame
            )
        )

        mar = self.mar_calculator.calculate(
            mouth_points
        )

        if mar >= self.yawning_threshold:

            if self.yawn_start_time is None:

                self.yawn_start_time = current_time

                self.yawn_counted = False

            yawn_duration = (
                current_time -
                self.yawn_start_time
            )

            if yawn_duration >= self.yawning_duration:

                self.is_yawning = True

                if not self.yawn_counted:

                    self.yawn_count += 1

                    self.yawn_counted = True

                    print(
                        "[YAWN] Yawn #"
                        f"{self.yawn_count}"
                    )

        else:

            self.yawn_start_time = None

            self.is_yawning = False

            self.yawn_counted = False

            yawn_duration = 0.0

        # ====================================================
        # 3. HEAD TILT DETECTION
        # ====================================================

        head_data = (
            self.head_pose_detector.update(
                frame
            )
        )

        self.is_head_tilted = (
            head_data["tilted"]
        )

        # ====================================================
        # OVERALL DROWSINESS CONDITION
        # ====================================================

        # A prolonged eye closure is the primary
        # drowsiness condition.

        # Yawning and head tilt are additional
        # indicators and are reported separately.

        # ====================================================
        # COMBINE DATA
        # ====================================================

        data = {

            # Eye information
            "state": state,

            "average_ear":
                eye_data["average_ear"],

            "left_ear":
                eye_data["left_ear"],

            "right_ear":
                eye_data["right_ear"],

            "left_eye":
                eye_data["left_eye"],

            "right_eye":
                eye_data["right_eye"],

            "closed_duration":
                closed_duration,

            # Drowsiness
            "drowsy":
                self.is_drowsy,

            "drowsiness_events":
                self.drowsiness_events,

            # Mouth
            "mar":
                mar,

            "yawning":
                self.is_yawning,

            "yawn_count":
                self.yawn_count,

            "mouth_points":
                mouth_points,

            # Head
            "roll_angle":
                head_data["roll_angle"],

            "head_tilted":
                self.is_head_tilted,

            "head_direction":
                head_data["direction"],

            "head_tilt_duration":
                head_data["tilt_duration"]
        }

        return data

    # ========================================================
    # DRAW STATUS
    # ========================================================

    def draw_status(self, frame, data):

        # ====================================================
        # DRAW EYE POINTS
        # ====================================================

        frame = (
            self.eye_state_detector
            .eye_detector
            .draw_eye_points(
                frame,
                data["left_eye"],
                data["right_eye"]
            )
        )

        # ====================================================
        # DRAW MOUTH POINTS
        # ====================================================

        frame = (
            self.eye_state_detector
            .eye_detector
            .draw_mouth_points(
                frame,
                data["mouth_points"]
            )
        )

        # ====================================================
        # EAR
        # ====================================================

        cv2.putText(
            frame,
            f"EAR: {data['average_ear']:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ====================================================
        # EYE STATE
        # ====================================================

        cv2.putText(
            frame,
            f"Eye State: {data['state']}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # ====================================================
        # CLOSED TIME
        # ====================================================

        cv2.putText(
            frame,
            f"Closed: {data['closed_duration']:.2f}s",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # ====================================================
        # MAR
        # ====================================================

        cv2.putText(
            frame,
            f"MAR: {data['mar']:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2
        )

        # ====================================================
        # YAWN
        # ====================================================

        yawn_status = (
            "YES"
            if data["yawning"]
            else "NO"
        )

        cv2.putText(
            frame,
            f"Yawning: {yawn_status}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2
        )

        # ====================================================
        # YAWN COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Yawns: {data['yawn_count']}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # HEAD ANGLE
        # ====================================================

        cv2.putText(
            frame,
            f"Head Tilt: {data['roll_angle']:.1f} deg",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # ====================================================
        # HEAD STATUS
        # ====================================================

        if data["head_tilted"]:

            head_text = (
                "Head: TILTED "
                + data["head_direction"]
            )

            head_color = (0, 0, 255)

        else:

            head_text = "Head: NORMAL"

            head_color = (0, 255, 0)

        cv2.putText(
            frame,
            head_text,
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            head_color,
            2
        )

        # ====================================================
        # DROWSINESS EVENTS
        # ====================================================

        cv2.putText(
            frame,
            f"Drowsiness Events: "
            f"{data['drowsiness_events']}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # OVERALL STATUS
        # ====================================================

        if data["drowsy"]:

            status = "DROWSINESS DETECTED!"

            status_color = (0, 0, 255)

        elif data["yawning"]:

            status = "YAWNING DETECTED"

            status_color = (0, 165, 255)

        elif data["head_tilted"]:

            status = "HEAD TILT WARNING"

            status_color = (0, 165, 255)

        else:

            status = "STATUS: ALERT"

            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (20, 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            status_color,
            3
        )

        return frame

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.eye_state_detector.close()

        self.head_pose_detector.close()


# ============================================================
# TEST
# ============================================================

def main():

    from camera import Camera

    camera = Camera()

    detector = DrowsinessDetector(
        eye_threshold=0.205,
        closure_duration=2.0,
        yawning_threshold=0.42,
        yawning_duration=1.0,
        head_tilt_threshold=15.0,
        head_tilt_duration=1.5
    )

    if not camera.open():

        detector.close()

        return

    print()
    print("=" * 65)
    print(" COMPLETE DROWSINESS ANALYSIS TEST")
    print("=" * 65)
    print()
    print("Testing:")
    print("  1. Eye closure")
    print("  2. Yawning")
    print("  3. Head tilt")
    print()
    print("Press Q to exit.")
    print("=" * 65)
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
                "Complete Drowsiness Analysis",
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
            f"[INFO] Drowsiness events: "
            f"{detector.drowsiness_events}"
        )

        print(
            f"[INFO] Yawns detected: "
            f"{detector.yawn_count}"
        )

        print(
            "[INFO] Head-pose detection completed."
        )


if __name__ == "__main__":

    main()