# ============================================================
# DRIVER DROWSINESS DETECTION SYSTEM
# File: main.py
# Purpose: Run the complete integrated system.
# ============================================================

import cv2

from camera import Camera
from calibration import Calibrator
from drowsiness_detector import DrowsinessDetector
from alarm import Alarm


def main():

    print()
    print("=" * 65)
    print(" DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 65)
    print()

    # ========================================================
    # STEP 1 - CALIBRATION
    # ========================================================

    print("[INFO] Starting eye calibration...")

    calibrator = Calibrator(
        duration=5
    )

    calibration_result = calibrator.calibrate()

    if calibration_result is None:

        print(
            "[ERROR] Calibration failed."
        )

        return

    # --------------------------------------------------------
    # Get personalized EAR threshold.
    # --------------------------------------------------------

    eye_threshold = calibration_result[
        "threshold"
    ]

    print()
    print(
        "[INFO] Calibration successful."
    )

    print(
        f"[INFO] Using EAR threshold: "
        f"{eye_threshold:.3f}"
    )

    # ========================================================
    # STEP 2 - CAMERA
    # ========================================================

    camera = Camera()

    if not camera.open():

        print(
            "[ERROR] Could not open camera."
        )

        return

    print(
        "[INFO] Camera opened successfully."
    )

    # ========================================================
    # STEP 3 - DROWSINESS DETECTOR
    # ========================================================

    detector = DrowsinessDetector(

        eye_threshold=eye_threshold,

        closure_duration=2.0,

        # Your measured values:
        # Normal mouth ≈ 0.30
        # Wide yawn   ≈ 0.50
        yawning_threshold=0.42,

        yawning_duration=1.0,

        # Head tilt
        head_tilt_threshold=15.0,

        head_tilt_duration=1.5
    )

    # ========================================================
    # STEP 4 - ALARM
    # ========================================================

    alarm = Alarm()

    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    print()
    print("=" * 65)
    print(" COMPLETE DETECTION STARTED")
    print("=" * 65)
    print()
    print("Detection modules:")
    print("  [1] Eye closure / EAR")
    print("  [2] Yawning / MAR")
    print("  [3] Head tilt")
    print("  [4] Drowsiness events")
    print("  [5] Alarm")
    print()
    print("Current configuration:")
    print()
    print(
        f"  EAR threshold       : "
        f"{eye_threshold:.3f}"
    )

    print(
        "  Eye closure time    : "
        "2.0 seconds"
    )

    print(
        "  Yawn MAR threshold  : "
        "0.42"
    )

    print(
        "  Yawn duration       : "
        "1.0 second"
    )

    print(
        "  Head tilt threshold : "
        "15.0 degrees"
    )

    print(
        "  Head tilt duration  : "
        "1.5 seconds"
    )

    print()
    print("Normal blink       -> No alarm")
    print("Eyes closed > 2 s  -> Drowsiness warning")
    print("Yawn > 1 s         -> Yawning warning")
    print("Head tilt > 15°    -> Head tilt warning")
    print()
    print("Press Q to exit.")
    print("=" * 65)
    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Read camera frame
            # ------------------------------------------------

            success, frame = camera.read()

            if not success:

                print(
                    "[ERROR] Failed to read camera frame."
                )

                break

            # ------------------------------------------------
            # Run all detection modules
            # ------------------------------------------------

            data = detector.update(
                frame
            )

            # ------------------------------------------------
            # Draw complete status
            # ------------------------------------------------

            frame = detector.draw_status(
                frame,
                data
            )

            # ------------------------------------------------
            # Alarm logic
            # ------------------------------------------------

            if data["drowsy"]:

                alarm.start()

            elif data["yawning"]:

                alarm.start()

            elif data["head_tilted"]:

                alarm.start()

            else:

                alarm.stop()

            # ------------------------------------------------
            # Show camera window
            # ------------------------------------------------

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            # ------------------------------------------------
            # Keyboard input
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] System interrupted by user."
        )

    finally:

        # ====================================================
        # SAFE SHUTDOWN
        # ====================================================

        print()
        print(
            "[INFO] Shutting down system..."
        )

        alarm.stop()

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

        print(
            "[INFO] Camera released."
        )

        print()
        print("=" * 65)
        print(" SESSION SUMMARY")
        print("=" * 65)
        print()

        print(
            f"Total drowsiness events : "
            f"{detector.drowsiness_events}"
        )

        print(
            f"Total yawns detected    : "
            f"{detector.yawn_count}"
        )

        print()

        print(
            "[INFO] System shut down safely."
        )

        print("=" * 65)


if __name__ == "__main__":

    main()