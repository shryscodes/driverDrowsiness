# Driver Drowsiness Detection System
# File: camera.py
# Purpose: Test and manage webcam frame capture.

import cv2


class Camera:

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.capture = None

    def open(self):
        """Open the camera."""

        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            print("[ERROR] Could not open the camera.")
            return False

        # Request the desired resolution.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("[INFO] Camera opened successfully.")

        return True

    def read(self):
        """Read one frame from the camera."""

        if self.capture is None:
            return False, None

        success, frame = self.capture.read()

        if not success:
            print("[WARNING] Could not read frame from camera.")

        return success, frame

    def release(self):
        """Release the camera."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        cv2.destroyAllWindows()

        print("[INFO] Camera released.")


def main():

    camera = Camera()

    if not camera.open():
        return

    print("[INFO] Press Q to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Display the live camera feed.
            cv2.imshow("Driver Drowsiness - Camera Test", frame)

            # Check whether the user pressed Q.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\n[INFO] Program interrupted by user.")

    finally:

        camera.release()


if __name__ == "__main__":
    main()