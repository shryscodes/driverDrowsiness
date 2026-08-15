# Driver Drowsiness Detection System
# File: alarm.py
# Purpose: Manage the drowsiness warning alarm.

import threading
import winsound


class Alarm:

    def __init__(
        self,
        frequency=1000,
        duration=500
    ):

        self.frequency = frequency
        self.duration = duration

        self.active = False

        self._stop_event = threading.Event()
        self._thread = None

    # ========================================================
    # START ALARM
    # ========================================================

    def start(self):

        # If the alarm is already running,
        # do not start another alarm.
        if self.active:
            return

        self.active = True

        self._stop_event.clear()

        print("[ALARM] DROWSINESS DETECTED!")

        self._thread = threading.Thread(
            target=self._alarm_loop,
            daemon=True
        )

        self._thread.start()

    # ========================================================
    # ALARM LOOP
    # ========================================================

    def _alarm_loop(self):

        while not self._stop_event.is_set():

            winsound.Beep(
                self.frequency,
                self.duration
            )

            # Short pause between warning beeps.
            if self._stop_event.wait(0.2):
                break

    # ========================================================
    # STOP ALARM
    # ========================================================

    def stop(self):

        if not self.active:
            return

        self._stop_event.set()

        self.active = False

        print("[ALARM] Alarm stopped.")

        # Wait for the alarm thread to finish.
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):

            self._thread.join(
                timeout=1.0
            )

        self._thread = None


# ============================================================
# ALARM TEST
# ============================================================

def main():

    import time

    alarm = Alarm()

    print("=" * 50)
    print(" ALARM TEST")
    print("=" * 50)
    print()

    print("[TEST] Starting alarm...")
    alarm.start()

    time.sleep(5)

    print("[TEST] Stopping alarm...")
    alarm.stop()

    print()
    print("[SUCCESS] Alarm test completed.")


if __name__ == "__main__":

    main()