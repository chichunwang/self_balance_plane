"""Ball-loss recovery checks without a camera, serial port, or servo."""
import unittest

import numpy as np

from ppreal.balance import BallRecovery


class BallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([1500., 1490., 1510.])
        self.last_pulses = np.array([1550., 1450., 1580.])
        self.recovery = BallRecovery(self.centers, 0)

    def test_short_missing_detection_keeps_tracking(self):
        waiting, override, transition = self.recovery.update(200000000, 0, 0, False, self.last_pulses)
        self.assertFalse(waiting)
        self.assertIsNone(override)
        self.assertIsNone(transition)

    def test_loss_returns_smoothly_then_waits_without_repeated_events(self):
        start = 310000000
        waiting, pulses, event = self.recovery.update(start, 0, 0, False, self.last_pulses)
        self.assertTrue(waiting)
        self.assertEqual(event, "lost")
        np.testing.assert_allclose(pulses, self.last_pulses)
        waiting, pulses, event = self.recovery.update(start + 500000000, 0, 0, False, pulses)
        np.testing.assert_allclose(pulses, (self.last_pulses + self.centers) / 2)
        self.assertIsNone(event)
        waiting, pulses, event = self.recovery.update(start + 60000000000, 0, 0, False, pulses)
        self.assertTrue(waiting)
        np.testing.assert_allclose(pulses, self.centers)
        self.assertIsNone(event)

    def test_resume_requires_frame_captured_after_center_return(self):
        start = 310000000
        self.recovery.update(start, 0, 0, False, self.last_pulses)
        waiting, pulses, event = self.recovery.update(start + 500000000, start + 500000000,
                                                     start + 440000000, True, self.last_pulses)
        self.assertTrue(waiting)
        self.assertIsNone(event)
        waiting, pulses, event = self.recovery.update(start + 1010000000, start + 1010000000,
                                                     start + 950000000, True, pulses)
        self.assertTrue(waiting)
        np.testing.assert_allclose(pulses, self.centers)
        tick = start + 1070000000
        waiting, pulses, event = self.recovery.update(tick, tick, start + 1010000000, True, pulses)
        self.assertFalse(waiting)
        self.assertIsNone(pulses)
        self.assertEqual(event, "resumed")
        self.assertEqual(self.recovery.last_seen_ns, tick)

    def test_new_valid_frame_resets_loss_timer(self):
        self.recovery.update(290000000, 290000000, 230000000, True, self.last_pulses)
        waiting, _, event = self.recovery.update(500000000, 290000000, 230000000, False, self.last_pulses)
        self.assertFalse(waiting)
        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
