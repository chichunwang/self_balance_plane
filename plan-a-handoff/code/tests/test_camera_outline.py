"""Regression checks for the calibrated outline and edge-ball detection margin."""
import unittest

import cv2
import numpy as np

from ppreal.balldetect import detect_ball
from ppreal.camtool import camera_geometry, draw_view


class CameraOutlineTests(unittest.TestCase):
    def test_outline_uses_saved_radius_and_edge_ball_stays_detectable(self):
        saved = {"map": {"u0": 200., "v0": 200., "mm_per_px": .5, "front_deg": 90.},
                 "platform_r_px": 120.}
        mapping, detector, outline = camera_geometry(saved)
        self.assertEqual(outline, (200., 200., 120.))
        self.assertEqual(detector.roi, (200., 200., 160.))
        self.assertEqual(mapping.to_mm(200., 200.), (0., 0.))
        gray = np.zeros((400, 400), dtype=np.uint8)
        # Its center is on the plate edge, with half of the ball beyond it.
        cv2.circle(gray, (320, 200), 40, 255, -1)
        self.assertTrue(detect_ball(gray, detector).found)
        empty = np.zeros_like(gray)
        preview = draw_view(empty, grid=False, roi=outline)
        self.assertTrue(np.array_equal(preview[200, 320], [255, 160, 0]))
        self.assertTrue(np.array_equal(preview[200, 360], [0, 0, 0]))

    def test_uncalibrated_camera_does_not_invent_plate_outline(self):
        mapping, detector, outline = camera_geometry({})
        self.assertIsNone(mapping)
        self.assertIsNone(outline)
        self.assertIsNone(detector.roi)


if __name__ == "__main__":
    unittest.main()
