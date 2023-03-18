import numpy as np
from unittest import TestCase

from src.plume_detector import PPlumeDetector
from src.plume_detector import Cluster

# Tests for georeferencing code which cannot be validated with the visualize script

class TestPPlumeDetector(TestCase):

    def setUp(self):
        self.detector = PPlumeDetector()
        self.detector.range_m = 50
        self.detector.num_clusters = 4
        self.detector.image_width_pixels = 400
        self.detector.labelled_clustered_img = np.zeros((404, 404), dtype=np.uint8)

        self.detector.clusters = [Cluster() for i in range(self.detector.num_clusters + 1)]


class TestGeoreferencing(TestPPlumeDetector):

    # Test 1 - Cluster detections in each of the four quadrants. No nav or instrument offset or heading rotations
    def test1(self):

        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 0

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 0

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 0

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 0

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 25.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, -25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 25.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, -25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, -25.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, -25.0, places=2)

    # Test 2 - Cluster detections in each of the four quadrants. Added instrument offset. No nav offset or heading rotations
    def test2(self):

        self.detector.instrument_offset_x_m = 5

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 0

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 0

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 0

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 0

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 30.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, -24.999, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 30, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, -25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, -20.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 25.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, -20.0, places=2)

    # Test 3 - Cluster detections in each of the four quadrants. 45 deg  heading rotation. No instrument offset or nav offset
    def test3(self):
        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 45

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 45

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 45

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 45

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, -35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, -35.355, places=2)


    # Test 4 - Cluster detections in each of the four quadrants. 135 deg  heading rotation. No instrument offset or nav offset
    def test4(self):
        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 135

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 135

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 135

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 135

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, -35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, 35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, 35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, -35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, 0.0, places=2)


    # Test 5 - Cluster detections in each of the four quadrants. 225 deg  heading rotation. No instrument offset or nav offset
    def test5(self):
        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 225

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 225

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 225

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 225

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, -35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, -35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, 35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, 35.355, places=2)


    # Test 6 - Cluster detections in each of the four quadrants. 315 deg  heading rotation. No instrument offset or nav offset
    def test6(self):
        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_x = 0
        self.detector.clusters[1].nav_heading = 315

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_x = 0
        self.detector.clusters[2].nav_heading = 315

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_x = 0
        self.detector.clusters[3].nav_heading = 315

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_x = 0
        self.detector.clusters[4].nav_heading = 315

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, -35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 0.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, -35.355, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 35.355, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, 0.0, places=2)


    # Test 7 - Cluster detections in each of the four quadrants. Added nav offset. No instrument offset of heading rotation
    def test7(self):
        self.detector.instrument_offset_x_m = 0

        self.detector.clusters[1].center_row = 101.5
        self.detector.clusters[1].center_col = 301.5
        self.detector.clusters[1].radius_pixels = 10
        self.detector.clusters[1].nav_x = 25
        self.detector.clusters[1].nav_y = 50
        self.detector.clusters[1].nav_heading = 0

        self.detector.clusters[2].center_row = 101.5
        self.detector.clusters[2].center_col = 101.5
        self.detector.clusters[2].radius_pixels = 10
        self.detector.clusters[2].nav_x = 25
        self.detector.clusters[2].nav_y = 50
        self.detector.clusters[2].nav_heading = 0

        self.detector.clusters[3].center_row = 301.5
        self.detector.clusters[3].center_col = 101.5
        self.detector.clusters[3].radius_pixels = 10
        self.detector.clusters[3].nav_x = 25
        self.detector.clusters[3].nav_y = 50
        self.detector.clusters[3].nav_heading = 0

        self.detector.clusters[4].center_row = 301.5
        self.detector.clusters[4].center_col = 301.5
        self.detector.clusters[4].radius_pixels = 10
        self.detector.clusters[4].nav_x = 25
        self.detector.clusters[4].nav_y = 50
        self.detector.clusters[4].nav_heading = 0

        self.detector.georeference_clusters()

        self.assertAlmostEqual(self.detector.clusters[1].local_x, 50.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[1].local_y, 75.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[2].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[2].local_y, 75.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[3].local_x, 0.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[3].local_y, 25.0, places=2)

        self.assertAlmostEqual(self.detector.clusters[4].local_x, 50.0, places=2)
        self.assertAlmostEqual(self.detector.clusters[4].local_y, 25.0, places=2)

