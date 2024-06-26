import os
import csv
import copy
import time
import math
import pymoos
import pyproj
import numpy as np
import cv2 as cv
from datetime import datetime
from brping import PingParser
from brping import definitions
from collections import OrderedDict
from skimage import measure
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

# Limitations
# 1) Processes data as it comes in, regardless of timeouts/power cycles, as
# data is processed when scanning reaches start/stop angle.
# 2) Num steps, start angle, stop angle and num samples can only be set on startup (however, range can be changed at run time)
# 3) Can only have one instance of the app running at any time

class Cluster():
    def __init__(self):
        # Cluster radius is the distance from the cluster center to the furthest point from it
        self.radius_pixels = 0
        self.radius_m = 0
        self.num_points = 0

        # Sonar beam angle at cluster center
        self.angle_degs = 0

        # Vehicle nav data at time that the sonar pinged the cluster center
        self.nav_x = 0
        self.nav_y = 0
        self.nav_heading = 0

        # Cluster center in pixel coordinates
        self.center_row = 0
        self.center_col = 0

        # Cluster center in the different reference frames
        self.instrument_x = 0
        self.instrument_x = 0
        self.body_x = 0
        self.body_y = 0
        self.local_x = 0
        self.local_y = 0
        self.lat = 0
        self.long = 0

    def serialize(self):
        return OrderedDict([
            ("center_row_pixels", self.center_row),
            ("center_col_pixels", self.center_col),
            ("num_points", self.num_points),
            ("radius_pixels", self.radius_pixels),
            ("center_local_x_m", self.local_x),
            ("center_local_y_m", self.local_y),
            ("radius_m", self.radius_m),
            ("center_lat", self.lat),
            ("center_long", self.long),
            ("nav_x_m", self.nav_x),
            ("nav_y_m", self.nav_y),
            ("nav_heading_deg", self.nav_heading)
        ])

class PPlumeDetector():
    def __init__(self):

        # Algorithm Parameters
        self.threshold = 0.1 * 255 #0.1 * 255 #0.25 * 255
        #self.dbscan_epsilon_m = 0.501#0.707#0.2 #0.75
        self.window_width_m = 3.0 #1.0
        self.dbscan_min_fill_percent = 1 #30
        self.noise_range_m = 0.5 #2 # Range within which data is ignored (usually just noise)
        self.image_width_pixels = 400# Width in pixels of sonar images
        # Distance between the Ping360 and INS center, measured along the vehicle's longitudinal axis
        self.instrument_offset_x_m = 3
        self.num_cluster_outputs = 5

        # Constants
        self.grads_to_rads = np.pi/200 #400 gradians per 360 degs
        self.rads_to_grads = 200/np.pi
        self.max_angle_grads = 400

        self.dbscan_epsilon_pixels = None
        self.dbscan_min_pts = None
        self.dbscan_output = None

        self.comms = pymoos.comms()

        # Vars for storing data from MOOS DB
        self.num_samples = None
        self.num_steps = None
        self.start_angle_grads = None
        self.stop_angle_grads = None
        self.transmit_enable = None
        self.speed_of_sound = None
        self.binary_device_data_msg = None # Encoded PING360_DEVICE_DATA message
        self.device_data_msg = None # Decoded PING360_DEVICE_DATA message
        self.range_m = None # Sonar range
        self.lat_origin = None
        self.long_origin = None

        # Angles at which to processs the sector scan data. Gets set to the start & stop angles
        self.scan_processing_angles = None

        # Matrix containing raw intensity data for a complete scan of pings between the start and stop angles
        self.scan_intensities = None
        self.scan_intensities_denoised = None # Scan intensities with the central noise data removed

        # Matrix containing segmented data (i.e. result from thresholding) from the scan of the entire sonar swath.
        # Row indexes define the sample number and each column is for a different scan angle
        self.seg_scan = None
        self.seg_scan_snapshot = None # Copy of seg scan, taken at start/stop angle

        self.seg_img = None           # self.seg_scan warped into an image (re-gridded to cartesian grid)
        self.labelled_clustered_img = None # seg_image * labelled_regions_img
        self.clustered_img = None # Black and white version of labelled_clustered_img
        self.output_img = None
        self.clustered_img_view = None

        self.num_scans = 0
        self.first_scan = True
        self.clustering_pending = False

        # Cluster information for the clusters detected in the current scan
        self.num_clusters = 0
        self.clusters = [] # List of Cluster data structures
        self.sorted_clusters = [] # List of Cluster data structures, sorted based on cluster radius (largest to smallest)
        self.cluster_centers_string = ""

        # Cluster information of clusters detected since the start
        self.cluster_centers = []

        self.clustering_time_secs = None
        self.total_processing_time_secs = None

        self.data_save_path = None
        self.orig_images_path = None
        self.viewable_images_path = None

        # Most recent nav data
        self.current_nav_x = 0
        self.current_nav_y = 0
        self.current_nav_heading = 0

        # Nav data is saved every time a ping message is received. There is storage for each position of the sonar head,
        # allowing for the nav data at the time of each ping to be stored.
        self.nav_x_history = np.zeros(self.max_angle_grads)
        self.nav_y_history = np.zeros(self.max_angle_grads)
        self.nav_heading_history = np.zeros(self.max_angle_grads)

        #plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['font.family'] = 'serif'

        self.state_string= 'DB_DISCONNECTED'
        self.states = {
            'DB_REGISTRATION_ERROR': -1,
            'DB_DISCONNECTED': 0,
            'DB_CONNECTED': 1,
            'STANDBY': 2,
            'ACTIVE': 3,
        }

        self.status_string = 'GOOD'
        self.statuses = {
            'GOOD': 1,
            'DB_REGISTRATION_ERROR': -1,
            'TIMEOUT': -2,
            'PROCESSING_ERROR': -3
        }

    def run(self):

        # Setup pymoos comms
        print("Initial State: {0}".format(self.state_string))
        self.comms.set_on_connect_callback(self.on_connect)
        self.comms.set_on_mail_callback(self.on_mail)
        self.comms.run('localhost', 9000, 'p_plume_detector')
        #self.comms.run('192.168.56.104', 9000, 'p_plume_detector')

        # Create folder for saving images. Save in /log directory if it exists, otherwise use current directory
        date_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        folder_name = "plume_detector_data_" + date_time
        if os.path.exists("/log"):
            log_dir = "/log"
        else:
            log_dir  = os.path.dirname(__file__)
        self.data_save_path = os.path.join(log_dir, folder_name)
        self.orig_images_path = os.path.join(self.data_save_path, "orig_images")
        self.viewable_images_path = os.path.join(self.data_save_path, "viewable_images")

        try:
            os.mkdir(self.data_save_path)
            os.mkdir(self.orig_images_path)
            os.mkdir(self.viewable_images_path)

        except OSError as error:
            print(error)

        while True:

            # Process scan when ping data from start/end of scan is received
            if self.clustering_pending:

                start_time = time.time()
                # Warp data into image with cartesian co-ordinates, then cluster
                self.seg_img = self.create_sonar_image(self.seg_scan_snapshot)
                self.cluster()
                self.get_cluster_center_nav()
                self.georeference_clusters()
                self.output_sorted_cluster_centers()
                self.save_images()

                # Calculate and output the total processing time
                end_time = time.time()
                self.total_processing_time_secs = end_time - start_time
                print_str = "Scan {}: Processing time is {:.3f} secs".format(self.num_scans,
                                                                           self.total_processing_time_secs)
                print(print_str)
                self.comms.notify('PLUME_DETECTOR_TOTAL_PROCESSING_TIME_SECS', self.total_processing_time_secs,
                                  pymoos.time())

                self.save_text_data()
                self.clustering_pending = False

            time.sleep(0.02)  # 50Hz
            # TODO P1: Check for timeout if in active State

        return

    def on_connect(self):
        '''Registers vars with the MOOS DB'''
        success = self.comms.register('SONAR_NUMBER_OF_SAMPLES', 0)
        success = success and self.comms.register('SONAR_NUM_STEPS', 0)
        success = success and self.comms.register('SONAR_START_ANGLE_GRADS', 0)
        success = success and self.comms.register('SONAR_STOP_ANGLE_GRADS', 0)
        success = success and self.comms.register('SONAR_RANGE', 0)
        success = success and self.comms.register('SONAR_PING_DATA', 0)
        success = success and self.comms.register('SONAR_TRANSMIT_ENABLE', 0)
        success = success and self.comms.register('SONAR_SPEED_OF_SOUND', 0)
        success = success and self.comms.register('NAV_X', 0)
        success = success and self.comms.register('NAV_Y', 0)
        success = success and self.comms.register('NAV_HEADING', 0)
        success = success and self.comms.register('LAT_ORIGIN', 0)
        success = success and self.comms.register('LONG_ORIGIN', 0)
        success = success and self.comms.register('PLUME_DETECTOR_CSV_WRITE_CMD', 0)

        if success:
            self.set_state('DB_CONNECTED')
            self.set_status('GOOD')
        else:
            self.set_state('DB_REGISTRATION_ERROR')
            self.set_status('DB_REGISTRATION_ERROR')

        return success

    def set_state(self, state_string_local):
        ''' Sets the state_string class var, and updates the PLUME_DETECTOR_STATE_STRING and PLUME_DETECTOR_STATE_NUM
        MOOS DB vars if the DB is connected'''
        self.state_string = state_string_local
        print("State: {0}".format(self.state_string))

        if self.states[self.state_string] > self.states['DB_DISCONNECTED']:
            self.comms.notify('PLUME_DETECTOR_STATE_STRING', self.state_string, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_STATE_NUM', self.states[self.state_string], pymoos.time())

        return

    def set_status(self, status_string_local):
        ''' Sets the status_string class var, and updates the PLUME_DETECTOR_STATUS_STRING and PLUME_DETECTOR_STATUS_NUM
        MOOS DB vars if the DB is connected'''

        if self.status_string != status_string_local:
            print("Status: {0}".format(self.status_string))

        self.status_string = status_string_local

        if self.states[self.state_string] > self.states['DB_DISCONNECTED']:
            self.comms.notify('PLUME_DETECTOR_STATUS_STRING', self.status_string, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_STATUS_NUM', self.statuses[self.status_string], pymoos.time())

        return

    def on_mail(self):
        '''Handles incoming messages - calls functions to configure the class and process the ping data. The num steps,
        start angle, stop angle and number of samples can only be set on startup. Once the class is configured, changes
        to these vars in the MOOS DB do not have any effect.'''

        # Save all new input data
        msg_list = self.comms.fetch()

        for msg in msg_list:

            self.save_input_var(msg)

            # Evaluate state machine, call function to process any new ping data
            if self.state_string in ['DB_DISCONNECTED', 'DB_REGISTRATION_ERROR']:
                # Do nothing if all vars not registered with the DB
                # Also, should not get here if state is 'DB_DISCONNECTED'
                pass

            elif self.state_string == 'DB_CONNECTED':
                if self.configure():
                    if self.transmit_enable:
                        self.set_state('ACTIVE')
                    else:
                        self.set_state('STANDBY')

            elif self.state_string == 'STANDBY':
                if self.transmit_enable:
                    self.set_state('ACTIVE')

            elif self.state_string == 'ACTIVE':
                if self.binary_device_data_msg is not None:
                    if self.process_ping_data():
                        self.set_status('GOOD')
                    else:
                        self.set_status('PROCESSING_ERROR')

        return True

    def save_input_var(self, msg):
        '''Saves message data in correct class var.'''

        # Save message data
        name = msg.name()
        if name == 'SONAR_PING_DATA':
            self.binary_device_data_msg = msg.binary_data()
        else: # Numeric data type
            val = msg.double()
            #print("Received {0}: {1}".format(name, val))

            if name == 'SONAR_NUMBER_OF_SAMPLES':
                self.num_samples = int(val)
            elif name == 'SONAR_NUM_STEPS':
                self.num_steps = int(val)
            elif name == 'SONAR_START_ANGLE_GRADS':
                self.start_angle_grads = int(val)
            elif name == 'SONAR_STOP_ANGLE_GRADS':
                self.stop_angle_grads = int(val)
            elif name == 'SONAR_RANGE':
                self.range_m = val
            elif name == 'SONAR_TRANSMIT_ENABLE':
                self.transmit_enable = int(val)
            elif name == 'SONAR_SPEED_OF_SOUND':
                self.speed_of_sound = int(val)
            elif name == 'NAV_X':
                self.current_nav_x = val
            elif name == 'NAV_Y':
                self.current_nav_y = val
            elif name == 'NAV_HEADING':
                self.current_nav_heading = val
            elif name == 'LAT_ORIGIN':
                self.lat_origin = val
            elif name == 'LONG_ORIGIN':
                self.long_origin = val
            elif name == 'PLUME_DETECTOR_CSV_WRITE_CMD':
                if val == 1:
                    self.write_cluster_centers_csv()

        return

    def configure(self):
        ''' Initialize class data storage arrays if all config variable have been set'''

        required_vars = [self.num_samples, self.range_m, self.num_steps, self.start_angle_grads, self.stop_angle_grads,
                         self.speed_of_sound, self.lat_origin, self.long_origin]

        # Class can be configured if all the config vars have been set
        if all(item is not None for item in required_vars):

            self.scan_intensities = np.zeros((self.num_samples, self.max_angle_grads), dtype=np.uint8)
            self.scan_intensities_denoised = np.zeros((self.num_samples, self.max_angle_grads), dtype=np.uint8)
            self.seg_scan = np.zeros((self.num_samples, self.max_angle_grads), dtype=np.uint8)

            self.scan_processing_angles = [self.start_angle_grads, self.stop_angle_grads]

            print("Config vars:samples: {0}, range: {1}, steps: {2}, start: {3}, stop: {4}, speed of sound: {5}, "
                  "lat origin: {6}, long origin: {7}".format(self.num_samples, self.range_m, self.num_steps,
                  self.start_angle_grads, self.stop_angle_grads, self.speed_of_sound, self.lat_origin, self.long_origin))

            return True

        else:
            return False


    def process_ping_data(self):
        '''Calls functions to decode the binary ping data and save nav data. If the transducer head is at the start/stop
        angle, it creates a copy of the sector scan data and sets a flag to indicate that the clustering can be run'''

        if not self.decode_device_data_msg():
            return False

        if not self.update_scan_intensities():
            return False

        self.save_nav_data()

        #print(str(self.device_data_msg.angle))
        # Process the data when at the start/stop angles
        if self.device_data_msg.angle in self.scan_processing_angles:
            self.num_scans = self.num_scans + 1
            self.comms.notify('PLUME_DETECTOR_NUM_SCANS', self.num_scans, pymoos.time())

            # Copy data and set flag for clustering to be completed in the run thread
            self.seg_scan_snapshot = copy.deepcopy(self.seg_scan)
            self.clustering_pending = True

        return True


    def decode_device_data_msg(self):
        '''Decodes the Ping360 device data message stored in self.binary_device_data_msg, and stores the decoded
          message in self.device_data_msg '''

        ping_parser = PingParser()

        # Decode binary device data message
        for byte in self.binary_device_data_msg:
            # If the byte fed completes a valid message, PingParser.NEW_MESSAGE is returned
            if ping_parser.parse_byte(byte) is PingParser.NEW_MESSAGE:
                self.device_data_msg = ping_parser.rx_msg

        # Set to None as an indicator that the data has been processed
        self.binary_device_data_msg = None

        if self.device_data_msg is None:
            print("Failed to parse message")
            return False

        if self.device_data_msg.message_id != definitions.PING360_DEVICE_DATA:
            print("Received {0} message instead of {1} message".format(self.device_data_msg.name,'device_data'))
            return False

        # TODO P2: Add debug control for print
        #print(self.device_data_msg)
        return True

    def update_scan_intensities(self):
        ''' Stores the intensity data, removes the noise close to the transducer and segments the data'''

        # TODO P2 - Store ping timestamps, and discard old data before clustering

        # Ensure that dataset is the correct size
        intensities = np.frombuffer(self.device_data_msg.data, dtype=np.uint8) # Convert intensity data bytearray to numpy array
        if intensities.size != self.num_samples:
            print("Intensities array length ({0}) does not match number of samples ({1}). Data not stored".format(intensities.size, self.num_samples))
            return False

        # Save the intensity data
        scanned_angle = self.device_data_msg.angle
        self.scan_intensities[:, scanned_angle] = intensities

        # Remove noise data close to the head
        noise_range_samples = int(self.noise_range_m / self.range_m * self.num_samples)
        self.scan_intensities_denoised[:,scanned_angle] = intensities
        self.scan_intensities_denoised[0:noise_range_samples, scanned_angle] = np.zeros((noise_range_samples), dtype=np.uint8)

        # Apply a threshold to segment the data
        self.seg_scan[:,scanned_angle] = (self.scan_intensities_denoised[:,scanned_angle]  > self.threshold).astype(np.uint8)

        #print('Angle: ' + str(scanned_angle))

        return True

    def save_nav_data(self):
        '''Stores the current nav data in the nav data history arrays, at the index location defined by the scanned angle'''

        scanned_angle = self.device_data_msg.angle

        self.nav_x_history[scanned_angle] = self.current_nav_x
        self.nav_y_history[scanned_angle] = self.current_nav_y
        self.nav_heading_history[scanned_angle] = self.current_nav_heading

        return

    def create_sonar_image(self, sector_intensities):
        '''First rearrages sector intensities matrix to match OpenCV reference - includes reference frame conversion as
        Ping 360 reference uses 0 towards aft while OpenCV uses 0 towards right. Then re-grids to cartesian co-ordinates
        using the OpenCV warpPolar function'''

        # Transpose sector intensities to match matrix format required for warping
        sector_intensities_t = copy.deepcopy(sector_intensities)
        sector_intensities_t = sector_intensities_t.transpose()

        # Rearrange sector_intensities matrix to match warp co-ordinates (0 is towards right)
        sector_intensities_mod = copy.deepcopy(sector_intensities_t)
        sector_intensities_mod[0:100] = sector_intensities_t[300:400]
        sector_intensities_mod[100:400] = sector_intensities_t[0:300]

        # Warp intensities matrix into circular image
        radius = int(self.image_width_pixels/2)
        warp_flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
        warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius,
                                    dsize=(2 * radius, 2 * radius),
                                    flags=warp_flags)

        return warped_image

    def cluster(self):
        '''Runs DBSCAN clustering on self.seg_img, and stores the clustering output in self.dbscan_output. Also
        calculates the cluster centers and radii, and saves them in the clusters data structure. An image
        representation of the DBSCAN output is saved in self.labelled_clustered_img.'''

        # Reset class vars
        self.num_clusters = 0
        self.clusters = []
        self.clustering_time_secs = 0
        self.labelled_clustered_img = np.zeros_like(self.seg_img, dtype=np.uint8)

        # Return if there are no segmented points
        n_points = self.seg_img.sum()
        if n_points == 0:
            return

        # List the coordinates of the segmented points (detections above the threshold)
        points = np.zeros(shape=(n_points, 2))
        index = 0
        for row in range(self.seg_img.shape[0]):
            for col in range(self.seg_img.shape[1]):
                if self.seg_img[row, col]:
                    points[index] = [col, row]
                    index = index + 1

        # Compute DBSCAN epsilon parameter value, in pixels
        # image_width_pixels = self.seg_img.shape[1]  # Assumes square image
        # self.dbscan_epsilon_pixels = self.dbscan_epsilon_m * image_width_pixels / (2 * self.range_m)
        # self.dbscan_epsilon_pixels = 2.51
        # if self.dbscan_epsilon_pixels < 1.5:
        #     print('Clipping DBSCAN epsilon to 1.5 pixels (minimum)')
        #     self.dbscan_epsilon_pixels = 1.5

        # Convert window width from meters to pixels
        image_width_pixels = self.seg_img.shape[1] # Assumes square image
        self.window_width_pixels = self.window_width_m * image_width_pixels / (2 * self.range_m)
        self.window_width_pixels = 2*math.floor(self.window_width_pixels/2) + 1 # Window size should be an odd number

        # Ensure widow width is at least 3 pixels
        if self.window_width_pixels < 3:
            print('Clipping clustering block with to 3 pixels (minimum)')
            self.window_width_pixels = 3

        self.dbscan_epsilon_pixels = self.window_width_pixels/2
        #self.dbscan_epsilon_pixels = self.window_width_pixels / math.sqrt(2)
        self.comms.notify('PLUME_DETECTOR_DBSCAN_EPSILON_PIXELS', self.dbscan_epsilon_pixels, pymoos.time())

        # Compute DBSCAN minimum points parameter value
        #area = math.pi * (self.dbscan_epsilon_pixels**2)
        area = (2*self.dbscan_epsilon_pixels)**2
        self.dbscan_min_pts = round(self.dbscan_min_fill_percent*area/100)


        #self.dbscan_min_pts = 8

        # Run DBSCAN clustering
        start = time.time()
        try:
            #self.dbscan_output = DBSCAN(eps=self.dbscan_epsilon_pixels, min_samples=self.dbscan_min_pts).fit(points)
            self.dbscan_output = DBSCAN(eps=self.dbscan_epsilon_pixels, min_samples=self.dbscan_min_pts, metric='chebyshev').fit(points)
        except Exception as e:
            print("DBSCAN error: ", e)
            return
        self.clustering_time_secs = time.time() - start
        print("DBSCAN clustering time: %.3f secs" % self.clustering_time_secs )

        # Calculate number of clusters, ignoring noise (labelled as '-1') if present.
        labels = self.dbscan_output.labels_
        self.num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate number of cluster points and noise points
        n_noise_points = list(labels).count(-1)
        n_cluster_points = n_points - n_noise_points

        print("DBSCAN epsilon, min samples: %.1f, %.1f" % (self.dbscan_epsilon_pixels, self.dbscan_min_pts))
        print("DBSCAN num points, num clusters: %d, %d" % (n_points, self.num_clusters))
        print("DBSCAN num cluster points, num noise points: %d, %d" % (n_cluster_points, n_noise_points))

        # Create the labelled_clustered_img, and calculate the cluster centers and radii
        self.clusters = [Cluster() for i in range(self.num_clusters)]
        for cluster_idx in np.arange(self.num_clusters):

            # Calculate cluster center
            cluster_member_mask = labels == cluster_idx
            cluster_points = points[cluster_member_mask]
            center_col = np.mean(cluster_points[:, 0])
            center_row = np.mean(cluster_points[:, 1])
            center = np.array([center_col, center_row])

            # Update the labelled clustered image with the current cluster's pixel labels & compute the cluster radius
            radius = 0
            for point in cluster_points:

                # Label corresponding pixel in the labelled clustered image. Add 1 since cluster indexing starts at 0.
                self.labelled_clustered_img[int(point[1]), int(point[0])] = cluster_idx + 1

                # Find point furthest from center to determine the cluster radius
                dist = np.linalg.norm(point - center)
                if dist > radius:
                    radius = dist

            # Save values in clusters data structure
            self.clusters[cluster_idx].center_row = center_row
            self.clusters[cluster_idx].center_col = center_col
            self.clusters[cluster_idx].radius_pixels = radius
            self.clusters[cluster_idx].num_points = len(cluster_points)

        return

    def get_cluster_center_nav(self):
        '''Gets the vehicle nav data at time that the sonar pinged the cluster center'''

        num_rows = num_cols = self.labelled_clustered_img.shape[0]  # Assumes square image

        for i in np.arange(self.num_clusters):
            # Cluster center coordinates relative to the top left corner of the image
            row = self.clusters[i].center_row
            col = self.clusters[i].center_col

            # Cluster center coordinates relative to the center of the image
            x = row - (num_rows - 1)/2
            y = -1 * (col - (num_cols - 1)/2)

            # Calculate sonar transmit angle at the time that the cluster center was scanned
            theta_rads = math.atan2(y,x)
            theta_grads = round(theta_rads * self.rads_to_grads)

            # Retrieve nav data at the time that the cluster center was scanned, using the sonar transmit angle
            self.clusters[i].nav_x = self.nav_x_history[theta_grads]
            self.clusters[i].nav_y = self.nav_y_history[theta_grads]
            self.clusters[i].nav_heading = self.nav_heading_history[theta_grads]

        return

    def georeference_clusters(self):
        '''For each cluster center, transforms from image coordinates to instrument, body, local and earth coordinates.
         Also converts each cluster radius from pixels to meters. Coordinates are stored in self.clusters. '''

        num_rows = num_cols = self.labelled_clustered_img.shape[0] # Assumes square image
        meters_per_pixel = (2 * self.range_m) / self.image_width_pixels

        for i in np.arange(self.num_clusters):

            # Retrieve cluster center and radius in image coordinates
            center_row = self.clusters[i].center_row
            center_col = self.clusters[i].center_col
            radius_pixels = self.clusters[i].radius_pixels

            # Calculate cluster radius in meteres
            self.clusters[i].radius_m = radius_pixels * meters_per_pixel

            # Calculate cluster center in instrument coordinates
            instrument_x = (num_rows - 1)/2 - center_row
            instrument_x = instrument_x * meters_per_pixel
            instrument_y = (num_cols - 1)/2 - center_col
            instrument_y = instrument_y * meters_per_pixel

            # Calculate cluster center in body coordinates
            body_x = instrument_x + self.instrument_offset_x_m
            body_y = instrument_y

            # Calculate cluster center in local coordinates
            theta = (self.clusters[i].nav_heading - 90) * np.pi/180
            local_x = self.clusters[i].nav_x + (body_x * math.cos(theta) + body_y * math.sin(theta))
            local_y = self.clusters[i].nav_y + (body_x * -math.sin(theta) + body_y * math.cos(theta))

            # Calculate cluster center in earth (lat,long) coordinates
            dist = math.hypot(local_x, local_y)
            fwd_azimuth = math.degrees(math.atan2(local_x, local_y)) # Use x/y because azimuth is wrt y axis
            long, lat, back_azimuth = (pyproj.Geod(ellps='WGS84').fwd(self.lat_origin, self.long_origin,
                                                                      fwd_azimuth, dist))

            # Save calculations
            self.clusters[i].instrument_x = instrument_x
            self.clusters[i].instrument_y = instrument_y
            self.clusters[i].body_x = body_x
            self.clusters[i].body_y = body_y
            self.clusters[i].local_x = local_x
            self.clusters[i].local_y = local_y
            self.clusters[i].lat = lat
            self.clusters[i].long = long

            self.cluster_centers.append((local_x,local_y,self.clusters[i].nav_x,self.clusters[i].nav_y))

            #print("nav: " + "{:.2f}".format(self.clusters[i].nav_x ) + "," + "{:.2f}".format(self.clusters[i].nav_y))
            #print("local: " + "{:.2f}".format(local_x) + "," + "{:.2f}".format(local_y))
            #print("earth: " + "{:.6f}".format(lat) + "," + "{:.6f}".format(long))

        return

    def output_sorted_cluster_centers(self):
        '''Sorts the clusters in order of radius (largest to smallest) and sets the cluster outputs for the app'''

        self.cluster_centers_string = ""

        # Sort list of clusters based on radius (largest to smallest)
        self.sorted_clusters = sorted(self.clusters, key=lambda cluster_i: cluster_i.radius_m, reverse=True)

        # Assemble string with cluster centers list
        # Sample format: <cluster 1 x>,<cluster 1 y>,<cluster 1 radius>:<cluster 2 x>,<cluster 2 y>,<cluster 2 radius>
        centers = ""
        for i in np.arange(self.num_clusters):
            cluster = self.sorted_clusters[i]
            # Add cluster <local_x,local_y,radius_m> to string
            centers = centers + "{:.2f}".format(cluster.local_x) + ","
            centers = centers + "{:.2f}".format(cluster.local_y) + ","
            centers = centers + "{:.2f}".format(cluster.radius_m)

            # Add colon delimiter between info for each cluster
            if i < (self.num_clusters-1):
                centers = centers + ":"
        self.cluster_centers_string = centers

        # Output number of clusters and cluster centers list
        self.comms.notify('PLUME_DETECTOR_NUM_CLUSTERS', self.num_clusters, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTERS_LIST', self.cluster_centers_string, pymoos.time())

        # Output the center coordinates  and radius of each cluster. If there are more outputs than detected clusters,
        # those outputs are set to 0.
        for i in range(self.num_cluster_outputs):

            # Construct variable name using cluster number
            cluster_num = i+1 #Numbering of cluster outputs starts at 1
            cluster_x_name = 'PLUME_DETECTOR_CLUSTER_'+ str(cluster_num)+ '_X_M'
            cluster_y_name = 'PLUME_DETECTOR_CLUSTER_' + str(cluster_num) + '_Y_M'
            cluster_radius_name = 'PLUME_DETECTOR_CLUSTER_' + str(cluster_num) + '_RADIUS_M'

            if i < self.num_clusters:
                # Output the center coordinates  and radius of each cluster.
                cluster = self.sorted_clusters[i]
                self.comms.notify(cluster_x_name, cluster.local_x, pymoos.time())
                self.comms.notify(cluster_y_name, cluster.local_y, pymoos.time())
                self.comms.notify(cluster_radius_name, cluster.radius_m, pymoos.time())

            else:
                # Outputs which number more than the detected clusters are set to 0
                self.comms.notify(cluster_x_name, 0, pymoos.time())
                self.comms.notify(cluster_y_name, 0, pymoos.time())
                self.comms.notify(cluster_radius_name, 0, pymoos.time())

    def save_images(self):
        '''Saves a set of original clustering images, as well as modified high-contrast viewable images'''

        ## Save original images
        dir = self.orig_images_path

        filename = "Scan_" + str(self.num_scans) + "_Im1_Segmented_Unwarped.png"
        cv.imwrite(os.path.join(dir, filename), self.seg_scan_snapshot)

        filename = "Scan_" + str(self.num_scans) + "_Im2_Segmented_Warped.png"
        cv.imwrite(os.path.join(dir, filename), self.seg_img)

        filename = "Scan_" + str(self.num_scans) + "_Img3_Clustered.png"
        cv.imwrite(os.path.join(dir, filename), self.labelled_clustered_img)

        ## Increase image contrast and save
        dir = self.viewable_images_path

        warped = self.create_sonar_image(self.scan_intensities)
        filename = "Scan_" + str(self.num_scans) + "_Img1_Input.png"
        cv.imwrite(os.path.join(dir, filename), warped)

        filename = "Scan_" + str(self.num_scans) + "_Img2_Segmented.png"
        cv.imwrite(os.path.join(dir, filename), 255*self.seg_img)

        # Convert labelled clustered image to a black and white image
        filename = "Scan_" + str(self.num_scans) + "_Img5_Clustered.png"
        self.clustered_img = np.zeros_like(self.labelled_clustered_img)
        self.clustered_img[self.labelled_clustered_img > 0] = 255 # Pixel values > 0 are set to 255
        cv.imwrite(os.path.join(dir, filename), self.clustered_img)

    def save_text_data(self):
        ''' Create test file with summary info for the scan, as well as detailed info for each cluster '''

        # Save text file with the viewable images
        filename = "Scan_" + str(self.num_scans) + "_Data.txt"
        file_path = os.path.join(self.viewable_images_path, filename)

        with open(file_path, "w") as file:

            # Write summary data
            date_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            file.write("Timestamp: " + date_time + "\n")
            file.write("Num Clusters: " + str(self.num_clusters) + "\n")
            file.write("Total processing time: {:.3f} secs\n".format(self.total_processing_time_secs))
            file.write("Clustering time: {:.3f} secs\n\n".format(self.clustering_time_secs))

            # Write data for each cluster
            for i in np.arange(self.num_clusters):
                data = self.sorted_clusters[i].serialize()
                file.write(str(data) + "\n")

    def create_output_image(self):
        ''' Creates the output image, by copying the labelled clustered image, and adding the cluster centers and
        circles to show to the cluster radii. This function is only meant to be used for post-processing.'''

        self.output_img = copy.deepcopy(self.labelled_clustered_img)

        # Increase image resolution before adding circles so that they are not pixelated
        scale = 4
        dim = (int(self.output_img.shape[1]*scale), int(self.output_img.shape[0]*scale))
        self.output_img = cv.resize(self.output_img, dim, interpolation = cv.INTER_NEAREST)
        self.output_img = self.output_img.astype(np.uint8)
        # Increment non-zero pixel values. Allows for '1' to be used for the cluster circles and centers
        self.output_img = np.where(self.output_img > 0, self.output_img+1, self.output_img)

        # Add cluster centers and circle encompassing the clusters to the image
        for i in np.arange(self.num_clusters):

            center_row = self.clusters[i].center_row
            center_col = self.clusters[i].center_col
            radius = self.clusters[i].radius_pixels

            # Add cluster center to image
            circle_center = (round(center_col * scale), round(center_row * scale))
            self.output_img = cv.circle(self.output_img, circle_center, 3, 1, -1)

            # Add circle encompassing the cluster
            circle_radius = round(radius*scale + scale/2)
            self.output_img = cv.circle(self.output_img, circle_center, circle_radius,1, 2, cv.LINE_8)

    def write_cluster_centers_csv(self):
        '''Writes the cluster center positions and AUV position at the time of detection to a csv file. All positions
        are in local coordinates'''

        csv_file_name = "../../src/cluster_centers.csv"

        with open(csv_file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write a header row
            csv_writer.writerow(['Cluster Center X', 'Cluster Center Y', 'AUV Nav X', 'AUV Nav Y'])
            csv_writer.writerows(self.cluster_centers)



if __name__ == "__main__":

    plume_detector = PPlumeDetector()
    plume_detector.run()
