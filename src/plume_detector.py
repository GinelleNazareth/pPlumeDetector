import copy
import time
import pymoos
import math
import numpy as np
from datetime import datetime
from brping import PingParser
from brping import definitions
from skimage import measure
import cv2 as cv
from matplotlib import pyplot as plt
import os
import pyproj

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

class PPlumeDetector():
    def __init__(self):

        # Algorithm Parameters
        self.threshold = 0.5 * 255 #0.3 * 255
        self.window_width_m = 0.5 #1.0 # Clustering window size
        self.cluster_min_fill_percent = 50 #30
        self.image_width_pixels = 400# Width in pixels of sonar images
        # Distance between the Ping360 and INS center, measured along the vehicle's longitudinal axis
        self.instrument_offset_x_m = 3

        # Constants
        self.grads_to_rads = np.pi/200 #400 gradians per 360 degs
        self.rads_to_grads = 200/np.pi
        self.max_angle_grads = 400

        self.window_width_pixels = None
        self.cluster_min_pixels = None

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
        self.noise_range_m = 1 # Range within which data is ignored (usually just noise)

        # Matrix containing segmented data (i.e. result from thresholding) from the scan of the entire sonar swath.
        # Row indexes define the sample number and each column is for a different scan angle
        self.seg_scan = None
        self.seg_scan_snapshot = None # Copy of seg scan, taken at start/stop angle

        self.seg_img = None           # self.seg_scan warped into an image (re-gridded to cartesian grid)
        self.clustered_cores_img = None # Image with core cluster pixels (percentage of pixels in surrounding > threshold)
        self.cluster_regions_img = None # Image wth all pixels in clustering windows set to 1  (used for labelling)
        self.labelled_regions_img = None # Cluster regions image, with unique label (pixel value) applied to each region
        self.labelled_clustered_img = None # seg_image * labelled_regions_img
        self.output_img = None
        self.clustered_img_view = None

        self.num_scans = 0
        self.first_scan = True
        self.clustering_pending = False

        self.num_clusters = 0
        self.clusters = [] # List of Cluster data structures
        self.output_cluster_num = 0  # Index of the cluster with the largest radiuslargest_cluster_num
        self.cluster_centers_string = ""

        self.img_save_path = None

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
        print("Initial State: {0}".format(self.state_string))
        self.comms.set_on_connect_callback(self.on_connect)
        self.comms.set_on_mail_callback(self.on_mail)
        self.comms.run('localhost', 9000, 'p_plume_detector')

        # Create directory for saving images
        date_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        folder_name = "sonar_data_plots_" + date_time
        parent_dir = "./"
        self.img_save_path = os.path.join(parent_dir, folder_name)

        try:
            os.mkdir(self.img_save_path)
            os.mkdir(os.path.join(self.img_save_path,'raw'))
            os.mkdir(os.path.join(self.img_save_path,'viewable'))
        except OSError as error:
            print(error)

        while True:

            # Process scan when ping data from start/end of scan is received
            if self.clustering_pending:
                # TODO P2: Print total processing time
                # Warp data into image with cartesian co-ordinates, then cluster
                self.seg_img = self.create_sonar_image(self.seg_scan_snapshot)
                self.cluster()
                self.calc_and_show_cluster_centers()
                self.get_cluster_center_nav()
                self.georeference_clusters()
                self.output_cluster_centers()
                self.clustering_pending = False

                self.save_plots()
                #self.create_plots()

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
            print("Intensities array length ({0}) does not match number of samples ({1}). Data not stored".format(intensities.size))
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
        '''Applies a square window clustering method to self.seg_img, and stores the image with the labelled clustered
        pixels as labelled_clustered_img'''

        # Reset class vars
        self.num_clusters = 0
        self.clusters = []

        start = time.time()

        # Convert window width from meters to pixels
        image_width_pixels = self.seg_img.shape[1] # Assumes square image
        self.window_width_pixels = self.window_width_m * image_width_pixels / (2 * self.range_m)
        self.window_width_pixels = 2*math.floor(self.window_width_pixels/2) + 1 # Window size should be an odd number

        # Ensure widow width is at least 3 pixels
        if self.window_width_pixels < 3:
            print('Clipping clustering window with to 3 pixels (minimum)')
            self.window_width_pixels = 3
        self.comms.notify('PLUME_DETECTOR_WINDOW_WIDTH', self.window_width_pixels, pymoos.time())

        # Convert minimum fill from percentage to pixels
        window_rows = self.window_width_pixels
        window_cols = self.window_width_pixels
        area = window_rows*window_cols
        self.cluster_min_pixels = self.cluster_min_fill_percent*area/100

        # Add border padding to image
        row_padding = math.floor(window_rows / 2)
        col_padding = math.floor(window_cols / 2)
        self.seg_img = cv.copyMakeBorder(self.seg_img, row_padding, row_padding, col_padding, col_padding,
                                         cv.BORDER_CONSTANT, None, value=0)

        # Initialize image matrices
        rows = self.seg_img.shape[0]
        cols = self.seg_img.shape[1]
        self.clustered_cores_img = np.zeros((rows, cols), dtype=np.uint8)
        self.cluster_regions_img = np.zeros((rows, cols), dtype=np.uint8)
        self.labelled_clustered_img = np.zeros((rows, cols), dtype=np.uint8)
        window_ones = np.ones((int(window_rows), int(window_cols)), dtype=np.uint8)

        # Create clustered_cores_img, identifying pixels at the center of high density windows.
        # Also create cluster_regions_img, identifying high density regions. Note: output matrices are zero padded
        for row in range(row_padding, rows-row_padding, 1):
            for col in range(col_padding, cols-col_padding, 1):
                if self.seg_img[row, col]:
                    start_row = row - row_padding
                    end_row   = row + row_padding + 1
                    start_col = col - col_padding
                    end_col   = col + col_padding + 1
                    filled = (self.seg_img[start_row:end_row, start_col:end_col]).sum()
                    if filled > self.cluster_min_pixels:
                        self.clustered_cores_img[row,col] = 1
                        self.cluster_regions_img[start_row:end_row, start_col:end_col] = window_ones

        # Identify and label separate regions
        self.labelled_regions_img, self.num_clusters = measure.label(self.cluster_regions_img, return_num=True, connectivity=2)

        # Mask input image with the labelled regions image to create the labelled clustered image
        self.labelled_clustered_img = self.labelled_regions_img * self.seg_img

        # Print clustering time
        end = time.time()
        clustering_time = end - start
        clustering_print_str = 'Scan ' + str(self.num_scans) + ": Clustering time is " + str(clustering_time) + " secs"
        print(clustering_print_str)


        return

    def calc_and_show_cluster_centers(self):
        '''Calculates the cluster centers and radii, and draws them on the output image'''

        self.clusters = [Cluster() for i in range(self.num_clusters+1)]
        self.output_img = copy.deepcopy(self.labelled_clustered_img)

        # Calculate cluster centers and radii
        # Note that cluster numbering starts at 1 (to match cluster pixel values)
        for cluster_num in np.arange(1,self.num_clusters+1,1):

            # Get indices of cluster pixels
            indices = np.nonzero(self.labelled_clustered_img==cluster_num)

            # Calculate center coordinates
            center_row = indices[0].mean()
            center_col = indices[1].mean()

            # Find furthest point & calculate cluster radius
            radius = 0
            center = np.array([center_row, center_col])
            for i in range(len(indices[0])):
                point = np.array([indices[0][i], indices[1][i]])
                dist = np.linalg.norm(point - center)
                #dist = math.dist(center, point)
                if dist > radius:
                    radius = dist

            # Save values in clusters data structure
            self.clusters[cluster_num].center_row = center_row
            self.clusters[cluster_num].center_col = center_col
            self.clusters[cluster_num].radius_pixels = radius


        # # Increase image resolution before adding circles so that they are not pixelated
        # scale = 4
        # dim = (int(self.output_img.shape[1]*scale), int(self.output_img.shape[0]*scale))
        # self.output_img = cv.resize(self.output_img, dim, interpolation = cv.INTER_NEAREST)
        # self.output_img = self.output_img.astype(np.uint8)
        # # Increment non-zero pixel values. Allows for '1' to be used for the cluster circles and centers
        # self.output_img = np.where(self.output_img > 0, self.output_img+1, self.output_img)
        #
        # # Add cluster centers and circle encompassing the clusters to the image
        # for cluster_num in np.arange(1, self.num_clusters + 1, 1):
        #
        #     center_row = self.clusters[cluster_num].center_row
        #     center_col = self.clusters[cluster_num].center_col
        #     radius = self.clusters[cluster_num].radius_pixels
        #
        #     # Add cluster center to image
        #     circle_center = (round(center_col * scale), round(center_row * scale))
        #     self.output_img = cv.circle(self.output_img, circle_center, 3, 1, -1)
        #
        #     # Add circle encompassing the cluster
        #     circle_radius = round(radius*scale + scale/2)
        #     self.output_img = cv.circle(self.output_img, circle_center, circle_radius,1, 2, cv.LINE_8)

        return

    def get_cluster_center_nav(self):
        '''Gets the vehicle nav data at time that the sonar pinged the cluster center'''

        num_rows = num_cols = self.labelled_clustered_img.shape[0]  # Assumes square image

        # Note that cluster numbering starts at 1 (to match cluster pixel values)
        for i in np.arange(1,self.num_clusters+1,1):
            # Cluster center coordinates relative to the top left corner of the image
            row = self.clusters[i].center_row
            col = self.clusters[i].center_col

            # Cluster center coordinates relative to the center of the image
            x = row - (num_rows - 1)/2
            y = -1 * (col - (num_cols - 1)/2)

            theta_rads = math.atan2(y,x)
            theta_degs = theta_rads * 180/math.pi
            theta_grads = round(theta_rads * self.rads_to_grads)

            self.clusters[i].nav_x = self.nav_x_history[theta_grads]
            self.clusters[i].nav_y = self.nav_y_history[theta_grads]
            self.clusters[i].nav_heading = self.nav_heading_history[theta_grads]

        return

    def georeference_clusters(self):
        '''For each cluster center, transforms from image coordinates to instrument, body, local and earth coordinates.
         Also converts each cluster radius from pixels to meters. Coordinates are stored in self.clusters. '''

        num_rows = num_cols = self.labelled_clustered_img.shape[0] # Assumes square image
        meters_per_pixel = (2 * self.range_m) / self.image_width_pixels

        # Note that cluster numbering starts at 1 (to match cluster pixel values)
        for i in np.arange(1,self.num_clusters+1,1):

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

            print("nav: " + "{:.2f}".format(self.clusters[i].nav_x ) + "," + "{:.2f}".format(self.clusters[i].nav_y))
            print("local: " + "{:.2f}".format(local_x) + "," + "{:.2f}".format(local_y))
            print("earth: " + "{:.6f}".format(lat) + "," + "{:.6f}".format(long))

        return

    def output_cluster_centers(self):
        '''Identifies the largest cluster and sets cluster outputs for the app'''

        self.output_cluster_num = 0
        self.cluster_centers_string = ""

        # Set all outputs to 0 if no clusters are detected
        if self.num_clusters == 0:
            self.comms.notify('PLUME_DETECTOR_CLUSTER_DETECTED', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_X_M', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_Y_M', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_LAT', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_LONG', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_RADIUS_M', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_NUM_CLUSTERS', 0, pymoos.time())
            self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTERS_LIST', "", pymoos.time())
            return

        # Identify the largest cluster
        max_radius_m = 0
        # Note that cluster numbering starts at 1 (to match cluster pixel values)
        for i in np.arange(1, self.num_clusters + 1, 1):
            if self.clusters[i].radius_m > max_radius_m:
                self.output_cluster_num = i
                max_radius_m = self.clusters[i].radius_m

        # Assemble string with cluster centers list
        # Sample format: <cluster 1 x>,<cluster 1 y>,<cluster 1 radius>:<cluster 2 x>,<cluster 2 y>,<cluster 2 radius>
        centers = ""
        for i in np.arange(1, self.num_clusters + 1, 1):

            # Add cluster <local_x,local_y,radius_m> to string
            centers = centers + "{:.2f}".format(self.clusters[i].local_x) + ","
            centers = centers + "{:.2f}".format(self.clusters[i].local_y) + ","
            centers = centers + "{:.2f}".format(self.clusters[i].radius_m)

            # Add colon delimiter between info for each cluster
            if i < self.num_clusters:
                centers = centers + ":"
        self.cluster_centers_string = centers

        # Set outputs
        cluster_x = self.clusters[self.output_cluster_num].local_x
        cluster_y = self.clusters[self.output_cluster_num].local_y
        cluster_lat = self.clusters[self.output_cluster_num].lat
        cluster_long = self.clusters[self.output_cluster_num].long
        cluster_radius_m = self.clusters[self.output_cluster_num].radius_m
        self.comms.notify('PLUME_DETECTOR_CLUSTER_DETECTED', 1, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_X_M', cluster_x, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_Y_M', cluster_y, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_LAT', cluster_lat, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTER_LONG', cluster_long, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_RADIUS_M',cluster_radius_m, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_NUM_CLUSTERS', self.num_clusters, pymoos.time())
        self.comms.notify('PLUME_DETECTOR_CLUSTER_CENTERS_LIST', self.cluster_centers_string, pymoos.time())


    def save_plots(self):

        dir = os.path.join(self.img_save_path, 'raw')

        filename = "Scan_" + str(self.num_scans) + "_Im1_Segmented_Unwarped.png"
        cv.imwrite(os.path.join(dir, filename), self.seg_scan_snapshot)

        filename = "Scan_" + str(self.num_scans) + "_Im2_Segmented_Warped.png"
        cv.imwrite(os.path.join(dir, filename), self.seg_img)

        filename = "Scan_" + str(self.num_scans) + "_Img3_ClusteredCores.png"
        cv.imwrite(os.path.join(dir, filename), self.clustered_cores_img)

        filename = "Scan_" + str(self.num_scans) + "_Img4_LabelledRegions.png"
        cv.imwrite(os.path.join(dir, filename), self.labelled_regions_img)

        filename = "Scan_" + str(self.num_scans) + "_Img5_Clustered.png"
        cv.imwrite(os.path.join(dir, filename), self.labelled_clustered_img)

        dir = os.path.join(self.img_save_path, 'viewable')

        warped = self.create_sonar_image(self.scan_intensities)
        filename = "Scan_" + str(self.num_scans) + "_Img1_Input.png"
        cv.imwrite(os.path.join(dir, filename), warped)

        filename = "Scan_" + str(self.num_scans) + "_Img2_Segmented.png"
        cv.imwrite(os.path.join(dir, filename), 255*self.seg_img)

        filename = "Scan_" + str(self.num_scans) + "_Img3_ClusteredCores.png"
        cv.imwrite(os.path.join(dir, filename), 255*self.clustered_cores_img)

        filename = "Scan_" + str(self.num_scans) + "_Img4_ClusterRegions.png"
        cv.imwrite(os.path.join(dir, filename), 255*self.cluster_regions_img)

        # TODO: Use separate image
        filename = "Scan_" + str(self.num_scans) + "_Img5_Clustered.png"
        self.labelled_clustered_img[self.labelled_clustered_img > 0] = 255
        cv.imwrite(os.path.join(dir, filename), self.labelled_clustered_img)


    def create_plots(self):

        range_m_int = round(self.range_m)

        # Create warped (polar) images
        warped = self.create_sonar_image(self.scan_intensities)
        #denoised_warped = self.create_sonar_image(self.scan_intensities_denoised)

        ### Plot Clustering Steps ###
        fig = plt.figure("Clustering Steps")
        #plt.subplots(2,2,layout="constrained")
        suptitle = 'Scan ' + str(self.num_scans) + ', ' + str(self.num_clusters) + ' Cluster(s)'
        plt.suptitle(suptitle)
        #plt.title(self.cluster_centers_string)
        plt.axis('off')

        # Labels and label positions for warped images
        rows, cols = warped.shape[0], warped.shape[1]
        x_label_pos = [0, 0.25 * cols, 0.5 * cols, 0.75 * cols, cols]
        x_labels = [str(range_m_int), str(0.5 * range_m_int), '0', str(0.5 * range_m_int), str(range_m_int)]
        y_label_pos = [0, 0.25 * rows, 0.5 * rows, 0.75 * rows, rows]
        y_labels = [str(range_m_int), str(0.5 * range_m_int), '0', str(0.5 * range_m_int), str(range_m_int)]

        # 1: Original data, warped
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(warped, interpolation='none', cmap='jet',vmin=0,vmax=255)
        ax.title.set_text('1: Original')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 2: Denoised data
        #ax = fig.add_subplot(2, 3, 2)
        #plt.imshow(denoised_warped, interpolation='none', cmap='jet',vmin=0,vmax=255)
        #ax.title.set_text('2: Denoised')
        #ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        #ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 2: Segmented data
        ax = fig.add_subplot(2, 2, 2)
        image = self.seg_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='RdYlBu')
        ax.title.set_text('3: Segmented')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 4: Clustered Cores
        #ax = fig.add_subplot(2, 3, 4)
        #image = self.clustered_cores_img.astype(float)
        #image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        #plt.imshow(image, interpolation='none', cmap='RdYlBu')
        #ax.title.set_text('4: Clustered Cores')
        #ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        #ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 3: Labelled Regions
        ax = fig.add_subplot(2, 2, 3)
        image = self.labelled_regions_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
        ax.title.set_text('5: Labelled Regions')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)
        ax.set_aspect('equal')

        # Label positions for output images - different because images are larger
        rows, cols = plume_detector.output_img.shape[0], plume_detector.output_img.shape[1]
        output_x_label_pos = [0, 0.25 * cols, 0.5 * cols, 0.75 * cols, cols]
        output_y_label_pos = [0, 0.25 * rows, 0.5 * rows, 0.75 * rows, rows]

        # 6: Final Output
        ax = fig.add_subplot(2, 2, 4)
        image = self.output_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
        ax.title.set_text('6: Labelled Clusters')
        ax.set_xticks(output_x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(output_y_label_pos), ax.set_yticklabels(y_labels)
        ax.set_aspect('equal')

        #fig.tight_layout()
        plt.subplot_tool()
        plt.show()
        image_name = "Clustering_Steps_Scan_" + str(self.num_scans)
        plt.savefig(os.path.join(self.img_save_path, image_name), dpi=400)
        #plt.clf()

        # ### Plot Clustering Overview ###
        # fig = plt.figure()
        # plt.suptitle('Scan ' + str(self.num_scans))
        # plt.title(str(self.num_clusters) + ' Cluster(s)')
        # plt.axis('off')
        #
        # # 1: Original data, warped
        # ax = fig.add_subplot(1, 2, 1)
        # plt.imshow(warped, interpolation='none', cmap='jet',vmin=0,vmax=255)
        # ax.title.set_text('Original')
        # ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        # ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)
        #
        # # 2: Plot clusters if num clusters > 0
        # if self.num_clusters > 0:
        #     ax = fig.add_subplot(1, 2, 2)
        #     image = self.output_img.astype(float)
        #     image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        #     plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
        #     ax.title.set_text('Labelled Clusters')
        #     ax.set_xticks(output_x_label_pos), ax.set_xticklabels(x_labels)
        #     ax.set_yticks(output_y_label_pos), ax.set_yticklabels(y_labels)
        #     ax.set_aspect('equal')
        #
        # fig.tight_layout()
        #
        # # plt.show()
        # image_name = "Clustering_Overview_Scan_" + str(self.num_scans)
        # plt.savefig(os.path.join(self.img_save_path, image_name), dpi=400)
        # plt.close(fig)




if __name__ == "__main__":

    plume_detector = PPlumeDetector()
    plume_detector.run()
