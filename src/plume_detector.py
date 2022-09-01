import copy
import sys
import time
import pymoos
import math
import numpy as np
from brping import PingMessage
from brping import PingParser
from brping import definitions
from datetime import datetime
from sklearn.cluster import DBSCAN

# Limitations
# 1) Processes data as it comes in, regardless of timeouts/power cycles, as
# data is processed when scanning reaches start/stop angle.
# 2) Num steps, start angle, stop angle and num samples can only be set on startup (however, range can be changed at run time)
# 3) Can only have one instance of the app running at any time

class PPlumeDetector():
    def __init__(self):

        # Constants
        self.k_const = 2 #Threshold for each ping = mean + K*std.dev
        self.threshold_min = 0.5*255
        self.threshold_max = 0.95 * 255
        self.grads_to_rads = np.pi/200
        self.threshold = 0

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

        # Angles at which to processs the sector scan data. Gets set to the start & stop angles
        self.scan_processing_angles = None

        # Matrix containing raw intensity data for a complete scan of pings between the start and stop angles
        self.scan_intensities = None
        self.scan_intensities_denoised = None # Scan intensities with the central noise data removed
        self.noise_range_m = 1 # Range within which data is ignored (usually just noise)

        # Matrix containing segmented data (i.e. result from thresholding) from the scan of the entire sonar swath.
        # Row indexes define the sample number and each column is for a different scan angle
        self.seg_scan = None

        self.clustered_seg = None

        # Array indicating whether the data in the corresponding columns of the scan_intensities and seg_scan matrices are valid
        self.scan_valid_cols = None

        self.scan_angles = None # Array containing scan angles (gradians) for each column of the seg_scan matrix
        self.full_scans = 0
        self.first_scan = True

        # Cartesian x-y co-ordinates of each point in the sector scan
        self.cart_x = None
        self.cart_y = None

        # Indicates whether all the config vars have been received, and the class data structures can be configured
        self.ready_for_config = False

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
        while True:
            time.sleep(0.1) #TODO P1: Reduce sleep time?
            #TODO P1: Check for timeout if in active State

        return

    def on_connect(self):
        '''Registers vars with the MOOS DB'''
        success = self.comms.register('SONAR_NUMBER_OF_SAMPLES', 0)
        success = success and self.comms.register('SONAR_NUM_STEPS', 0)
        success = success and self.comms.register('SONAR_START_ANGLE_GRADS', 0)
        success = success and self.comms.register('SONAR_STOP_ANGLE_GRADS', 0)
        success = success and self.comms.register('SONAR_PING_DATA', 0)
        success = success and self.comms.register('SONAR_TRANSMIT_ENABLE', 0)
        success = success and self.comms.register('SONAR_SPEED_OF_SOUND', 0)

        if success:
            self.set_state('DB_CONNECTED')
            self.status('GOOD')
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
        self.status_string = status_string_local
        print("Status: {0}".format(self.status_string))

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
            if self.ready_for_config:
                self.configure()
                if self.transmit_enable:
                    self.state_string = 'ACTIVE'
                else:
                    self.state_string = 'STANDBY'

        elif self.state_string == 'STANDBY':
            if self.transmit_enable:
                self.state_string = 'ACTIVE'

        elif self.state_string == 'ACTIVE':
            if self.binary_device_data_msg is not None:
                if self.process_ping_data():
                    self.set_status('GOOD')
                else:
                    self.set_status('PROCESSING_ERROR')

        return

    def save_input_var(self, msg):
        '''Saves message data in correct class var. Also sets 'ready_for_config' if all config vars received'''

        # Save message data
        name = msg.name()
        if name == 'SONAR_PING_DATA':
            self.binary_device_data_msg = msg.binary_data()
        else: # Numeric data type
            val = int(msg.double())
            print("Received {0}: {1}".format(name, val))

            if name == 'SONAR_NUMBER_OF_SAMPLES':
                self.num_samples = val
            elif name == 'SONAR_NUM_STEPS':
                self.num_steps = val
            elif name == 'SONAR_START_ANGLE_GRADS':
                self.start_angle_grads = val
            elif name == 'SONAR_STOP_ANGLE_GRADS':
                self.stop_angle_grads = val
            elif name == 'SONAR_TRANSMIT_ENABLE':
                self.transmit_enable = val
            elif name == 'SONAR_SPEED_OF_SOUND':
                self.speed_of_sound = val

        # Class can be configured once all the vars have been set
        if self.num_samples and self.num_steps and self.start_angle_grads and self.stop_angle_grads and self.speed_of_sound:
            print("Config vars:samples: {0}, steps: {1}, start: {2}, stop: {3}, speed of sound: {4}".format(self.num_samples,
                    self.num_steps, self.start_angle_grads, self.stop_angle_grads, self.speed_of_sound))
            self.ready_for_config = True

        return

    def configure(self):
        '''Initializes the self.scan_angles and self.seg_scan vars. The scan angles are computed base don the start/stop
        angles and the number of steps. '''

        # Calculate scan angles
        if self.start_angle_grads > self.stop_angle_grads: # Sector crosses 399->0 grads
            # Add 400 to stop angle because scan angles generation is easier when stop angle > start angle
            stop_angle_adjusted_grads = self.stop_angle_grads + 400
            scan_angles_list = [i for i in range(self.start_angle_grads, stop_angle_adjusted_grads+1, self.num_steps)]
            self.scan_angles = np.array(scan_angles_list, dtype=np.uint)
            for index, angle in enumerate(self.scan_angles):
                if angle >= 400: # Remove 400 grads offset that was added
                    self.scan_angles[index] = angle - 400
        else: # Sector does not cross 399 -> 0 grads
            scan_angles_list = [i for i in range(self.start_angle_grads, self.stop_angle_grads+1, self.num_steps)]
            self.scan_angles = np.array(scan_angles_list, dtype=np.uint)

        # Initialize self.seg_scan and self.scan_valid_cols
        self.scan_intensities = np.zeros((self.num_samples, len(self.scan_angles)), dtype=np.uint8)
        self.scan_intensities_denoised = np.zeros((self.num_samples, len(self.scan_angles)), dtype=np.uint8)
        self.seg_scan = np.zeros((self.num_samples, len(self.scan_angles)), dtype=np.uint8)
        self.scan_valid_cols = np.zeros(len(self.scan_angles), dtype=np.uint)

        self.scan_processing_angles = [self.start_angle_grads, self.stop_angle_grads]

        print ("PPlumeDetector configured with {0} scanning field".format(self.seg_scan.shape))
        return

    def init_cart_xy(self):
        '''Calculates cartesian x-y co-ordinates of each point in the sector scan, and stores them in self.cart_x, self.cart_y'''

        self.cart_x = np.zeros((self.num_samples, len(self.scan_angles)))
        self.cart_y = np.zeros((self.num_samples, len(self.scan_angles)))

        for sample_num in range(self.num_samples):
            for col, ping360_angle_grads in enumerate(self.scan_angles):

                range_m = self.calc_range(sample_num)

                # Convert angle from ping360 reference (0 towards bottom, clockwise rotation) to standard reference (0 towards
                # right, counter-clockwise rotation)
                angle_grads = 300 - ping360_angle_grads
                if angle_grads < 0:
                    angle_grads = angle_grads + 400

                # Convert angle in gradians to angle in radians
                angle_rads = angle_grads * 360 / 400 * math.pi / 180

                # Convert polar to cartesian co-ordinates
                self.cart_x[sample_num, col] = range_m * math.cos(angle_rads)
                self.cart_y[sample_num, col] = range_m * math.sin(angle_rads)


    def process_ping_data(self):

        if not self.decode_device_data_msg():
            return False

        # If not done, initialize arrays with cartesian x-y co-ordinates of each point in the sector scan
        # Done here because the sample period from the device data message is required for calculating ranges
        if self.cart_x is None:
            self.init_cart_xy()

        if not self.update_scan_intensities():
            return False

        if self.device_data_msg.angle in self.scan_processing_angles:
            self.segment_scan()

            # TODO P1 - Cluster Data

            # Reset valid flags for new data
            self.scan_valid_cols = np.zeros(len(self.scan_angles), dtype=np.uint)

            # TODO P1 - Also reset sector_intensities, but leave beginning/end (also reset valid flag to match)

        return False


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
        '''Extracts the intensities from the Ping360 device data message stored in self.device_data_msg, and stores the
        segmented data into self.seg_scan'''

        scanned_angle = self.device_data_msg.angle
        if scanned_angle not in self.scan_angles:
            print("Device data angle({0}) not within scan angles. Data not stored.".format(scanned_angle))
            return False

        # Ensure that dataset is the correct size
        intensities = np.frombuffer(self.device_data_msg.data, dtype=np.uint8) # Convert intensity data bytearray to numpy array
        if intensities.size != self.num_samples:
            print("Intensities array length ({0}) does not match number of samples ({1})".format(intensities.size))
            return False

        # Save segmented data and valid flag
        # First get column index for storage. The [0][0] required because np.where returns tuple containing an array
        scanned_index = np.where(self.scan_angles==scanned_angle)[0][0]
        self.scan_intensities[:,scanned_index] = intensities
        self.scan_valid_cols[scanned_index] = 1

        return True

    def segment_scan(self):
        ''' First cuts out the data close to the sonar head. Then adaptively thresholds the scan intensities to segment
        the data.'''

        # Remove noise data close to the head
        noise_range_samples = int((self.noise_range_m / self.calc_range(self.num_samples)) * self.num_samples)
        self.scan_intensities_denoised = copy.deepcopy(self.scan_intensities)
        self.scan_intensities_denoised[0:noise_range_samples, ] = np.zeros((noise_range_samples, len(self.scan_angles)),
                                                                           dtype=np.uint8)

        # Calculate mean an standard deviation of denoised data
        data = self.scan_intensities_denoised[noise_range_samples:,]
        mean = np.mean(self.scan_intensities_denoised[noise_range_samples:,])
        std_dev = np.std(self.scan_intensities_denoised[noise_range_samples:,])

        # Calculate threshold and clip if required
        self.threshold = mean + self.k_const * std_dev
        self.threshold = max(self.threshold, self.threshold_min)  # Clip threshold to the maximum
        self.threshold = min(self.threshold, self.threshold_max)  # Clip threshold to the minimum

        # Segment data
        self.seg_scan = (self.scan_intensities_denoised > self.threshold).astype(np.uint8)

        return

    def cluster_seg_scan_old(self, image):

        window_width_m = 0.5
        image_width_pixels = image.shape[1] # Assumes square image
        range_m = self.calc_range(self.num_samples)
        window_width_pixels = window_width_m * image_width_pixels / (2 * range_m)
        print("Window width is ", window_width_pixels)

        start = time.time()
        print("Start: ", start)

        # Window sizes should be odd numbers
        window_rows = window_width_pixels
        window_cols = window_width_pixels
        area = window_rows*window_cols
        row_padding = math.floor(window_rows / 2)
        col_padding = math.floor(window_cols / 2)

        #rows = self.seg_scan.shape[0]
        #cols = self.seg_scan.shape[1]
        #self.clustered_seg = np.zeros((rows, cols), dtype=np.uint8)
        rows = image.shape[0]
        cols = image.shape[1]
        self.clustered_seg = np.zeros((rows, cols), dtype=np.uint8)

        # Note: output matrix is zero padded
        for row in range(row_padding, rows-row_padding, 1):
            for col in range(col_padding, cols-col_padding, 1):
                if image[row, col]:
                    start_row = row - row_padding
                    end_row   = row + row_padding + 1
                    start_col = col - col_padding
                    end_col   = col + col_padding + 1
                    filled = (image[start_row:end_row, start_col:end_col]).sum()
                    if filled > 0.50*area:
                        self.clustered_seg[row,col] = 1

        end = time.time()
        print("End: ", end)
        print("Clustering time is ", end-start)

        return

    def cluster_seg_scan(self, image):
        # TODO P1 - Clean up

        window_width_m = 0.25
        image_width_pixels = image.shape[1] # Assumes square image
        range_m = self.calc_range(self.num_samples)
        window_width_pixels = window_width_m * image_width_pixels / (2 * range_m) # Image width is 2* range
        print("Window width is ", window_width_pixels)

        # From the image, extract the list of points (detections above the threshold)
        num_points = self.seg_scan.sum()
        points = np.zeros(shape=(num_points,2))
        index = 0
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row,col]:
                    #points[index] = [self.cart_x[row,col], self.cart_y[row,col]]
                    #points[index] = [row, col]
                    points[index] = [col, row]
                    index = index + 1

        start = time.time()
        print("Start: ", start)

        db = DBSCAN(eps=window_width_pixels, min_samples=6).fit(points)

        end = time.time()
        print("End: ", end)
        print("DBSCAN time is ", end-start)

        return points, db

    def calc_range(self, sample_num):
        '''Calculates the one-way range (meters) for the specified sample number'''

        if self.device_data_msg is None:
            raise Exception("Cannot calculate range without sample period from device data message")

        # Sample period in device data message is measured in 25 nano-second increments
        sample_period_sec = self.device_data_msg.sample_period * 25e-9

        range = (sample_period_sec * sample_num * self.speed_of_sound) / 2

        return range











