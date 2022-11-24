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

# Limitations
# 1) Processes data as it comes in, regardless of timeouts/power cycles, as
# data is processed when scanning reaches start/stop angle.
# 2) Num steps, start angle, stop angle and num samples can only be set on startup (however, range can be changed at run time)
# 3) Can only have one instance of the app running at any time

class PPlumeDetector():
    def __init__(self):

        # Constants
        self.k_const = 2 #Threshold for each ping = mean + K*std.dev
        self.window_width_m = 0.5
        self.cluster_min_fill_percent = 50
        self.threshold_min = 0.5*255
        self.threshold_max = 0.95 * 255
        self.grads_to_rads = np.pi/200
        self.threshold = 0.5*255

        self.window_width_pixels = None
        self.cluster_min_pixels = None

        self.comms = pymoos.comms()

        # Vars for storing data from MOOS DB
        self.num_samples = None
        self.num_steps = None
        self.start_angle_grads = 0
        self.stop_angle_grads = 399
        self.sonar_start_angle_grads = None
        self.sonar_stop_angle_grads = None
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
        self.seg_scan_snapshot = None # Copy of seg scan, taken at start/stop angle

        self.seg_img = None           # self.seg_scan warped into an image (re-gridded to cartesian grid)
        self.clustered_cores_img = None # Image with core cluster pixels (percentage of pixels in surrounding > threshold)
        self.cluster_regions_img = None # Image wth all pixels in clustering windows set to 1  (used for labelling)
        self.labelled_regions_img = None # Cluster regions image, with unique label (pixel value) applied to each region
        self.labelled_clustered_img = None # seg_image * labelled_regions_img

        # Array indicating whether the data in the corresponding columns of the scan_intensities and seg_scan matrices are valid
        self.scan_valid_cols = None

        self.scan_angles = None # Array containing scan angles (gradians) for each column of the seg_scan matrix
        self.num_scans = 0
        self.first_scan = True
        self.clustering_pending = False

        # Cartesian x-y co-ordinates of each point in the sector scan
        self.cart_x = None
        self.cart_y = None

        # Indicates whether all the config vars have been received, and the class data structures can be configured
        self.ready_for_config = False

        self.img_save_path = None

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
        except OSError as error:
            print(error)

        while True:

            # Process scan when ping data from start/end of scan is received
            if self.clustering_pending:

                # Warp data into image with cartesian co-ordinates, then cluster
                self.seg_img = self.create_sonar_image(self.seg_scan_snapshot)
                self.cluster()
                self.clustering_pending = False

                self.create_plots()

            time.sleep(0.02)  # 50Hz
            # TODO P1: Check for timeout if in active State

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

        return True

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
                self.sonar_start_angle_grads = val
            elif name == 'SONAR_STOP_ANGLE_GRADS':
                self.sonar_stop_angle_grads = val
            elif name == 'SONAR_TRANSMIT_ENABLE':
                self.transmit_enable = val
            elif name == 'SONAR_SPEED_OF_SOUND':
                self.speed_of_sound = val

            required_vars = [self.num_samples, self.num_steps, self.sonar_start_angle_grads, self.sonar_stop_angle_grads, self.speed_of_sound]

            # Class can be configured once all the vars have been set
            if all(item is not None for item in required_vars):
            #if self.num_samples and self.num_steps and self.start_angle_grads and self.stop_angle_grads and self.speed_of_sound:
                print("Config vars:samples: {0}, steps: {1}, start: {2}, stop: {3}, speed of sound: {4}".format(self.num_samples,
                        self.num_steps, self.sonar_start_angle_grads, self.sonar_stop_angle_grads, self.speed_of_sound))
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

        self.scan_processing_angles = [self.sonar_start_angle_grads, self.sonar_stop_angle_grads]

        print ("PPlumeDetector configured with {0} scanning field".format(self.seg_scan.shape))
        return

    def process_ping_data(self):

        if not self.decode_device_data_msg():
            return False

        if not self.update_scan_intensities():
            return False

        # Process the data when at the start/stop angles
        if self.device_data_msg.angle in self.scan_processing_angles:

            self.num_scans = self.num_scans + 1

            # Copy data and set flag for clustering to be completed in the run thread
            self.seg_scan_snapshot = copy.deepcopy(self.seg_scan)
            self.clustering_pending = True

            # Reset valid flags for new data
            self.scan_valid_cols = np.zeros(len(self.scan_angles), dtype=np.uint)

            # TODO P1 - Also reset sector_intensities, but leave beginning/end (also reset valid flag to match)

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
        '''Extracts the intensities from the Ping360 device data message stored in self.device_data_msg, and stores the
        segmented data into self.seg_scan'''
        ''' First cuts out the data close to the sonar head. Then adaptively thresholds the scan intensities to segment
                the data.'''

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

        # Remove noise data close to the head
        noise_range_samples = int((self.noise_range_m / self.calc_range(self.num_samples)) * self.num_samples)
        self.scan_intensities_denoised[:,scanned_index] = intensities
        self.scan_intensities_denoised[0:noise_range_samples, scanned_index] = np.zeros((noise_range_samples), dtype=np.uint8)

        # Segment data
        self.seg_scan[:,scanned_index] = (self.scan_intensities_denoised[:,scanned_index]  > self.threshold).astype(np.uint8)

        #print('Angle: ' + str(scanned_angle))

        return True

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
        radius = 200  # Output image will be 200x200 pixels
        warp_flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
        warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius,
                                    dsize=(2 * radius, 2 * radius),
                                    flags=warp_flags)

        return warped_image

    def cluster(self):
        '''Applies a square window clustering method to self.seg_img, and stores the image with the labelled clustered
        pixels as labelled_clustered_img'''

        image_width_pixels = self.seg_img.shape[1] # Assumes square image
        range_m = self.calc_range(self.num_samples)
        self.window_width_pixels = self.window_width_m * image_width_pixels / (2 * range_m)
        self.window_width_pixels = 2*math.floor(self.window_width_pixels/2) + 1 # Window size should be an odd number
        print("Window width is ", self.window_width_pixels)

        if self.window_width_pixels < 3:
            print('Clipping clustering window with to 3 pixels (minimum)')
            self.window_width_pixels = 3

        start = time.time()
        #print("Start: ", start)

        window_rows = self.window_width_pixels
        window_cols = self.window_width_pixels
        area = window_rows*window_cols
        self.cluster_min_pixels = self.cluster_min_fill_percent*area/100

        row_padding = math.floor(window_rows / 2)
        col_padding = math.floor(window_cols / 2)
        rows = self.seg_img.shape[0]
        cols = self.seg_img.shape[1]

        self.clustered_cores_img = np.zeros((rows, cols), dtype=np.uint8)
        self.cluster_regions_img = np.zeros((rows, cols), dtype=np.uint8)
        self.labelled_clustered_img = np.zeros((rows, cols), dtype=np.uint8)
        window_ones = np.ones((int(window_rows), int(window_cols)), dtype=np.uint8)

        # Note: output matrix is zero padded
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

        self.labelled_regions_img, num_regions = measure.label(self.cluster_regions_img, return_num=True, connectivity=2)
        self.labelled_clustered_img = self.labelled_regions_img * self.seg_img

        end = time.time()
        #print("End: ", end)
        print("Clustering time is ", end-start)

        return

    def calc_range(self, sample_num):
        '''Calculates the one-way range (meters) for the specified sample number'''

        if self.device_data_msg is None:
            raise Exception("Cannot calculate range without sample period from device data message")

        # Sample period in device data message is measured in 25 nano-second increments
        sample_period_sec = self.device_data_msg.sample_period * 25e-9

        range = (sample_period_sec * sample_num * self.speed_of_sound) / 2

        return range

    def create_plots(self):

        range_m = self.calc_range(self.num_samples)
        range_m_int = round(range_m)

        # Create warped (polar) images
        warped = self.create_sonar_image(self.scan_intensities)
        denoised_warped = self.create_sonar_image(self.scan_intensities_denoised)

        # Setup plot
        fig = plt.figure()
        suptitle = 'Scan ' + str(self.num_scans)
        plt.suptitle(suptitle)
        plt.axis('off')

        # Labels and label positions for warped images
        rows, cols = warped.shape[0], warped.shape[1]
        x_label_pos = [0, 0.25 * cols, 0.5 * cols, 0.75 * cols, cols]
        x_labels = [str(range_m_int), str(0.5 * range_m_int), '0', str(0.5 * range_m_int), str(range_m_int)]
        y_label_pos = [0, 0.25 * rows, 0.5 * rows, 0.75 * rows, rows]
        y_labels = [str(range_m_int), str(0.5 * range_m_int), '0', str(0.5 * range_m_int), str(range_m_int)]

        # 1: Original data, warped
        ax = fig.add_subplot(2, 3, 1)
        plt.imshow(warped, interpolation='none', cmap='jet')
        ax.title.set_text('1: Original')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 2: Denoised data
        ax = fig.add_subplot(2, 3, 2)
        plt.imshow(denoised_warped, interpolation='none', cmap='jet')
        ax.title.set_text('2: Denoised')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 3: Segmented data
        ax = fig.add_subplot(2, 3, 3)
        image = self.seg_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='RdYlBu')
        ax.title.set_text('3: Segmented')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 4: Clustered Cores
        ax = fig.add_subplot(2, 3, 4)
        image = self.clustered_cores_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='RdYlBu')
        ax.title.set_text('4: Clustered Cores')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)

        # 5: Labelled Regions
        ax = fig.add_subplot(2, 3, 5)
        image = self.labelled_regions_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
        ax.title.set_text('5: Labelled Regions')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)
        ax.set_aspect('equal')

        # 6: Final Output
        ax = fig.add_subplot(2, 3, 6)
        image = self.labelled_clustered_img.astype(float)
        image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
        plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
        ax.title.set_text('6: Labelled Clusters')
        ax.set_xticks(x_label_pos), ax.set_xticklabels(x_labels)
        ax.set_yticks(y_label_pos), ax.set_yticklabels(y_labels)
        ax.set_aspect('equal')

        fig.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(self.img_save_path, suptitle))
        plt.close()



if __name__ == "__main__":

    plume_detector = PPlumeDetector()
    plume_detector.run()