from decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from src.plume_detector import PPlumeDetector
import numpy as np
import cv2 as cv
import copy
import os
import re
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# 20210305-031345328.bin
# ping360_20210901_123108.bin

# Script settings
start_time = "00:10:00.000"
start_angle_grads = 0
stop_angle_grads = 399
num_samples = 1200
num_steps = 1


def create_sonar_images(sector_intensities):
    '''Rearrages sector intensities matrix to  '''

    # Rearrange sector_intensities matrix to match warp co-ordinates (0 is towards right)
    sector_intensities_t = copy.deepcopy(sector_intensities)
    sector_intensities_t = sector_intensities_t.transpose()

    sector_intensities_mod = copy.deepcopy(sector_intensities_t)
    sector_intensities_mod[0:100] = sector_intensities_t[300:400]
    sector_intensities_mod[100:400] = sector_intensities_t[0:300]

    # Warp intensities matrix into circular image
    radius = int(400 / (2 * np.pi))
    warp_flags = flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
    warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius, dsize=(2 * radius, 2 * radius),
                          flags=warp_flags)

    return warped_image

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("file",
                        help="File that contains PingViewer sensor log file.")
    args = parser.parse_args()

    # Create directory for saving images
    date_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    folder_name = f"""sonar_data_plots_{date_time}"""
    parent_dir = "./"
    img_save_path = os.path.join(parent_dir, folder_name)

    try:
        os.mkdir(img_save_path)
    except OSError as error:
        print(error)

    # Setup Plume Detector
    plume_detector = PPlumeDetector
    plume_detector.start_angle_grads = start_angle_grads
    plume_detector.stop_angle_grads = stop_angle_grads
    plume_detector.num_samples = num_samples
    plume_detector.num_steps = num_steps
    plume_detector.configure(plume_detector)

    # Open log and begin processing
    log = PingViewerLogReader(args.file)

    first_iteration = True
    sector_intensities = np.array([])
    scan_num = 0
    start_timestamp = ""
    start_time_obj = datetime.strptime(start_time, "%H:%M:%S.%f")

    for index, (timestamp, decoded_message) in enumerate(log.parser()):

        timestamp = re.sub(r"\x00", "", timestamp) # Strip any extra \0x00 (null bytes)
        time_obj = datetime.strptime(timestamp,"%H:%M:%S.%f")

        # Skip to start time
        if time_obj < start_time_obj:
            continue

        # Save timestamp of start of each scan
        if start_timestamp == "":
            start_timestamp = timestamp

        # Extract ping data from message
        angle = decoded_message.angle
        ping_intensities = np.frombuffer(decoded_message.data,
                                    dtype=np.uint8)  # Convert intensity data bytearray to numpy array

        if first_iteration == True:
            first_iteration = False
            sector_intensities = np.zeros((len(ping_intensities),400), dtype=np.uint8)

        sector_intensities[:, angle] = ping_intensities

        if angle == 199:
            scan_num += 1
            print('Last timestamp',timestamp)

            warped = create_sonar_images(sector_intensities)

            # Display images
            fig = plt.figure()
            suptitle = 'Scan ' + str(scan_num)
            plt.suptitle(suptitle)
            plt.title('Start time: ' + start_timestamp + ', End time: ' + timestamp)
            plt.axis('off')
            fig.add_subplot(2,1,1)
            plt.imshow(sector_intensities.transpose(),interpolation='bilinear',cmap='jet')

            fig.add_subplot(2, 1, 2)
            plt.imshow(warped, interpolation='bilinear',cmap='jet')
            plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))

            start_timestamp = ""


