from decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from src.plume_detector import PPlumeDetector
import numpy as np
import cv2 as cv
import math
import copy
import os
import re
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# 20210305-031345328.bin
# ping360_20210901_123108.bin

# Script settings
#start_time = "00:00:30.000"
start_time = "00:10:02.000"
start_angle_grads = 0
stop_angle_grads = 399
num_samples = 1200
num_steps = 1
speed_of_sound = 1500


def create_sonar_images(sector_intensities):
    '''Rearrages sector intensities matrix to  '''

    # Transpose sector intensities to match required orientation for warping
    sector_intensities_t = copy.deepcopy(sector_intensities)
    sector_intensities_t = sector_intensities_t.transpose()

    # Rearrange sector_intensities matrix to match warp co-ordinates (0 is towards right)
    sector_intensities_mod = copy.deepcopy(sector_intensities_t)
    sector_intensities_mod[0:100] = sector_intensities_t[300:400]
    sector_intensities_mod[100:400] = sector_intensities_t[0:300]

    # Warp intensities matrix into circular image
    radius = int(600 / (2 * np.pi))
    warp_flags = flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_NEAREST
    warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius, dsize=(2 * radius, 2 * radius),
                          flags=warp_flags)

    return warped_image


def calc_scatter(plume_detector, sector_intensities):
    '''Calculate x-y co-ordinates of points for scatter plot. 'Colours' are intensity values'''
    sector_intensities[:, angle] = ping_intensities

    it = np.nditer(sector_intensities, flags = ['multi_index'])
    x = np.zeros(sector_intensities.size)
    y = np.zeros(sector_intensities.size)
    colours = np.zeros(sector_intensities.size)

    i = 0
    while not it.finished:

        # Get angle and range
        sample_num, ping360_angle_grads = it.multi_index
        range_m = plume_detector.calc_range(plume_detector, sample_num)

        # Convert angle from ping360 reference (0 towards bottom, clockwise rotation) to standard reference (0 towards
        # right, counter-clockwise rotation)
        angle_grads = 300 - ping360_angle_grads
        if angle_grads < 0:
            angle_grads = angle_grads + 400

        # Convert angle in gradians to angle in radians
        angle_rads = angle_grads * 360/400 * math.pi/180

        # Convert polar to cartesian co-ordinates
        x[i] = range_m * math.cos(angle_rads)
        y[i] = range_m * math.sin(angle_rads)
        colours[i] = sector_intensities[sample_num,ping360_angle_grads]

        i = i+1
        is_not_finished = it.iternext()

    return x, y, colours

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
    plume_detector.speed_of_sound = speed_of_sound
    plume_detector.configure(plume_detector)

    # Open log and begin processing
    log = PingViewerLogReader(args.file)

    sector_intensities = np.zeros((num_samples, 400), dtype=np.uint8)
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

        plume_detector.device_data_msg = decoded_message
        plume_detector.update_seg_scan(plume_detector)


        sector_intensities[:, angle] = ping_intensities

        # Display data at the end of each sector scan
        if angle == 199:
            scan_num += 1
            print('Last timestamp',timestamp)
            range_m = plume_detector.calc_range(plume_detector, num_samples)
            print('Range: ', range)

            seg_sector = 255*plume_detector.seg_scan

            plume_detector.cluster_seg_scan(plume_detector)
            clustered_seg = 255*plume_detector.clustered_seg

            warped = create_sonar_images(sector_intensities)
            seg_warped = create_sonar_images(seg_sector)
            clustered_seg_warped = create_sonar_images(clustered_seg)


            # Display images
            fig = plt.figure()
            suptitle = 'Scan ' + str(scan_num)
            plt.suptitle(suptitle)
            plt.title('Start time: ' + start_timestamp + ', End time: ' + timestamp)
            plt.axis('off')

            ax = fig.add_subplot(2, 2, 1)
            x, y, colours = calc_scatter(plume_detector, sector_intensities)
            #plt.scatter(x,y,c=colours)
            #ax.set_aspect('equal')

            #plt.imshow(sector_intensities.transpose(), interpolation='nearest', cmap='jet')
            #ax.set_xticks([0, 0.25*num_samples, 0.5*num_samples, 0.75*num_samples, num_samples],
            #              labels=['0', str(0.25*range_m), str(0.5*range_m), str(0.75*range_m), str(range_m)])


            #fig.add_subplot(2, 3, 2)
            #plt.imshow(seg_sector.transpose(),interpolation='nearest',cmap='jet')

            #fig.add_subplot(2, 3, 3)
            #plt.imshow(clustered_seg.transpose(),interpolation='nearest',cmap='jet')


            # Labels and label positions for warped images
            rows, cols = warped.shape[0], warped.shape[1]
            range_m_int = round(range_m)
            x_label_pos = [0, 0.25*cols, 0.5*cols, 0.75*cols, cols]
            x_labels    = [str(range_m_int), str(0.5*range_m_int), '0', str(0.5*range_m_int), str(range_m_int)]
            y_label_pos = [0, 0.25*rows, 0.5*rows, 0.75*rows, rows]
            y_labels    = [str(range_m_int), str(0.5*range_m_int), '0', str(0.5*range_m_int), str(range_m_int)]

            # Original data, warped
            ax = fig.add_subplot(2, 2, 3)
            plt.imshow(warped, interpolation='bilinear',cmap='jet')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # Segmented data, warped
            ax = fig.add_subplot(2, 2, 2)
            plt.imshow(seg_warped, interpolation='nearest',cmap='jet')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # Segmented & Clustered data, warped
            ax = fig.add_subplot(2, 2, 4)
            plt.imshow(clustered_seg_warped, interpolation='nearest',cmap='jet')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))

            start_timestamp = ""


