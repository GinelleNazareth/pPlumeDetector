from decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from src.plume_detector import PPlumeDetector
import numpy as np
import cv2 as cv
import time
import math
import copy
import os
import re
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# 20210305-031345328.bin
# ping360_20210901_123108.bin

# Script settings
start_time = "00:00:30.000"
#start_time = "00:10:00.000"
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
    radius = 100 # Output image will be 200x200 pixels
    warp_flags = flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_NEAREST
    warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius, dsize=(2 * radius, 2 * radius),
                          flags=warp_flags)

    return warped_image


def calc_scatter(plume_detector, sector_intensities):
    '''Calculate x-y co-ordinates of points for scatter plot. 'Colours' are intensity values'''

    it = np.nditer(sector_intensities, flags = ['multi_index'])
    x = np.zeros(sector_intensities.size)
    y = np.zeros(sector_intensities.size)
    colours = np.zeros(sector_intensities.size)

    i = 0
    while not it.finished:

        # Get angle and range
        sample_num, angle = it.multi_index

        x[i] = plume_detector.cart_x[sample_num, angle]
        y[i] = plume_detector.cart_y[sample_num, angle]
        colours[i] = sector_intensities[sample_num, angle]

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
    plume_detector = PPlumeDetector()
    plume_detector.start_angle_grads = start_angle_grads
    plume_detector.stop_angle_grads = stop_angle_grads
    plume_detector.num_samples = num_samples
    plume_detector.num_steps = num_steps
    plume_detector.speed_of_sound = speed_of_sound
    plume_detector.configure()

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

        plume_detector.binary_device_data_msg = bytes(decoded_message.pack_msg_data())
        plume_detector.process_ping_data()

        sector_intensities[:, angle] = ping_intensities

        # Display data at the end of each sector scan
        if angle == 199:
            scan_num += 1
            print('Last timestamp',timestamp)
            range_m = plume_detector.calc_range(num_samples)
            range_m_int = round(range_m)
            print('Range: ', range)

            seg_sector = 255*plume_detector.seg_scan

            warped = create_sonar_images(sector_intensities)
            seg_warped = create_sonar_images(seg_sector)

            X, db = plume_detector.cluster_seg_scan(seg_warped)
            #plume_detector.cluster_seg_scan_old(seg_warped)
            #clustered_seg_warped = 255*plume_detector.clustered_seg


            # Display images
            fig = plt.figure()
            suptitle = 'Scan ' + str(scan_num)
            plt.suptitle(suptitle)
            plt.title('Start time: ' + start_timestamp + ', End time: ' + timestamp)
            plt.axis('off')

            ax = fig.add_subplot(2, 2, 1)
            #x, y, colours = calc_scatter(plume_detector, sector_intensities)
            #plt.scatter(x,y,c=colours)
            #ax.set_aspect('equal')

            plt.imshow(sector_intensities.transpose(), interpolation='nearest', cmap='jet')
            #ax.set_xticks([0, 0.25*num_samples, 0.5*num_samples, 0.75*num_samples, num_samples],
            #              labels=['0', str(0.25*range_m_int), str(0.5*range_m_int), str(0.75*range_m_int), str(range_m_int)])
            x_label_pos = [0, 0.25*num_samples, 0.5*num_samples, 0.75*num_samples, num_samples]
            x_labels    = ['0', str(0.25*range_m_int), str(0.5*range_m_int), str(0.75*range_m_int), str(range_m_int)]
            ax.set_xticks(x_label_pos, labels=x_labels)

            # Labels and label positions for warped images
            rows, cols = warped.shape[0], warped.shape[1]
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
            #plt.imshow(clustered_seg_warped, interpolation='nearest',cmap='jet')
            #ax.set_xticks(x_label_pos, labels = x_labels)
            #ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            ax.set_xlim([0, 200])
            ax.set_xlim([0, 200])
            ax.set_ylim([200, 0])

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=4,
                )

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=2,
                )




            plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))

            start_timestamp = ""


