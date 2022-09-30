from decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from src.plume_detector import PPlumeDetector
from sklearn.cluster import DBSCAN
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


def cluster_dbscan(plume_detector):

    # From the image, extract the list of points (detections above the threshold)
    num_points = plume_detector.seg_scan.sum()
    points = np.zeros(shape=(num_points, 2))
    index = 0
    for row in range(plume_detector.seg_img.shape[0]):
        for col in range(plume_detector.seg_img.shape[1]):
            if plume_detector.seg_img[row, col]:
                points[index] = [col, row]
                index = index + 1

    start = time.time()
    print("Start: ", start)

    circle_to_square_area_ratio = math.pi/4
    cluster_min_pixels = round(plume_detector.cluster_min_pixels*circle_to_square_area_ratio)
    db = DBSCAN(eps=plume_detector.window_width_pixels/2, min_samples=cluster_min_pixels).fit(points)

    end = time.time()
    print("End: ", end)
    print("DBSCAN time is ", end - start)

    # Number of clusters in labels, ignoring noise if present.
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    dbscan_img = np.zeros_like(plume_detector.seg_img, dtype=np.uint8)

    unique_labels = set(labels)
    for k in unique_labels:

        class_member_mask = labels == k

        class_points = points[class_member_mask]
        for col, row in class_points:
            # Add 1 since labelling starts at -1, which is noise
            # 'Noise' pixels are then set to 0, and not plotted
            dbscan_img[int(row), int(col)] = k+1

    return dbscan_img

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

    #try:
        #os.mkdir(img_save_path)
    #except OSError as error:
        #print(error)

    # Setup Plume Detector
    plume_detector = PPlumeDetector()
    plume_detector.start_angle_grads = start_angle_grads
    plume_detector.stop_angle_grads = stop_angle_grads
    plume_detector.num_samples = num_samples
    plume_detector.num_steps = num_steps
    plume_detector.speed_of_sound = speed_of_sound
    plume_detector.configure()
    plume_detector.scan_processing_angles = [199] # Over-write default, which is the start and stop angles

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

        # Extract ping data from message & feed in to plume detector
        angle = decoded_message.angle
        ping_intensities = np.frombuffer(decoded_message.data,
                                    dtype=np.uint8)  # Convert intensity data bytearray to numpy array
        plume_detector.binary_device_data_msg = bytes(decoded_message.pack_msg_data())
        plume_detector.process_ping_data()

        # Display data at the end of each sector scan
        if angle == 199:

            # DBSCAN Clustering
            dbscan_img= cluster_dbscan(plume_detector)


            scan_num += 1
            print('Scan:', scan_num)
            print('Last timestamp',timestamp)
            range_m = plume_detector.calc_range(num_samples)
            range_m_int = round(range_m)
            print('Range: ', range)

            # Create warped (polar) images
            warped = plume_detector.create_sonar_image(plume_detector.scan_intensities)

            # Setup plot
            fig = plt.figure()
            suptitle = 'Scan ' + str(scan_num) + ' (Start time: ' + start_timestamp + ', End time: ' + timestamp + ')'
            #plt.suptitle(suptitle)
            plt.axis('off')

            # Labels and label positions for warped images
            rows, cols = warped.shape[0], warped.shape[1]
            x_label_pos = [0, 0.25*cols, 0.5*cols, 0.75*cols, cols]
            x_labels    = [str(range_m_int), str(0.5*range_m_int), '0', str(0.5*range_m_int), str(range_m_int)]
            y_label_pos = [0, 0.25*rows, 0.5*rows, 0.75*rows, rows]
            y_labels    = [str(range_m_int), str(0.5*range_m_int), '0', str(0.5*range_m_int), str(range_m_int)]

            # 1: Original data, warped
            ax = fig.add_subplot(2, 2, 1)
            plt.imshow(warped, interpolation='none', cmap='jet')
            ax.title.set_text('1: Original')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)


            # 2: Segmented data
            ax = fig.add_subplot(2, 2, 2)
            image = plume_detector.seg_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none',cmap='RdYlBu')
            ax.title.set_text('3: Segmented')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 3: Window Clustering
            ax = fig.add_subplot(2, 2, 3)
            image = plume_detector.labelled_clustered_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            ax.title.set_text('3: Window Clustering')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            # 4: DBSCAN Clustering
            ax = fig.add_subplot(2, 2, 4)
            image = dbscan_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            ax.title.set_text('4: DBSCAN Clustering')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            fig.tight_layout()
            plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))

            start_timestamp = ""



