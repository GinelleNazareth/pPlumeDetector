from decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from src.plume_detector import PPlumeDetector
import numpy as np
import time
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

            # Call functions to create an image of the scan and cluster it
            plume_detector.seg_img = plume_detector.create_sonar_image(plume_detector.seg_scan_snapshot)
            plume_detector.cluster()
            plume_detector.calc_cluster_centers()

            scan_num += 1
            print('Scan:', scan_num)
            print('Last timestamp',timestamp)
            range_m = plume_detector.calc_range(num_samples)
            range_m_int = round(range_m)
            print('Range: ', range)

            # Create warped (polar) images
            warped = plume_detector.create_sonar_image(plume_detector.scan_intensities)
            denoised_warped = plume_detector.create_sonar_image(plume_detector.scan_intensities_denoised)

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
            ax = fig.add_subplot(2, 3, 1)
            plt.imshow(warped, interpolation='none', cmap='jet',vmin=0,vmax=255)
            ax.title.set_text('1: Original')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 2: Denoised data
            ax = fig.add_subplot(2, 3, 2)
            plt.imshow(denoised_warped, interpolation='none',cmap='jet',vmin=0,vmax=255)
            ax.title.set_text('2: Denoised')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 3: Segmented data
            ax = fig.add_subplot(2, 3, 3)
            image = plume_detector.seg_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none',cmap='RdYlBu')
            ax.title.set_text('3: Segmented')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 4: Clustered Cores
            ax = fig.add_subplot(2, 3, 4)
            image = plume_detector.clustered_cores_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none',cmap='RdYlBu')
            ax.title.set_text('4: Clustered Cores')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 5: Labelled Regions
            ax = fig.add_subplot(2, 3, 5)
            image = plume_detector.labelled_regions_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            ax.title.set_text('5: Labelled Regions')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            # Label positions for output images - different because images are larger
            rows, cols = plume_detector.output_img.shape[0], plume_detector.output_img.shape[1]
            x_label_pos = [0, 0.25*cols, 0.5*cols, 0.75*cols, cols]
            y_label_pos = [0, 0.25*rows, 0.5*rows, 0.75*rows, rows]

            # 6: Final Output
            ax = fig.add_subplot(2, 3, 6)
            image = plume_detector.output_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            ax.title.set_text('6: Labelled Clusters')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            fig.tight_layout()
            #plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))

            # # Setup plot
            ax = plt.figure().add_subplot(1,1,1)
            suptitle = 'Scan ' + str(scan_num) + ' (Start time: ' + start_timestamp + ', End time: ' + timestamp + ')'
            plt.suptitle(suptitle)
            #plt.axis('off')

            # 6: Final Output
            image = plume_detector.output_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            plt.show()

            start_timestamp = ""