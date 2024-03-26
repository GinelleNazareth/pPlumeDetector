from utils.decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from utils.dbscan.plume_detector import PPlumeDetector
import numpy as np
import os
import re
from matplotlib import rcParams
import csv

# To use script, set file name and ensure that script parameters match sonar data

rcParams['font.family'] = 'serif'

# 20210305-031345328.bin - Bubble plume data collected by JH
# ping360_20210901_123108.bin
# ping360_20230131_175845.bin - AUV data (saw bottom?)

#start_time = "18:12:30.000"
#start_time = "18:30:00.000"
#start_time = "18:16:00.000"

#start_time = "00:00:30.000"
start_time = "00:10:00.000"

# Script parameters
# start_angle_grads = 33
# stop_angle_grads = 167
# num_samples = 1200
# range_m = 50
# num_steps = 1
# speed_of_sound = 1500
# lat_origin = 47.389271
# long_origin = -53.134431

# Script parameters
start_angle_grads = 120
stop_angle_grads = 280
num_samples = 1200
range_m = 5
num_steps = 1
speed_of_sound = 1500
lat_origin = 71.376350
long_origin = -70.076860

# Baffin data files
#ping360_20230914_161057_dive1.bin
#ping360_20230914_165855_dive2.bin
#ping360_20230914_192204_dive3.bin
#ping360_20230915_123753_dive1.bin
#ping360_20230915_135657_dive2.bin
#ping360_20230915_163658_dive3.bin
#ping360_20230915_174350_dive4.bin
#ping360_20230915_181058_dive5.bin
#ping360_20230915_181658_dive6.bin
#ping360_20230915_183112_dive7.bin
#ping360_20230915_183952_dive8.bin
#ping360_20230915_194741_dive9.bin
#ping360_20230916_130117_dive3.bin
#ping360_20230916_131845_dive4.bin
#ping360_20230916_134647_dive5.bin
#ping360_20230916_173640_dive7.bin
#ping360_20230917_172804_dive1.bin
#ping360_20230917_173927_dive2.bin
#ping360_20230917_175546_dive3.bin
#ping360_20230917_191702_dive4.bin
#ping360_20230917_195638_dive5.bin
#ping360_20230917_211400_dive6.bin
#ping360_20230918_162410_dive1.bin
#ping360_20230918_190244_dive2.bin


# Script settings
#start_time = "00:00:00.000"
#start_time = "17:05:30.000"


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
    plume_detector.range_m = range_m
    plume_detector.num_steps = num_steps
    plume_detector.speed_of_sound = speed_of_sound
    plume_detector.configure()
    #plume_detector.scan_processing_angles = [399] # Over-write default, which is the start and stop angles
    plume_detector.lat_origin = lat_origin
    plume_detector.long_origin = long_origin
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

        # Extract ping data from message & feed in to plume detector
        angle = decoded_message.angle
        #if angle >= start_angle_grads and angle <= stop_angle_grads:
        ping_intensities = np.frombuffer(decoded_message.data,
                                            dtype=np.uint8)  # Convert intensity data bytearray to numpy array
        plume_detector.binary_device_data_msg = bytes(decoded_message.pack_msg_data())
        plume_detector.process_ping_data()

        # Display data at the end of each sector scan
        if angle == 399:
        #if angle == start_angle_grads or angle == stop_angle_grads:

            scan_num += 1
            print('Scan:', scan_num)
            print('Last timestamp', timestamp)
            range_m_int = round(range_m)
            print('Range: ', range)

            if scan_num < 2:
                continue
            elif scan_num > 2:
                exit()

            # Call functions to create an image of the scan and cluster it
            plume_detector.seg_img = plume_detector.create_sonar_image(plume_detector.seg_scan_snapshot)
            plume_detector.cluster()
            plume_detector.get_cluster_center_nav()
            plume_detector.georeference_clusters()
            plume_detector.output_sorted_cluster_centers()
            plume_detector.create_output_image()

            # Open the file in append mode
            with open('computation_time.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                # Write the new data
                writer.writerow([plume_detector.window_width_pixels, plume_detector.labelled_clustered_img.max(),
                                 plume_detector.clustering_time_secs])

            # Create warped (polar) images
            warped = plume_detector.create_sonar_image(plume_detector.scan_intensities)
            denoised_warped = plume_detector.create_sonar_image(plume_detector.scan_intensities_denoised)

            # Setup plot
            fig = plt.figure()
            #suptitle = 'Scan ' + str(scan_num) + ' (Start time: ' + start_timestamp + ', End time: ' + timestamp + ')'
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
            plt.imshow(warped, interpolation='none', cmap='jet',vmin=0,vmax=255)
            ax.title.set_text('1: Original')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 2: Denoised data
            #ax = fig.add_subplot(2, 3, 2)
            #plt.imshow(denoised_warped, interpolation='none',cmap='jet',vmin=0,vmax=255)
            #ax.title.set_text('2: Denoised')
            #ax.set_xticks(x_label_pos, labels = x_labels)
            #ax.set_yticks(y_label_pos, labels= y_labels)

            # 2: Segmented data
            ax = fig.add_subplot(2, 2, 2)
            image = plume_detector.seg_img.astype(float)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            plt.imshow(image, interpolation='none',cmap='RdYlBu')
            ax.title.set_text('2: Segmented')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)

            # 3: Labelled Clusters
            ax = fig.add_subplot(2, 2, 3)
            image = plume_detector.labelled_clustered_img.astype(float)
            # Increment non-zero pixel values.'1'is used for cluster circles and centers in the output image
            image = np.where(image > 0, image + 1, image)
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            #plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            plt.imshow(image, interpolation='none', cmap='Dark2', vmin = 1, vmax = 8)
            ax.title.set_text('3: Labelled Clusters')
            ax.set_xticks(x_label_pos, labels = x_labels)
            ax.set_yticks(y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            # Label positions for output images - different because images are larger
            rows, cols = plume_detector.output_img.shape[0], plume_detector.output_img.shape[1]
            output_x_label_pos = [0, 0.25*cols, 0.5*cols, 0.75*cols, cols]
            output_y_label_pos = [0, 0.25*rows, 0.5*rows, 0.75*rows, rows]

            # 4: Final Output
            ax = fig.add_subplot(2, 2, 4)
            image = plume_detector.output_img.astype(float)
            image = np.where(image == 1, 8, image)  # Make circles grey
            image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
            #plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
            plt.imshow(image, interpolation='none', cmap='Dark2', vmin=1, vmax=8)
            ax.title.set_text('4: Labelled Clusters')
            ax.set_xticks(output_x_label_pos, labels = x_labels)
            ax.set_yticks(output_y_label_pos, labels= y_labels)
            ax.set_aspect('equal')

            fig.tight_layout()
            #plt.show()
            #plt.savefig(os.path.join(img_save_path, suptitle))
            image_name = "Clustering_Steps_Scan_" + str(scan_num)
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)
            plt.close(fig)


            # Setup plot
            fig = plt.figure()
            plt.suptitle('Scan ' + str(scan_num))
            title = 'Start time: ' + start_timestamp + ', End time: ' + timestamp + ', '
            title = title + str(plume_detector.num_clusters) + ' Clusters'
            plt.title(title)
            plt.axis('off')

            #fig = plt.figure()
            #suptitle = 'Scan ' + str(scan_num) + ' (Start time: ' + start_timestamp + ', End time: ' + timestamp + ')'
            #plt.suptitle(suptitle)
            #plt.axis('off')

            # 1: Original data, warped
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(warped, interpolation='none', cmap='jet', vmin=0, vmax=255)
            ax.title.set_text('Original')
            ax.set_xticks(x_label_pos, labels=x_labels)
            ax.set_yticks(y_label_pos, labels=y_labels)

            # 2: Plot clusters if num clusters > 0
            if plume_detector.num_clusters > 0:
                ax = fig.add_subplot(1, 2, 2)
                image = plume_detector.output_img.astype(float)
                image[image==0] = np.nan # Set zeroes to nan so that they are not plotted
                plt.imshow(image, interpolation='none', cmap='nipy_spectral', vmin=0)
                ax.title.set_text('Labelled Clusters')
                ax.set_xticks(output_x_label_pos, labels=x_labels)
                ax.set_yticks(output_y_label_pos, labels=y_labels)
                ax.set_aspect('equal')

            #plt.show()
            image_name = "Clustering_Overview_Scan_" + str(scan_num)
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)
            plt.close(fig)

            if scan_num == 80:
                break

            start_timestamp = ""