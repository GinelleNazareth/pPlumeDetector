from utils.decode_sensor_binary import PingViewerLogReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from utils.dbscan.plume_detector import PPlumeDetector
import numpy as np
import os
import re
from matplotlib import rcParams
import cv2 as cv
import copy

# Modified version of visualize script, for generating plots for papers
# To use script, set file name and ensure that script parameters match sonar data

rcParams['font.family'] = 'serif'

# 20210305-031345328.bin - Bubble plume data collected by JH
# ping360_20210901_123108.bin
# ping360_20230131_175845.bin - AUV data (saw bottom?)

# Script settings
# start_time = "00:00:00.000"
# start_time = "18:12:30.000"
#start_time = "18:31:00.000" # Start of major detections
# start_time = "18:16:00.000"

# start_time = "00:00:30.000"
start_time = "00:10:00.000" # For JH data

# Script parameters
# start_angle_grads = 33
# stop_angle_grads = 167
# num_samples = 1200
# range_m = 50
# num_steps = 1
# speed_of_sound = 1500
# lat_origin = 47.389271
# long_origin = -53.134431

# For JH data
start_angle_grads = 120
stop_angle_grads = 280
num_samples = 1200
range_m = 5
num_steps = 1
speed_of_sound = 1500
lat_origin = 71.376350
long_origin = -70.076860


def create_sonar_image(sector_intensities, image_width_pixels):
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
    radius = int(image_width_pixels / 2)
    warp_flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
    warped_image = cv.warpPolar(sector_intensities_mod, center=(radius, radius), maxRadius=radius,
                                dsize=(2 * radius, 2 * radius),
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
    plume_detector = PPlumeDetector()
    plume_detector.start_angle_grads = start_angle_grads
    plume_detector.stop_angle_grads = stop_angle_grads
    plume_detector.num_samples = num_samples
    plume_detector.range_m = range_m
    plume_detector.num_steps = num_steps
    plume_detector.speed_of_sound = speed_of_sound
    plume_detector.configure()
    # plume_detector.scan_processing_angles = [399] # Over-write default, which is the start and stop angles
    plume_detector.lat_origin = lat_origin
    plume_detector.long_origin = long_origin
    plume_detector.configure()

    # Open log and begin processing
    log = PingViewerLogReader(args.file)

    sector_intensities = np.zeros((num_samples, 400), dtype=np.uint8)
    scan_num = 0
    start_timestamp = ""
    start_time_obj = datetime.strptime(start_time, "%H:%M:%S.%f")

    fig = plt.figure()

    # Create polar grid axes with trasparency
    axes_coords = [0, 0, 1, 1]  # plotting full width and height
    ax_polar = fig.add_axes(axes_coords, projection='polar', label="ax polar")
    ax_polar.set_ylim(0, 50)
    ax_polar.set_yticks([10,20], labels = ['10m', '20m'])
    # Change the fontsize of the radial labels
    radial_labels = ax_polar.get_yticklabels()
    for label in radial_labels:
        label.set_size(6)  # Set the desired fontsize

    ax_polar.set_rlabel_position(118)
    ax_polar.set_xlim(0, 2*np.pi)
    ax_polar.set_xticks([(180-62)*np.pi/180, (180-20)*np.pi/180, (180+20)*np.pi/180, (180+62)*np.pi/180])

    ax_polar.grid(True, color='white', linewidth = 0.3, alpha=0.5)
    plt.setp(ax_polar.spines.values(), color='white')
    image_name = "Gridlines White"
    plt.savefig(os.path.join(img_save_path, image_name), dpi=400, transparent=True)

    ax_polar.grid(True, color='black', linewidth = 0.3, alpha=0.5)
    plt.setp(ax_polar.spines.values(), color='white')
    image_name = "Gridlines Black "
    plt.savefig(os.path.join(img_save_path, image_name), dpi=400, transparent=True)

    ax_polar.grid(True, color='white', linewidth = 0.3, alpha=0.0)
    plt.setp(ax_polar.spines.values(), color='white')
    image_name = "Gridlines Transparent"
    plt.savefig(os.path.join(img_save_path, image_name), dpi=400, transparent=True)

    plt.clf()
    ax_polar.cla()

    for index, (timestamp, decoded_message) in enumerate(log.parser()):

        timestamp = re.sub(r"\x00", "", timestamp)  # Strip any extra \0x00 (null bytes)
        time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")

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
        if angle == 399: #For JH data
        #if angle == start_angle_grads or angle == stop_angle_grads:
            # Call functions to create an image of the scan and cluster it
            plume_detector.seg_img = plume_detector.create_sonar_image(plume_detector.seg_scan_snapshot)
            plume_detector.cluster()
            plume_detector.get_cluster_center_nav()
            plume_detector.georeference_clusters()
            plume_detector.output_sorted_cluster_centers()
            plume_detector.create_output_image()

            scan_num += 1
            print('Scan:', scan_num)
            print('Last timestamp', timestamp)
            range_m_int = round(range_m)
            print('Range: ', range)

            # Create warped (polar) images
            warped = create_sonar_image(plume_detector.scan_intensities, 1200)
            denoised_warped = create_sonar_image(plume_detector.scan_intensities_denoised, 600)
            segmented = create_sonar_image(plume_detector.seg_scan_snapshot, 600)

            # Setup axes
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 1: Original data, warped
            ax.imshow(warped, interpolation='none', cmap='jet', vmin=0, vmax=255)
            image_name = "Scan_" + str(scan_num) + "_Image_1_Original"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            plt.clf()
            ax.cla()
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 2: Denoised data
            ax.imshow(denoised_warped, interpolation='none', cmap='jet', vmin=0, vmax=255)
            image_name = "Scan_" + str(scan_num) + "_Image_2_Denoised"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            plt.clf()
            ax.cla()
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 3: Segmented data
            image = segmented.astype(float)
            image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            ax.imshow(image, interpolation='none', cmap='RdYlBu')
            image_name = "Scan_" + str(scan_num) + "_Image_3_Segmented"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            plt.clf()
            ax.cla()
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 4: Segmented & Downsampled Data
            image = plume_detector.seg_img.astype(float)
            image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            ax.imshow(image, interpolation='none', cmap='RdYlBu')
            image_name = "Scan_" + str(scan_num) + "_Image_4_Downsampled"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            plt.clf()
            ax.cla()
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 5: Cluster Regions
            #image = plume_detector.cluster_regions_img.astype(float)
            #image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            #ax.imshow(image, interpolation='none', cmap='RdYlBu')
            #image_name = "Scan_" + str(scan_num) + "_Image_5_Cluster_Regions"
            #plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            #plt.clf()
            #ax.cla()
            #axes_coords = [0, 0, 1, 1]  # plotting full width and height
            #ax = fig.add_axes(axes_coords)
            #ax.axis('off')

            # 6: Labelled Regions
            #image = plume_detector.labelled_regions_img.astype(float)
            # Increment non-zero pixel values.'1'is used for cluster circles and centers in the output image
            #image = np.where(image > 0, image + 1, image)
            #image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            #ax.imshow(image, interpolation='none', cmap='Dark2', vmin=1, vmax=8)
            #image_name = "Scan_" + str(scan_num) + "_Image_6_Labelled_Regions"
            #plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            #plt.clf()
            #ax.cla()
            #axes_coords = [0, 0, 1, 1]  # plotting full width and height
            #ax = fig.add_axes(axes_coords)
            #ax.axis('off')

            # 7: Labelled Clusters
            image = plume_detector.labelled_clustered_img.astype(float)
            # Increment non-zero pixel values.'1'is used for cluster circles and centers in the output image
            image = np.where(image > 0, image + 1, image)
            image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            ax.imshow(image, interpolation='none', cmap='Dark2', vmin=1, vmax=8)
            image_name = "Scan_" + str(scan_num) + "_Image_7_Labelled_Clusters"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # Reset plot and axes
            plt.clf()
            ax.cla()
            axes_coords = [0, 0, 1, 1]  # plotting full width and height
            ax = fig.add_axes(axes_coords)
            ax.axis('off')

            # 8: Final Output
            image = plume_detector.output_img.astype(float)
            image = np.where(image == 1, 8, image) # Make circles grey
            image[image == 0] = np.nan  # Set zeroes to nan so that they are not plotted
            ax.imshow(image, interpolation='none', cmap='Dark2', vmin=1, vmax=8)
            image_name = "Scan_" + str(scan_num) + "_Image_8_Final_Output"
            plt.savefig(os.path.join(img_save_path, image_name), dpi=400)

            # fig.tight_layout()
            # #plt.show()
            # #plt.savefig(os.path.join(img_save_path, suptitle))
            # image_name = "Clustering_Steps_Scan_" + str(scan_num)
            # plt.savefig(os.path.join(img_save_path, image_name), dpi=400)
            # plt.close(fig)

            start_timestamp = ""
