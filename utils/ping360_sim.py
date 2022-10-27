from decode_sensor_binary import PingViewerLogReader
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import time
import pymoos
import re

# 20210305-031345328.bin
# ping360_20210901_123108.bin

# Script settings
start_time = "00:00:30.000"
#start_time = "00:10:00.000"

def on_connect():
    # Do nothing - no events to register for
    print('on connect')
    return True

def on_mail():
    # Do nothing - no events expected
    return

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("file",
                        help="File that contains PingViewer sensor log file.")
    args = parser.parse_args()

    # Open log and begin processing
    log = PingViewerLogReader(args.file)

    start_timestamp = ""
    start_time_obj = datetime.strptime(start_time, "%H:%M:%S.%f")

    comms = pymoos.comms()
    comms.set_on_connect_callback(on_connect)
    comms.set_on_mail_callback(on_mail)
    comms.run('localhost', 9000, 'ping360_sim')

    first = True

    for index, (timestamp, decoded_message) in enumerate(log.parser()):

        timestamp = re.sub(r"\x00", "", timestamp) # Strip any extra \0x00 (null bytes)
        time_obj = datetime.strptime(timestamp,"%H:%M:%S.%f")

        # Skip to start time
        if time_obj < start_time_obj:
            continue

        if first:
            first = False

            # TODO: Read from ping 360 message
            comms.notify('SONAR_NUMBER_OF_SAMPLES', 1200, pymoos.time())
            comms.notify('SONAR_NUM_STEPS', 1, pymoos.time())
            comms.notify('SONAR_START_ANGLE_GRADS', 0, pymoos.time())
            comms.notify('SONAR_STOP_ANGLE_GRADS', 399, pymoos.time())
            comms.notify('SONAR_TRANSMIT_ENABLE', 1, pymoos.time())
            comms.notify('SONAR_SPEED_OF_SOUND', 1500, pymoos.time())

        # Extract ping data from message
        angle = decoded_message.angle
        ping_intensities = np.frombuffer(decoded_message.data,
                                    dtype=np.uint8)  # Convert intensity data bytearray to numpy array
        binary_device_data_msg = bytes(decoded_message.pack_msg_data())
        comms.notify_binary('SONAR_PING_DATA', binary_device_data_msg, pymoos.time())

        time.sleep(0.05)







