#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
import cv2
import os
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import argparse
import glob
from os.path import join, isdir
from os import mkdir

# Arguments
parser = argparse.ArgumentParser(description='save ros2 bag')
parser.add_argument("--topics", type=str, 
                   default=['/zed/zed_node/left/image_rect_color', 
                          '/zed/zed_node/right/image_rect_color'], 
                   help="topic", nargs="+")
parser.add_argument("--outdir", type=str, default='', help="where to save the images")
parser.add_argument("--inputdir", type=str, 
                   default='/data2/yuhengq/dsta_payload/2024_08_09/run2/zed_bags', 
                   help="the folder for input bag file")
args = parser.parse_args()
print(args)

def process_bag(bag_path, output_path):
    # Initialize the ROS2 node
    rclpy.init()
    
    # Create CV bridge for converting ROS images to OpenCV format
    bridge = CvBridge()
    
    print(f"Opening bag file: {bag_path}")
    
    # Setup reader with sqlite3 fallback
    try:
        # Use the directory path instead of the .mcap file
        bag_dir = os.path.dirname(bag_path) if bag_path.endswith('.mcap') else bag_path
        print(f"Using bag directory: {bag_dir}")
        
        storage_options = StorageOptions(uri=bag_dir, storage_id="mcap")
        converter_options = ConverterOptions(input_serialization_format="cdr",
                                          output_serialization_format="cdr")
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
    except RuntimeError as e:
        print(f"Failed to open with mcap: {e}")
        try:
            # If mcap fails, try sqlite3 format
            storage_options = StorageOptions(uri=bag_dir, storage_id="sqlite3")
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
        except RuntimeError as e:
            print(f"Error opening bag {bag_dir}: {e}")
            return
    
    # Get topic types from metadata
    topic_types = reader.get_all_topics_and_types()
    
    # Print available topics for debugging
    print("\nAvailable topics in bag:")
    for topic_type in topic_types:
        print(f"Topic: {topic_type.name}, Type: {topic_type.type}")
    print()
    
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    
    # Process each requested topic
    for subfolder, topic in zip(["rgb_l", "rgb_r"], args.topics):
        if topic not in type_map:
            print(f"Warning: Topic {topic} not found in bag file")
            continue
            
        folder_path = join(output_path, subfolder)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\nProcessing topic: {topic}")
        print(f"Saving images to: {folder_path}")
        
        # Open timestamp file
        time_txt = open(join(folder_path, "times.txt"), "w")
        
        # Reset reader for each topic
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        msg_count = 0
        
        # Process messages
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            
            if topic_name != topic:
                continue
                
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            
            # Print message encoding for first message
            if msg_count == 0:
                print(f"Message encoding: {msg.encoding}")
            
            # Convert ROS Image message to OpenCV image
            image_np = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            
            # Generate timestamp-based filename
            time_stamp = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}"
            image_name = f"{time_stamp}.png"
            image_path = join(folder_path, image_name)
            
            # Save image and timestamp
            cv2.imwrite(image_path, image_np)
            time_txt.write(f"{time_stamp}\n")
            msg_count += 1
            
            # Print progress
            if msg_count % 100 == 0:
                print(f"Processed {msg_count} images for {topic}")
        
        print(f"Saved {msg_count} images for topic {topic}")
        time_txt.close()
    
    rclpy.shutdown()

def main():
    # Process all bag files in the input directory
    
    bag_files = glob.glob(join(args.inputdir, "*/*.mcap"))  # Changed from *.mcap to *.db3
    if not bag_files:
        print(f"No ROS2 bag files found in {args.inputdir}")
        return
        
    print(f"Found {len(bag_files)} bag files")
    
    for bag_file in bag_files:
        print(f"\nProcessing bag file: {bag_file}")
        
        # Create output directory structure
        bag_name = os.path.basename(bag_file)
        local_path = join(args.outdir, bag_name)
        
        if not isdir(local_path):
            mkdir(local_path)
            mkdir(join(local_path, "rgb_l"))
            mkdir(join(local_path, "rgb_r"))
            
        # Process the bag file
        process_bag(bag_file, local_path)

if __name__ == '__main__':
    main()
