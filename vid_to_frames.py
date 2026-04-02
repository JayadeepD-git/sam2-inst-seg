#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:27:10 2026

@author: djayadeep
"""

import os
import subprocess
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract every Nth frame from a video for SAM2 processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-i", "--input", required=True, help="Path to the input .mp4 video file")
    parser.add_argument("-o", "--output", required=True, help="Directory where extracted frames will be saved")
    
    parser.add_argument("-n", "--interval", type=int, default=30, help="Extract every Nth frame (e.g., 30)")
    parser.add_argument("-q", "--quality", type=int, default=2, help="JPEG quality ")
    
    return parser.parse_args()

def main():
    args = parse_args()

    video_path = args.input
    frame_dir = args.output
    n = args.interval
    quality = args.quality

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)
        print(f"Created directory: {frame_dir}")

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"select='not(mod(n,{n}))',setpts=N/FRAME_RATE/TB",
        '-fps_mode', 'vfr',
        '-q:v', str(quality),
        os.path.join(frame_dir, '%05d.jpg')
    ]

    print(f"Running FFmpeg: Extracting 1 every {n} frames...")

    try:
        subprocess.run(command, check=True)
        
        num_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        print(f"Extracted {num_frames} frames to {frame_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(" Error: ffmpeg not found.", file=sys.stderr)

if __name__ == "__main__":
    main()
