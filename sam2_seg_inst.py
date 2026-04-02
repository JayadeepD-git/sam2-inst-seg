#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:20:25 2026

@author: djayadeep
"""

''' 
Clone SAM2:
pip install --no-cache-dir -U git+https://github.com/facebookresearch/segment-anything-2.git
Create checkpoint directory
mkdir -p /mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints
Download pretrained model
wget -P /mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
'''

import torch
import numpy as np
import cv2
import os
import json
import shutil
import argparse
from sam2.build_sam import build_sam2_video_predictor

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Tracking V3")
    
    # Paths
    parser.add_argument("-f", "--frames", required=True, help="Path to video frames directory")
    parser.add_argument("-j", "--json", required=True, help="Path to annotation JSON")
    parser.add_argument("-c", "--checkpoint",default="/mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints/sam2_hiera_base_plus.pt", help="Path to SAM2 .pt checkpoint")
    parser.add_argument("-m", "--model_cfg", default="sam2_hiera_b+.yaml", help="Model config file")

    # Output  
    parser.add_argument("-o", "--output_dir", required=True, help="Base output directory")
    
    parser.add_argument(
        "--export_mode", 
        choices=["combined", "individual"], 
        default="combined", 
        help="save masks: 'combined' (1 mask image per frame) or 'individual' (1 folder per object). Default is combined."
    )

    
    return parser.parse_args()

def main():
    args = parse_args()

    overlay_dir = os.path.join(args.output_dir, "overlays")
    mask_base_dir = os.path.join(args.output_dir, "masks")

    for d in [overlay_dir, mask_base_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        
    # Colors (BGR): 1:Green (Left Pince), 2:Blue (Right Pince), 3:Magenta (Needle), 4:Yellow (Iris)
    colors = {1: [0, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255], 4: [0, 255, 255]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device)
    inference_state = predictor.init_state(video_path=args.frames)

    with open(args.json, 'r') as f:
        data = json.load(f)

    start_idx = data.get("start_frame_index", 0)
    frame_names = sorted([f for f in os.listdir(args.frames) if f.lower().endswith(('.jpg', '.png'))])

    pince_count = 0
    for shape in data['shapes']:
        label_name = shape['label']
        pts = np.array(shape['points'], dtype=np.float32)
        lbls = np.array(shape.get('labels', [1] * len(pts)), dtype=np.int32)
        
        if label_name == "Iris": oid = 4
        elif label_name == "Needle": oid = 3
        else:
            pince_count += 1
            oid = pince_count
        predictor.add_new_points_or_box(inference_state, start_idx, oid, pts, lbls)

    video_segments = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        print("Tracking Forward from anchor frame")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_idx):
            video_segments[out_frame_idx] = (out_obj_ids, out_mask_logits)
        print("Tracking Backward from anchor frame")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_idx, reverse=True):
            video_segments[out_frame_idx] = (out_obj_ids, out_mask_logits)


    print(f"Exporting results to {args.output_dir}")
    for idx in sorted(video_segments.keys()):
        obj_ids, mask_logits = video_segments[idx]
        fname = frame_names[idx]
        
        frame = cv2.imread(os.path.join(args.frames, fname))
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        items = sorted(list(zip(obj_ids, mask_logits)), key=lambda x: 0 if int(x[0]) == 4 else int(x[0]))
        
        for oid, logit in items:
            mask = (logit > 0.0).cpu().numpy().squeeze()
            if mask.any():
                overlay[mask] = colors.get(int(oid), [255, 255, 255])
                
                if args.export_mode == "combined":
                    combined_mask[mask] = int(oid)
                else:
                    obj_folder = os.path.join(mask_base_dir, f"obj_{oid}")
                    os.makedirs(obj_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(obj_folder, f"mask_{fname}"), (mask * 255).astype(np.uint8))

        cv2.imwrite(os.path.join(overlay_dir, f"overlay_{fname}"), 
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0))
        
        if args.export_mode == "combined":
            cv2.imwrite(os.path.join(mask_base_dir, f"mask_{fname}"), combined_mask)

    print(f"{args.output_dir} contains 'masks' and 'overlays' folders.")


if __name__ == "__main__":
    main()
