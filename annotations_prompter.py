#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:32:55 2026

@author: djayadeep
"""

import gradio as gr
import json
import os
import shutil
import numpy as np

FOLDER_PATH = "" 
OUTPUT_JSON = ""
TARGET_INDEX =  #mention frame to annotate 

all_files = sorted([f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.png'))])
if not all_files:
    raise FileNotFoundError(f"No images found in {FOLDER_PATH}")

IMAGE_PATH = os.path.join(FOLDER_PATH, all_files[TARGET_INDEX])

clicks = []

def log_click(evt: gr.SelectData, iris_mode):
    x, y = evt.index

    click_type = 1
    if len(clicks) >= 8 and iris_mode == "Negative (Background)":
        click_type = 0

    clicks.append({"coords": [x, y], "label": click_type})

    shapes = []
    def get_pts(start, end): return [c["coords"] for c in clicks[start:end]]
    def get_labels(start, end): return [c["label"] for c in clicks[start:end]]

    if len(clicks) >= 1:
        shapes.append({"label": "Pince", "points": get_pts(0, 4), "labels": get_labels(0, 4)})
    if len(clicks) >= 5:
        shapes.append({"label": "Pince", "points": get_pts(4, 6), "labels": get_labels(4, 6)})
    if len(clicks) >= 7:
        shapes.append({"label": "Needle", "points": get_pts(6, 8), "labels": get_labels(6, 8)})
    if len(clicks) >= 9:
        shapes.append({"label": "Iris", "points": get_pts(8, None), "labels": get_labels(8, None)})

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"start_frame_index": TARGET_INDEX, "shapes": shapes}, f, indent=4)

    status = f"Frame {TARGET_INDEX} | "
    if len(clicks) < 4: 
        msg = status + f"L Pince: {len(clicks)}/4"
    elif len(clicks) < 6: 
        msg = status + f"R Pince: {len(clicks)-4}/2"
    elif len(clicks) < 8: 
        msg = status + f"Needle: {len(clicks)-6}/2"
    else:
        mode = "NEGATIVE" if click_type == 0 else "POSITIVE"
        msg = status + f"Iris {mode}: {len(clicks)-8} pts total (JSON Saved)"

    return msg

def reset_clicks():
    global clicks
    clicks = []
    return "Reset. Start with L Pince (4 clicks)."

with gr.Blocks(title="SAM 2 Annotator") as demo:
    gr.Markdown(f"Current Image: `{all_files[TARGET_INDEX]}`")
    
    with gr.Row():
        img = gr.Image(value=IMAGE_PATH, type="filepath", interactive=False)
        
        with gr.Column():
            out_text = gr.Textbox(label="Status", value="Ready! Start with L Pince.")
            iris_type = gr.Radio(
                ["Positive (Iris)", "Negative (Background)"], 
                value="Positive (Iris)", 
                label="Iris Click Mode"
            )
            btn_reset = gr.Button("Reset and Start Over")

    
    img.select(log_click, [iris_type], out_text)
    
    btn_reset.click(reset_clicks, None, out_text)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
