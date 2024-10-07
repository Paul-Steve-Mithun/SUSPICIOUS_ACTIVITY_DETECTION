from ultralytics import YOLO
import time
import torch
import cv2
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar, VERTICAL, RIGHT, Y
from PIL import Image, ImageTk

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker


deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=80)

# Initialize YOLO model and DeepSort tracker
model = YOLO("person_gun.pt")


# Tkinter window setup
window = tk.Tk()
window.title("Real-time Object Detection")
window.geometry("1000x800")

# Global variables
cap = None
running = False  # To control the webcam feed loop
panel = None  # Panel for video display
info_text = None  # Text widget to display inference information

# Static variables to track changes
current_person_count = 0
current_camera_view_status = "Normal"
current_running_status = "No Running Detected"
alert_person_ids = []

# Variables for tracking
counter, fps, elapsed = 0, 1, 0  # Initialize fps to 1 to avoid division by zero
start_time = time.perf_counter()
unique_track_ids = set()
track_labels = {}
track_times = {}
track_positions = {}
obstruction_frames = 0
obstruction_blocked = False
running_threshold = 0.5
obstruction_threshold = 0.60
running_counters = {}  # Dictionary to track consecutive frames of running detection


def start_detection():
    """Function to start the object detection process."""
    global cap, running, fps
    if not running:
        cap = cv2.VideoCapture(0)  # Start the webcam feed
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Attempt to get the FPS; default to 30 if it fails
        else:
            fps = 30  # Default value in case of failure
        running = True
        update_frame()

def stop_detection():
    """Function to stop the object detection process."""
    global cap, running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def reset_alerts():
    """Function to reset all alert notifications."""
    global alert_person_ids
    alert_person_ids.clear()
    update_static_info_display()

def update_frame():
    """Function to read a frame from the webcam, process it, and update static variables if changes are detected."""
    global cap, panel, counter, fps, elapsed, start_time
    global unique_track_ids, track_labels, track_times, track_positions, obstruction_frames, obstruction_blocked
    global alert_person_ids, current_person_count, current_camera_view_status, current_running_status
    global running_counters  # Add this line
    alert_person_ids.clear()

    if running and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            og_frame = frame.copy()
            results = model(frame, device=0, classes=0, conf=0.75)
            active_track_ids = set()

            for result in results:
                boxes = result.boxes
                cls = boxes.cls.tolist()
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh.cpu().numpy()

            tracks = tracker.update(bboxes_xywh, conf, og_frame)

            # Reset running status
            new_running_status = "No Running"

            for track in tracker.tracker.tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_tlbr()
                w = x2 - x1
                h = y2 - y1

                # Define color for bounding box
                red_color = (0, 0, 255)
                blue_color = (255, 0, 0)
                green_color = (0, 255, 0)
                color_id = track_id % 3
                color = red_color if color_id == 0 else blue_color if color_id == 1 else green_color
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                if track_id not in track_labels:
                    track_labels[track_id] = "Person"
                    track_times[track_id] = 0
                    track_positions[track_id] = (x1, y1)
                    running_counters[track_id] = 0  # Initialize running counter

                track_times[track_id] += 1
                prev_x1, prev_y1 = track_positions[track_id]
                displacement = np.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)

                # Check to avoid division by zero
                if fps > 0:
                    speed = displacement / fps
                else:
                    speed = 0

                track_positions[track_id] = (x1, y1)

                # Determine if the person is running
                if speed > running_threshold and w * h > 5000:
                    running_counters[track_id] += 1  # Increment counter for consecutive running frames
                    if running_counters[track_id] > fps/2:  # More than 1 second of running
                        track_labels[track_id] = "Running"
                        new_running_status = "Running Detected"
                else:
                    running_counters[track_id] = 0  # Reset counter if no running detected
                    track_labels[track_id] = "Person"

                total_seconds = track_times[track_id] / fps if fps > 0 else 0
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)

                 # Trigger alert only after 60 seconds have passed
                if total_seconds > 60 and track_id not in alert_person_ids:
                    alert_person_ids.append(track_id)  # Append only unique IDs

                cv2.putText(og_frame, f"{track_labels[track_id]} {minutes:02}:{seconds:02}", 
                           (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                active_track_ids.add(track_id)

            unique_track_ids.intersection_update(active_track_ids)
            unique_track_ids.update(active_track_ids)

            # Update static variables if changes are detected
            person_count = len(unique_track_ids)
            if person_count != current_person_count:
                current_person_count = person_count
                update_static_info_display()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            if blur < 50:
                obstruction_frames += 1
                if obstruction_frames > fps * 2:
                    obstruction_blocked = True
            else:
                obstruction_frames = 0
                obstruction_blocked = False

            # Update camera view status
            new_camera_view_status = "Blocked" if obstruction_blocked else "Normal"
            if new_camera_view_status != current_camera_view_status:
                current_camera_view_status = new_camera_view_status
                update_static_info_display()

            # Update running status
            if new_running_status != current_running_status:
                current_running_status = new_running_status
                update_static_info_display()

            current_time = time.perf_counter()
            elapsed = (current_time - start_time)
            counter += 1
            if elapsed > 1:
                fps = counter / elapsed
                counter = 0
                start_time = current_time

            # Convert the image to RGB (Tkinter-compatible format) and display it
            img = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            if panel is None:
                panel = Label(window)
                panel.pack(padx=10, pady=10)
            panel.imgtk = imgtk  # Keep a reference to avoid garbage collection
            panel.config(image=imgtk)  # Update the image in the panel
            window.after(10, update_frame)  # Schedule the next frame update


def update_static_info_display():
    """Function to display static variables without real-time changes."""
    global info_text, current_person_count, current_camera_view_status, current_running_status, alert_person_ids
    
    # Enable editing in the text widget
    info_text.config(state=tk.NORMAL)
    
    # Clear the current text
    info_text.delete(1.0, tk.END)

    # Change color to red if the person count exceeds 2
    if current_person_count > 2:
        info_text.tag_configure("count_exceed", foreground="red")
        info_text.insert(tk.END, f"Person Count: {current_person_count}\n", "count_exceed")
    else:
        info_text.insert(tk.END, f"Person Count: {current_person_count}\n")

    # Change color to red if the camera view is blocked
    if current_camera_view_status == "Blocked":
        info_text.tag_configure("blocked", foreground="red")
        info_text.insert(tk.END, f"Camera View: {current_camera_view_status}\n", "blocked")
    else:
        info_text.insert(tk.END, f"Camera View: {current_camera_view_status}\n")

    info_text.insert(tk.END, f"Running Status: {current_running_status}\n")

    # Display alert for prolonged stay
    if alert_person_ids:
        info_text.insert(tk.END, f"Prolonged Stay - Person IDs: {', '.join(map(str, alert_person_ids))} over 1 minute\n")
    else:
        info_text.insert(tk.END, "Prolonged Stay: NIL\n")

    # Disable editing in the text widget
    info_text.config(state=tk.DISABLED)
    
    # Automatically scroll to the end
    info_text.see(tk.END)



def on_close():
    """Function to handle window close event."""
    stop_detection()
    window.destroy()


# UI Controls and Layout
button_frame = Frame(window)
button_frame.pack(side="top", pady=10)

start_button = Button(button_frame, text="Start Detection", command=start_detection)
start_button.pack(side="left", padx=5)

stop_button = Button(button_frame, text="Stop Detection", command=stop_detection)
stop_button.pack(side="left", padx=5)

reset_button = Button(button_frame, text="Reset Alerts", command=reset_alerts)
reset_button.pack(side="left", padx=5)

# Add a frame for displaying inference overlays
info_frame = Frame(window)
info_frame.pack(side="bottom", fill="x")

scrollbar = Scrollbar(info_frame, orient=VERTICAL)
scrollbar.pack(side=RIGHT, fill=Y)

info_text = Text(info_frame, height=8, wrap=tk.WORD, yscrollcommand=scrollbar.set)
info_text.pack(side="left", fill="x", padx=5, pady=5)
scrollbar.config(command=info_text.yview)

window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()