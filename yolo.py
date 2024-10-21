import torch
import cv2
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer

from cotracker.predictor import CoTrackerOnlinePredictor

from ultralytics import YOLO
# from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np

# Initialize YOLO and CoTracker models
yolo_model = YOLO("model-400.pt")
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to('cuda' if torch.cuda.is_available() else 'cpu')
# cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def extract_bounding_boxes(results):
    """
    Extract bounding boxes from a list of detection results.

    Parameters:
    results (list): A list of results, each containing multiple bounding box objects.

    Returns:
    list: A list of bounding boxes, where each box is represented as [x_min, y_min, x_max, y_max].
    """
    bounding_boxes = []  # List to store all bounding boxes

    for result in results:
        boxes = result.boxes  # Get the Boxes object for this result

        for box in boxes:  # Iterate over all detections within this result
            # Extract the bounding box coordinates from xyxy (format: [x_min, y_min, x_max, y_max])
            x_min, y_min, x_max, y_max = box.xyxy[0]
            # Convert to a standard Python list and append it to the list
            bounding_boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

    return bounding_boxes

def get_center_points(bounding_boxes, frame_number):
    """
    Calculate the center points of bounding boxes with a frame number prefix.

    Parameters:
    bounding_boxes (list of lists): A list of bounding boxes, each specified as [x_min, y_min, x_max, y_max].
    frame_number (int or float): The frame number to associate with each center point.

    Returns:
    torch.Tensor: A tensor of center points with shape (1, N, 3),
                  where N is the number of bounding boxes, and each point is [frame_number, x_center, y_center].
    """
    # Calculate center points
    center_points = []
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        center_points.append([frame_number, x_center, y_center])

    # Convert to a tensor for easier use with PyTorch models
    center_points_tensor = torch.tensor(center_points, dtype=torch.float32)

    # Add a batch dimension to match CoTracker's expected input shape (1, N, 3)
    center_points_tensor = center_points_tensor.unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        center_points_tensor = center_points_tensor.cuda()

    return center_points_tensor

# Function to process a single frame
def process_frame(frame, frame_number):
    # Predict bounding boxes using YOLO
    img, results = predict_and_detect(yolo_model, frame)
    
    # Extract bounding boxes
    bounding_boxes = extract_bounding_boxes(results)
    
    # Convert bounding boxes to center points (queries) for CoTracker
    queries = get_center_points(bounding_boxes, frame_number)
    
    return img, queries

# Load video frames
video_path = "assets/trim.wmv"
frames = iio.imread(video_path, plugin="FFMPEG")

# Initialize variables
window_frames = []
tracked_frames = []
is_first_step = True

# Iterate over video frames
for frame_number, frame in enumerate(frames):
    img, queries = process_frame(frame, frame_number)

    # Process with CoTracker in chunks of `model.step`
    if frame_number % cotracker.step == 0 and frame_number != 0:
        video_chunk = torch.tensor(np.stack(window_frames[-cotracker.step * 2:])).float().permute(0, 3, 1, 2)[None]
        video_chunk = video_chunk.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        pred_tracks, pred_visibility = cotracker(
            video_chunk, 
            is_first_step=is_first_step, 
            queries=queries if is_first_step else None
        )
        
        is_first_step = False

    window_frames.append(img)
    tracked_frames.append(pred_tracks)

# Save and visualize the tracked results
vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
video_tensor = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)[None]
vis.visualize(video_tensor, tracked_frames, pred_visibility)
