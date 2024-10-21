# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    
    # Argument to specify how many frames to wait before stopping the tracker
    parser.add_argument(
        "--wait_frames",
        type=int,
        default=10,
        help="Number of frames to wait when an object becomes invisible before stopping the tracker.",
    )

    # New argument to specify the path to the queries JSON file
    parser.add_argument(
        "--queries_path",
        type=str,
        default=None,
        help="Path to a JSON file containing the queries (frame number, x, y coordinates)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    # Load queries from the JSON file if provided
    if args.queries_path:
        with open(args.queries_path, "r") as f:
            queries_data = json.load(f)

        # Convert queries into a tensor (assuming JSON structure is a list of lists)
        queries = torch.tensor(queries_data)
        queries = queries.to(DEFAULT_DEVICE)
        if len(queries.shape) == 2:  # Reshape to (1, N, 3) if necessary
            queries = queries.unsqueeze(0)

    else:
        queries = None

    print("Loaded queries:", queries)

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    # A dictionary to keep track of how long each object has been invisible
    invisible_counter = {}

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, queries, wait_frames):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        pred_tracks, pred_visibility = model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

        # Track visibility and remove trackers for invisible objects beyond wait_frames
        if pred_visibility is not None:
            # Initialize counters for first step
            if is_first_step:
                for i in range(pred_visibility.shape[2]):
                    invisible_counter[i] = 0

            for i in range(pred_visibility.shape[2]):  # Iterate over objects
                # If the object is visible, reset its counter
                if pred_visibility[:, :, i].sum() > 0:
                    invisible_counter[i] = 0
                else:
                    # Increment the invisible counter if the object is not visible
                    invisible_counter[i] += 1

                # Stop tracking if the object has been invisible for more than wait_frames
                if invisible_counter[i] >= wait_frames:
                    pred_tracks[:, :, i] = -1  # Remove by setting tracks to invalid values

            # Filter out the removed tracks (those marked with -1)
            valid_tracks = pred_tracks[:, :, (pred_tracks[0, 0, :, 0] != -1)]
            valid_visibility = pred_visibility[:, :, (pred_tracks[0, 0, :, 0] != -1)]

            return valid_tracks, valid_visibility

        return pred_tracks, pred_visibility

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                queries=queries if is_first_step else None,  # Pass queries only for the first step
                wait_frames=args.wait_frames,  # Pass the wait_frames argument
            )
            is_first_step = False
        window_frames.append(frame)

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        queries=queries,
        wait_frames=args.wait_frames  # Pass the wait_frames argument
    )
    print("Tracks", pred_tracks)
    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )
