import os
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import cv2  # type: ignore[import]
import numpy as np
from scipy import spatial  # type: ignore[import]
from typer import Argument, Option, Typer  # type: ignore[import]

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle", "person", "car", "motorbike", "bus", "truck", "train"]

# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416


class Point(NamedTuple):
    x: int
    y: int


class Line(NamedTuple):
    a: Point
    b: Point


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class Cross(NamedTuple):
    vtype: str
    ts: str
    ID: int


def is_right(L: Line, p: Point) -> bool:
    return ((L.b.x - L.a.x) * (p.y - L.a.y) - (L.b.y - L.a.y) * (p.x - L.a.x)) < 0


def displayVehicleCount(
    frame: np.ndarray,
    vehicle_count: int,
    crossers: list[Cross],
) -> None:
    """
    PURPOSE: Displays the vehicle count on the top-left corner of the frame
    PARAMETERS: Frame on which the count is displayed, the count number of vehicles
    RETURN: N/A
    """
    num_bikes = len(list(filter(lambda x: x.vtype == "bicycle", crossers)))
    num_person = len(list(filter(lambda x: x.vtype == "person", crossers)))
    num_other = len(crossers) - num_bikes - num_person
    cv2.putText(
        frame,
        f"Detected: {vehicle_count} // Bikes: {num_bikes} // Person: {num_person} // Other: {num_other}",
        (20, 20),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


def displayFPS(
    start_time: int, num_frames: int, last_num_frames: int
) -> tuple[int, int, int]:
    """
    PURPOSE: Displaying the FPS of the detected video
    PARAMETERS: Start time of the frame, number of frames within the same second
    RETURN: New start time, new number of frames
    """
    current_time = int(time.time())
    if num_frames > (last_num_frames + 10):
        os.system("clear")  # Equivalent of CTRL+L on the terminal
        fps = (num_frames - last_num_frames) / (current_time - start_time)
        print(f"FPS: {fps:.2f}")
        last_num_frames = num_frames
        start_time = current_time
    return start_time, num_frames, last_num_frames


def drawDetectionBoxes(
    idxs: np.ndarray,
    boxes: list[Box],
    classIDs: list[int],
    confidences: list[float],
    frame: np.ndarray,
    labels: list[str],
    colors: list[list[int]],
) -> None:
    """
    PURPOSE: Draw all the detection boxes with a green dot at the center
    RETURN: N/A
    """
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i].x, boxes[i].y)
            (w, h) = (boxes[i].w, boxes[i].h)

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(
                frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            # Draw a green dot in the middle of the box
            cv2.circle(
                frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2
            )


def initializeVideoWriter(  # type: ignore[no-any-unimported]
    video_width: int,
    video_height: int,
    videoStream: cv2.VideoCapture,
    outputVideoPath: Path,
) -> cv2.VideoWriter:
    """
    PURPOSE: Initializing the video writer with the output video path and the same number
    of fps, width and height as the source video
    PARAMETERS: Width of the source video, Height of the source video, the video stream
    RETURN: The initialized video writer
    """
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(
        str(outputVideoPath), fourcc, sourceVideofps, (video_width, video_height), True
    )


def boxInPreviousFrames(
    previous_frame_detections: list[dict[Point, int]],
    current_detections: dict[Point, int],
    current_point: Point,
    max_dist: int,
) -> int:
    """
    PURPOSE: Identifying if the current box was present in the previous frames
    PARAMETERS: All the vehicular detections of the previous frames,
                the coordinates of the box of previous detections
    RETURN: True if the box was current box was present in the previous frames;
            False if the box was not present in the previous frames
    """
    # centerX, centerY, width, height = current_box
    x, y = current_point
    dist = np.inf  # Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if (
            len(coordinate_list) == 0
        ):  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([current_point])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if dist > max_dist:
        return -1

    # Keeping the vehicle ID constant
    return previous_frame_detections[frame_num][coord]


def count_vehicles(
    labels: list[str],
    idxs: np.ndarray,
    boxes: list[Box],
    classIDs: list[int],
    vehicle_count: int,
    previous_frame_detections: list[dict[Point, int]],
    frame: np.ndarray,
    type_counts: dict[str, int],
    vehicle_types: dict[int, str],
    use_x: bool,
) -> tuple[int, dict[Point, int], dict[str, int], dict[int, str]]:
    current_detections: dict[Point, int] = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            box = boxes[i]
            x, y = box.x, box.y
            w, h = box.w, box.h

            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            vehicle_type = labels[classIDs[i]]
            if vehicle_type in list_of_vehicles:
                current_detections[Point(centerX, centerY)] = vehicle_count
                prev_id = boxInPreviousFrames(
                    previous_frame_detections,
                    current_detections,
                    Point(centerX, centerY),
                    max_dist=max(w, h) / 2,
                )
                if prev_id >= 0:
                    current_detections[Point(centerX, centerY)] = prev_id
                else:
                    vehicle_count += 1
                    type_counts[vehicle_type] += 1

                # Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection
                ID = current_detections[Point(centerX, centerY)]
                vehicle_types[ID] = vehicle_type
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if list(current_detections.values()).count(ID) > 1:
                    current_detections[Point(centerX, centerY)] = vehicle_count
                    vehicle_count += 1
                    type_counts[vehicle_type] += 1

                # Display the ID at the center of the box
                if use_x:
                    cv2.putText(
                        frame,
                        str(ID),
                        (centerX, centerY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [0, 0, 255],
                        2,
                    )

    return vehicle_count, current_detections, type_counts, vehicle_types


def process(
    labels: list[str],
    colors: list[list[int]],
    weights: Path,
    config: Path,
    video: Path,
    output: Path | bool,
    preDefinedConfidence: float,
    preDefinedThreshold: float,
    use_gpu: bool,
    display_video: bool,
) -> None:
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(str(config), str(weights))

    # Using GPU if flag is passed
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    videoStream = cv2.VideoCapture(str(video))
    video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Specifying coordinates for a default line
    cross_line = Line(Point(0, 0), Point(video_width, video_height))

    # Initialization
    previous_frame_detections: list[dict[Point, int]] = [
        {Point(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)
    ]
    vehicle_locs: dict[int, bool] = {}
    crossers: list[Cross] = []
    type_counts: dict[str, int] = defaultdict(int)
    vehicle_types: dict[int, str] = {}

    num_frames, last_num_frames, vehicle_count = 0, 0, 0
    if output and isinstance(output, Path):
        writer = initializeVideoWriter(video_width, video_height, videoStream, output)
    start_time = int(time.time())
    # loop over frames from the video file stream
    while True:
        num_frames += 1
        # Initialization for each iteration
        boxes, confidences, classIDs = [], [], []

        ts = str(
            timedelta(seconds=videoStream.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        ).split(".")[0]

        # Calculating fps each second
        start_time, num_frames, last_num_frames = displayFPS(
            start_time, num_frames, last_num_frames
        )
        # read the next frame from the file
        grabbed, frame = videoStream.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False
        )
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # loop over each of the layer outputs
        for layer in layerOutputs:
            # loop over each of the detections
            for i, detection in enumerate(layer):
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = int(np.argmax(scores))
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > preDefinedConfidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array(
                        [video_width, video_height, video_width, video_height]
                    )
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append(Box(x, y, int(width), int(height)))
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        cv2.line(frame, cross_line.a, cross_line.b, (0, 0xFF, 0), 2)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, preDefinedConfidence, preDefinedThreshold
        )

        # Draw detection box
        if output or display_video:
            drawDetectionBoxes(
                idxs, boxes, classIDs, confidences, frame, labels, colors
            )

        vehicle_count, current_detections, type_counts, vehicle_types = count_vehicles(
            labels,
            idxs,
            boxes,
            classIDs,
            vehicle_count,
            previous_frame_detections,
            frame,
            type_counts,
            vehicle_types,
            bool(output or display_video),
        )

        print(f"{ts=}\t{num_frames=}\t{len(crossers)=}")

        for p, ID in current_detections.items():
            # new_loc = centerY > (video_height // 2)
            new_loc = is_right(cross_line, p)
            if ID in vehicle_locs.keys():
                old_loc = vehicle_locs[ID]
                if new_loc != old_loc and ID not in [c[2] for c in crossers]:
                    print(f"\tCROSSED: {ts=} {ID=} {p=} {old_loc=} {new_loc=}")
                    vt = vehicle_types[ID]
                    crossers.append(Cross(vt, ts, ID))
            vehicle_locs[ID] = new_loc

        # Display Vehicle Count if a vehicle has passed the line
        if output or display_video:
            displayVehicleCount(frame, vehicle_count, crossers)

        # write the output frame to disk
        if output:
            writer.write(frame)

        if display_video:
            cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Updating with the current frame detections
        previous_frame_detections.pop(0)  # Removing the first frame from the list
        previous_frame_detections.append(current_detections)

    # release the file pointers
    print("[INFO] cleaning up...")
    if output:
        writer.release()
    videoStream.release()
    print(f"Type counts: {dict(type_counts)}")
    print(f"Crossers: {crossers}")


def main(
    video: Path = Argument(..., help="Path to input video"),
    model: Path = Option("yolo-coco", help="Path to Yolo model dir"),
    output: Path = Option("", help="Path to save output if wanted"),
    confidence: float = Option(
        0.5, help="Minimum probability to filter weak detections"
    ),
    threshold: float = Option(
        0.3, help="Threshold when applying non-maxima suppression"
    ),
    display: bool = Option(False, help="Display tagged video while creating"),
    use_gpu: bool = Option(False, help="Accelerate with CUDA/GPU"),
) -> None:
    video = Path(video)
    model = Path(model)
    output_new: Path | bool = Path(output) if output else False
    labels = (model / "coco.names").open().read().strip().split("\n")
    weights = model / "yolov3.weights"
    config = model / "yolov3.cfg"
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8").tolist()

    process(
        labels,
        colors,
        weights,
        config,
        video,
        output_new,
        confidence,
        threshold,
        use_gpu,
        display,
    )


def cli() -> None:
    app = Typer(add_completion=False)
    app.command()(main)
    app()


if __name__ == "__main__":
    cli()
