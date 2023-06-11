import glob
import os
import json

import numpy as np
import cv2

print(cv2.__version__)

window_name = "BDD100K Validation Image"

val_json_filepath = "/Users/ihmmaru99/IHMJB/Programmers/Autonomous_Driving/13th/DAY2/labels/det_20/det_val.json"
img_filepath_root = "/Users/ihmmaru99/IHMJB/Programmers/Autonomous_Driving/13th/DAY2/images/100k/val/"

visualization = False

YOLO_OUTPUT_DIRECTORY_PREFIX="/Users/ihmmaru99/IHMJB/Programmers/Autonomous_Driving/13th/DAY2/yolo_labels/val/"

CLASS_ID_MAPPING_TABLE = {
    "pedestrian":0,
    "rider":1,
    "car":2,
    "truck":3,
    "bus":4,
    "train":5,
    "motorcycle":6,
    "bicycle":7,
    "traffic light":8,
    "traffic sign":9
}

def get_image_file(img_filename:str, visualization: bool):
    img_file_path = os.path.join(img_filepath_root, img_filename)
    image = cv2.imread(img_file_path, cv2.IMREAD_ANYCOLOR)
    if visualization == True:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey()
    return image

with open(val_json_filepath, 'r', encoding='UTF-8') as json_file:
    val_labels = json.load(json_file)

for val_label in val_labels:
    img_filename = val_label["name"]
    text_filename = img_filename.split(".")[0] + ".txt"
    image = get_image_file(img_filename, visualization=visualization)
    img_height, img_width, channel = np.shape(image)
    labels = val_label["labels"]
    with open(YOLO_OUTPUT_DIRECTORY_PREFIX+text_filename, "w+") as file:
        for label in labels:
            if label["category"] in CLASS_ID_MAPPING_TABLE.keys():
                class_id = CLASS_ID_MAPPING_TABLE[label['category']]
            else:
                continue
            
            box2d = label["box2d"]

            left_box2d = box2d["x1"]
            right_box2d = box2d["x2"]
            top_box2d = box2d["y1"]
            bottom_box2d = box2d["y2"]

            x = (left_box2d + right_box2d) / 2
            y = (top_box2d + bottom_box2d) / 2

            box_width = right_box2d - left_box2d
            box_height = bottom_box2d - top_box2d

            yolo_x = x / img_height
            yolo_y = y / img_width

            yolo_box_width = box_width / img_width
            yolo_box_height = box_height / img_height

        #print(class_id, yolo_x, yolo_y, yolo_box_width, yolo_box_height)
            file.write(f"{class_id} {yolo_x} {yolo_y} {yolo_box_width} {yolo_box_height}\n")

            if visualization == visualization:
                cv2.rectangle(image, (int(left_box2d), int(top_box2d)), (int(right_box2d), int(bottom_box2d)), (255,255,0), 4)
                cv2.circle(image, (int(x), int(y)), 3, (0,0,255))

        if visualization == True:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image)
            cv2.waitKey()

