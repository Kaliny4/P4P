import os
import cv2
import json
import numpy as np
import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

# --- Paths ---
model_path = "/home/student/Documents/output_ray_50b_nb/model_best.pth"
input_folder = "/home/student/Documents/Project7.v1-no_augmentation.coco/test"
output_image_folder = "/home/student/Documents/output_ray_50b_nb/predicted_images"
output_json_path = (
    "/home/student/Documents/output_ray_50b_nb/instances_predictions.json"
)
os.makedirs(output_image_folder, exist_ok=True)

# --- Config ---
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
    2  # Update if your model has a different number of classes
)

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
    [16, 32, 64, 128, 256]
]  # smaller anchors for plants
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.3, 0.5, 1.0]]  # tall and square plants
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]  # standard detection, no rotation
cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5  # typically recommended setting


predictor = DefaultPredictor(cfg)

# --- Metadata ---
category_names = ["OR", "WR"]  # <-- make sure this matches your training
metadata = MetadataCatalog.get("my_dataset_test")
metadata.set(thing_classes=category_names)

# --- COCO Output Template ---
coco_output = {
    "info": {
        "description": "Mask R-CNN Predictions",
        "date_created": datetime.datetime.now().isoformat(),
    },
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": name} for i, name in enumerate(category_names)],
}
annotation_id = 1

# --- Inference Loop ---
image_files = [
    f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for image_id, filename in enumerate(image_files, start=1):
    file_path = os.path.join(input_folder, filename)
    image = cv2.imread(file_path)
    if image is None:
        print(f"⚠️ Skipping unreadable image: {filename}")
        continue

    height, width = image.shape[:2]
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    print(f"[{image_id}] {filename} → {len(instances)} instances")

    # --- Add image info to COCO
    coco_output["images"].append(
        {"id": image_id, "file_name": filename, "width": width, "height": height}
    )

    # --- Draw and save visualization
    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,
    )
    out = v.draw_instance_predictions(instances)
    base_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_image_folder, f"{base_name}_pred.jpg")
    cv2.imwrite(out_path, out.get_image()[:, :, ::-1])

    # --- Convert to COCO-format annotations
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = [c.flatten().tolist() for c in contours if len(c.flatten()) >= 6]
        if not segmentation:
            continue

        x1, y1, x2, y2 = instances.pred_boxes.tensor[i].numpy().tolist()
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        area = float(np.sum(mask))

        coco_output["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(instances.pred_classes[i]),
                "segmentation": segmentation,
                "bbox": [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                "area": area,
                "iscrowd": 0,
                "score": float(instances.scores[i]),
            }
        )
        annotation_id += 1

# --- Save final JSON
with open(output_json_path, "w") as f:
    json.dump(coco_output, f)

print(f"\n✅ Done! Annotations saved to: {output_json_path}")
