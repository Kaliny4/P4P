import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
import json
import datetime
from ultralytics import YOLO

# --- Paths ---
model_path = "/home/student/Documents/new_dataset/yolo11s_seg_raynoise_blurr/weights/best.pt"  # "/home/student/Documents/new_dataset/yolo11n_seg_ray_noise_blurr/weights/best.pt"
input_path = "/home/student/Documents/Mixed_17"  # "/home/student/Documents/Project7.v1-no_augmentation.coco/test"
# output_folder = os.path.join("/home/student/Documents/new_dataset/yolo11n_seg_ray_noise_blurr", "predictions")
output_folder = os.path.join("/home/student/Documents/Mixed_17", "predictions")
os.makedirs(output_folder, exist_ok=True)


# --- Load model ---
model = YOLO(model_path)

# --- Get list of images ---
image_paths = [
    os.path.join(input_path, f)
    for f in os.listdir(input_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# --- COCO Format Init ---
coco_output = {
    "info": {
        "description": "YOLOv11 Predictions",
        "date_created": datetime.datetime.now().isoformat(),
    },
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": name} for i, name in model.names.items()],
}
annotation_id = 1

# --- Process Each Image ---
for image_id, img_path in enumerate(image_paths, start=1):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # --- Run predictions once ---
results_iter = model.predict(source=image_paths, stream=True)

# --- Process results ---
for image_id, (img_path, results) in enumerate(zip(image_paths, results_iter), start=1):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    annotated = img_rgb.copy()

    # HOTFIX: squeeze protos if needed
    if hasattr(results, "masks") and hasattr(results.masks, "protos"):
        if results.masks.protos.ndim == 4:
            results.masks.protos = results.masks.protos.squeeze(0)

    # Add image info
    coco_output["images"].append(
        {
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h,
        }
    )

    # Boxes
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.cpu().item())
            conf = float(box.conf.cpu().item())
            xyxy = box.xyxy.cpu().numpy().squeeze()
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            bbox = [x1, y1, width, height]
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(
                annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # Masks
    if results.masks is not None and results.masks.data is not None:
        for m_id, mask in enumerate(results.masks.data.cpu().numpy()):
            cls_id = int(results.boxes.cls[m_id].item())
            conf = float(results.boxes.conf[m_id].item())
            binary_mask = (mask > 0.5).astype(np.uint8)
            color_mask = np.zeros_like(annotated)
            color_mask[binary_mask == 1] = [0, 255, 0]
            annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.5, 0)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = [c.flatten().tolist() for c in contours if len(c.flatten()) >= 6]
            if not polygons:
                continue
            y_indices, x_indices = np.where(binary_mask == 1)
            x_min, y_min = int(x_indices.min()), int(y_indices.min())
            x_max, y_max = int(x_indices.max()), int(y_indices.max())
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = int(np.sum(binary_mask))
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": polygons,
                    "bbox": [float(x) for x in bbox],
                    "area": float(area),
                    "iscrowd": 0,
                    "score": float(conf),
                }
            )
            annotation_id += 1

    # Save annotated image
    out_img = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(out_img, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_img}")

# Save final annotations
with open(os.path.join(output_folder, "instances_predictions.json"), "w") as f:
    json.dump(coco_output, f)
print("✅ Saved COCO annotations to instances_predictions.json")
