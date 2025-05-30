##Convert coco predictions into format that accepts fiftyone

import json
import os
import numpy as np
from tqdm import tqdm
import fiftyone as fo
from fiftyone.core.labels import Detections, Detection, Segmentation
import fiftyone.utils.coco as fouc
from PIL import Image, ImageDraw


def delete_datasets():
    datasets = fo.list_datasets()
    print(datasets)
    for dataset_name in datasets:
        fo.delete_dataset(dataset_name)
        print(f"Deleted dataset: {dataset_name}")
    view = None


delete_datasets()

images_dir = "/home/student/Documents/Project7.v1-no_augmentation.coco/test"
gt_annotations = "/home/student/Documents/Project7.v1-no_augmentation.coco/test/_annotations.coco.json"
yolo12s_det_raynoise_blurr = "/home/student/Documents/new_dataset/yolo12s_det_raynoise_blurr/predictions/instances_predictions.json"
yolo12s_aug_det_noise_blurr = "/home/student/Documents/new_dataset/yolo12s_aug_det_noise_blurr/predictions/instances_predictions.json"
yolo11n_seg_ray_noise_blurr = "/home/student/Documents/new_dataset/yolo11n_seg_ray_noise_blurr/predictions/instances_predictions.json"
yolo11s_seg_raynoise_blurr = "/home/student/Documents/new_dataset/yolo11s_seg_raynoise_blurr/predictions/instances_predictions.json"
mask50_r_cnn = "/home/student/Documents/output_ray_50b_nb/instances_predictions.json"
#########################################################################################################################
# === Load Ground Truth ===

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=gt_annotations,
    label_field="gt",
    include_id=True,
)
# Inspect the first sample to find the correct attribute name
sample = dataset.first()

# Load your COCO predictions JSON
with open(yolo12s_det_raynoise_blurr) as f:
    coco_data = json.load(f)

# Get class list from COCO categories
classes = {category["id"]: category["name"] for category in coco_data["categories"]}
# Verify the sample IDs
sample_ids = [sample.id for sample in dataset]

# Ensure the annotations match the sample IDs
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    # if image_id not in sample_ids:
    #    print(f"Image ID {image_id} not found in dataset")

# Initialize dictionaries to store detections and segmentations
detections_dict = {}
segmentations_dict = {}

# Add predictions to dictionaries using filepaths
with fo.ProgressBar() as pb:
    for annotation in pb(coco_data["annotations"]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation.get("segmentation", None)

        # Find the corresponding sample using its filepath
        image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
        sample = dataset.match(
            {"filepath": os.path.join(images_dir, image_info["file_name"])}
        ).first()

        if sample is None:
            print(f"Sample for image ID {image_id} not found")
            continue
        # Convert bbox to FiftyOne format
        x, y, width, height = bbox
        rel_box = [
            x / sample.metadata.width,
            y / sample.metadata.height,
            width / sample.metadata.width,
            height / sample.metadata.height,
        ]

        # Create detection
        detection = fo.Detection(
            label=classes[category_id],
            bounding_box=rel_box,
            confidence=annotation.get(
                "score", 1.0
            ),  # Default to 1.0 if score is not provided
        )
        # Store detection in dictionary
        if sample.id not in detections_dict:
            detections_dict[sample.id] = [detection]
        else:
            detections_dict[sample.id].append(detection)

        if segmentation:
            mask = Image.new("L", (int(width), int(height)), 0)
            draw = ImageDraw.Draw(mask)
            for seg in segmentation:

                seg = np.array(seg, dtype=np.float64).reshape(-1, 2)
                # seg = np.array(seg).reshape(-1, 2)
                seg[:, 0] -= x
                seg[:, 1] -= y
                seg[:, 0] = np.clip(seg[:, 0], 0, width - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, height - 1)
                draw.polygon([tuple(point) for point in seg], outline=1, fill=1)
            mask = np.array(mask, dtype=bool)

            segmentation = fo.Detection(
                label=classes[category_id],
                bounding_box=rel_box,
                iscrowd=0,
                mask=mask,
                confidence=annotation.get(
                    "score", 1.0
                ),  # Default to 1.0 if score is not provided
            )

            # Store segmentation in dictionary
            if sample.id not in segmentations_dict:
                segmentations_dict[sample.id] = [segmentation]
            else:
                segmentations_dict[sample.id].append(segmentation)

# Assign detections and segmentations to samples
for sample in dataset:
    image_id = sample.id

    if image_id in detections_dict:
        sample["predictions_bbox"] = fo.Detections(detections=detections_dict[image_id])
        print(f"{image_id} - {len(detections_dict[image_id])}")

    if image_id in segmentations_dict:
        sample["predictions_bbox"] = fo.Detections(
            detections=segmentations_dict[image_id]
        )  # "predictions_mask"
        print(f"Seg: {image_id} - {len(detections_dict[image_id])}")
        # for detection in sample["predictions_mask"].detections:
        # print(f"PRED Segmentation mask size: {detection.mask.shape}")
        # print(detection)
    sample.save()
sample = dataset.first()

# gt_segmentations = sample["gt_segmentations"]
# for detection in gt_segmentations.detections:
#    print(f"GT Segmentation mask size: {detection.mask.shape}")
sample_with_segmentation = next(
    (s for s in dataset if s.id in segmentations_dict), None
)

if sample_with_segmentation:
    # Get the first segmentation mask and bounding box from the selected sample
    first_segmentation = segmentations_dict[sample_with_segmentation.id][0]
    first_bbox_detection = detections_dict[sample_with_segmentation.id][0]

    # Create binary image showing the segmentation mask with the size of the bounding box
    binary_mask_image_path = "binary_mask_image.png"
    binary_mask_image = Image.fromarray(first_segmentation.mask.astype(np.uint8) * 255)
    binary_mask_image.save(binary_mask_image_path)

print("Binary mask image saved successfully.")
print("Fields in dataset:", dataset.get_field_schema().keys())
dataset.rename_sample_field("predictions_bbox", "yolo12s_det_raynoise_blurr")
results_y12s_ray = dataset.evaluate_detections(
    "yolo12s_det_raynoise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks1",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_yolo12s_det_raynoise_blurr:", results_y12s_ray.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y12s_ray.print_report()

# Inspect the first sample to find the correct attribute name
sample = dataset.first()

#######################################################################################################################
# Load your COCO predictions JSON
with open(yolo12s_aug_det_noise_blurr) as f:
    coco_data = json.load(f)

# Get class list from COCO categories
classes = {category["id"]: category["name"] for category in coco_data["categories"]}
# Verify the sample IDs
sample_ids = [sample.id for sample in dataset]

# Ensure the annotations match the sample IDs
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    # if image_id not in sample_ids:
    #    print(f"Image ID {image_id} not found in dataset")

# Initialize dictionaries to store detections and segmentations
detections_dict = {}
segmentations_dict = {}

# Add predictions to dictionaries using filepaths
with fo.ProgressBar() as pb:
    for annotation in pb(coco_data["annotations"]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation.get("segmentation", None)

        # Find the corresponding sample using its filepath
        image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
        sample = dataset.match(
            {"filepath": os.path.join(images_dir, image_info["file_name"])}
        ).first()

        if sample is None:
            print(f"Sample for image ID {image_id} not found")
            continue
        # Convert bbox to FiftyOne format
        x, y, width, height = bbox
        rel_box = [
            x / sample.metadata.width,
            y / sample.metadata.height,
            width / sample.metadata.width,
            height / sample.metadata.height,
        ]

        # Create detection
        detection = fo.Detection(
            label=classes[category_id],
            bounding_box=rel_box,
            confidence=annotation.get(
                "score", 1.0
            ),  # Default to 1.0 if score is not provided
        )
        # Store detection in dictionary
        if sample.id not in detections_dict:
            detections_dict[sample.id] = [detection]
        else:
            detections_dict[sample.id].append(detection)

        if segmentation:
            mask = Image.new("L", (int(width), int(height)), 0)
            draw = ImageDraw.Draw(mask)
            for seg in segmentation:

                seg = np.array(seg, dtype=np.float64).reshape(-1, 2)
                # seg = np.array(seg).reshape(-1, 2)
                seg[:, 0] -= x
                seg[:, 1] -= y
                seg[:, 0] = np.clip(seg[:, 0], 0, width - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, height - 1)
                draw.polygon([tuple(point) for point in seg], outline=1, fill=1)
            mask = np.array(mask, dtype=bool)

            segmentation = fo.Detection(
                label=classes[category_id],
                bounding_box=rel_box,
                iscrowd=0,
                mask=mask,
                confidence=annotation.get(
                    "score", 1.0
                ),  # Default to 1.0 if score is not provided
            )

            # Store segmentation in dictionary
            if sample.id not in segmentations_dict:
                segmentations_dict[sample.id] = [segmentation]
            else:
                segmentations_dict[sample.id].append(segmentation)

# Assign detections and segmentations to samples
for sample in dataset:
    image_id = sample.id

    if image_id in detections_dict:
        sample["predictions_bbox"] = fo.Detections(detections=detections_dict[image_id])
        print(f"{image_id} - {len(detections_dict[image_id])}")

    if image_id in segmentations_dict:
        sample["predictions_bbox"] = fo.Detections(
            detections=segmentations_dict[image_id]
        )  # "predictions_mask"
        print(f"Seg: {image_id} - {len(detections_dict[image_id])}")
        # for detection in sample["predictions_mask"].detections:
        # print(f"PRED Segmentation mask size: {detection.mask.shape}")
        # print(detection)
    sample.save()
sample = dataset.first()

# gt_segmentations = sample["gt_segmentations"]
# for detection in gt_segmentations.detections:
#    print(f"GT Segmentation mask size: {detection.mask.shape}")
sample_with_segmentation = next(
    (s for s in dataset if s.id in segmentations_dict), None
)

if sample_with_segmentation:
    # Get the first segmentation mask and bounding box from the selected sample
    first_segmentation = segmentations_dict[sample_with_segmentation.id][0]
    first_bbox_detection = detections_dict[sample_with_segmentation.id][0]

    # Create binary image showing the segmentation mask with the size of the bounding box
    binary_mask_image_path = "binary_mask_image.png"
    binary_mask_image = Image.fromarray(first_segmentation.mask.astype(np.uint8) * 255)
    binary_mask_image.save(binary_mask_image_path)

print("Binary mask image saved successfully.")
print("Fields in dataset:", dataset.get_field_schema().keys())
dataset.rename_sample_field("predictions_bbox", "yolo12s_aug_det_noise_blurr")
results_y12s_aug = dataset.evaluate_detections(
    "yolo12s_aug_det_noise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks2",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_yolo12s_aug_det_noise_blurr:", results_y12s_aug.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y12s_aug.print_report()

#################################################################################################################################
# Inspect the first sample to find the correct attribute name
sample = dataset.first()

# Load your COCO predictions JSON
with open(yolo11n_seg_ray_noise_blurr) as f:
    coco_data = json.load(f)

# Get class list from COCO categories
classes = {category["id"]: category["name"] for category in coco_data["categories"]}
# Verify the sample IDs
sample_ids = [sample.id for sample in dataset]

# Ensure the annotations match the sample IDs
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    # if image_id not in sample_ids:
    #    print(f"Image ID {image_id} not found in dataset")

# Initialize dictionaries to store detections and segmentations
detections_dict = {}
segmentations_dict = {}

# Add predictions to dictionaries using filepaths
with fo.ProgressBar() as pb:
    for annotation in pb(coco_data["annotations"]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation.get("segmentation", None)

        # Find the corresponding sample using its filepath
        image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
        sample = dataset.match(
            {"filepath": os.path.join(images_dir, image_info["file_name"])}
        ).first()

        if sample is None:
            print(f"Sample for image ID {image_id} not found")
            continue
        # Convert bbox to FiftyOne format
        x, y, width, height = bbox
        rel_box = [
            x / sample.metadata.width,
            y / sample.metadata.height,
            width / sample.metadata.width,
            height / sample.metadata.height,
        ]

        # Create detection
        detection = fo.Detection(
            label=classes[category_id],
            bounding_box=rel_box,
            confidence=annotation.get(
                "score", 1.0
            ),  # Default to 1.0 if score is not provided
        )
        # Store detection in dictionary
        if sample.id not in detections_dict:
            detections_dict[sample.id] = [detection]
        else:
            detections_dict[sample.id].append(detection)

        if segmentation:
            mask = Image.new("L", (int(width), int(height)), 0)
            draw = ImageDraw.Draw(mask)
            for seg in segmentation:

                seg = np.array(seg, dtype=np.float64).reshape(-1, 2)
                # seg = np.array(seg).reshape(-1, 2)
                seg[:, 0] -= x
                seg[:, 1] -= y
                seg[:, 0] = np.clip(seg[:, 0], 0, width - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, height - 1)
                draw.polygon([tuple(point) for point in seg], outline=1, fill=1)
            mask = np.array(mask, dtype=bool)

            segmentation = fo.Detection(
                label=classes[category_id],
                bounding_box=rel_box,
                iscrowd=0,
                mask=mask,
                confidence=annotation.get(
                    "score", 1.0
                ),  # Default to 1.0 if score is not provided
            )

            # Store segmentation in dictionary
            if sample.id not in segmentations_dict:
                segmentations_dict[sample.id] = [segmentation]
            else:
                segmentations_dict[sample.id].append(segmentation)

# Assign detections and segmentations to samples
for sample in dataset:
    image_id = sample.id

    if image_id in detections_dict:
        sample["predictions_bbox"] = fo.Detections(detections=detections_dict[image_id])
        print(f"{image_id} - {len(detections_dict[image_id])}")

    if image_id in segmentations_dict:
        sample["predictions_mask"] = fo.Detections(
            detections=segmentations_dict[image_id]
        )  # "predictions_mask"
        print(f"Seg: {image_id} - {len(detections_dict[image_id])}")
        # for detection in sample["predictions_mask"].detections:
        # print(f"PRED Segmentation mask size: {detection.mask.shape}")
        # print(detection)
    sample.save()
sample = dataset.first()

# gt_segmentations = sample["gt_segmentations"]
# for detection in gt_segmentations.detections:
#    print(f"GT Segmentation mask size: {detection.mask.shape}")
sample_with_segmentation = next(
    (s for s in dataset if s.id in segmentations_dict), None
)

if sample_with_segmentation:
    # Get the first segmentation mask and bounding box from the selected sample
    first_segmentation = segmentations_dict[sample_with_segmentation.id][0]
    first_bbox_detection = detections_dict[sample_with_segmentation.id][0]

    # Create binary image showing the segmentation mask with the size of the bounding box
    binary_mask_image_path = "binary_mask_image.png"
    binary_mask_image = Image.fromarray(first_segmentation.mask.astype(np.uint8) * 255)
    binary_mask_image.save(binary_mask_image_path)

print("Binary mask image saved successfully.")
print("Fields in dataset:", dataset.get_field_schema().keys())
dataset.rename_sample_field("predictions_mask", "mask_yolo11n_seg_ray_noise_blurr")
dataset.rename_sample_field("predictions_bbox", "bbox_yolo11n_seg_ray_noise_blurr")
results_y11n_ray_mask = dataset.evaluate_detections(
    "mask_yolo11n_seg_ray_noise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks3",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_mask_yolo11n_seg_ray_noise_blurr:", results_y11n_ray_mask.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y11n_ray_mask.print_report()

results_y11n_ray_bbox = dataset.evaluate_detections(
    "bbox_yolo11n_seg_ray_noise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks4",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_bbox_yolo11n_seg_ray_noise_blurr:", results_y11n_ray_bbox.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y11n_ray_bbox.print_report()


###########################################################################################################################################
# Inspect the first sample to find the correct attribute name
sample = dataset.first()

# Load your COCO predictions JSON
with open(yolo11s_seg_raynoise_blurr) as f:
    coco_data = json.load(f)

# Get class list from COCO categories
classes = {category["id"]: category["name"] for category in coco_data["categories"]}
# Verify the sample IDs
sample_ids = [sample.id for sample in dataset]

# Ensure the annotations match the sample IDs
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    # if image_id not in sample_ids:
    #    print(f"Image ID {image_id} not found in dataset")

# Initialize dictionaries to store detections and segmentations
detections_dict = {}
segmentations_dict = {}

# Add predictions to dictionaries using filepaths
with fo.ProgressBar() as pb:
    for annotation in pb(coco_data["annotations"]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation.get("segmentation", None)

        # Find the corresponding sample using its filepath
        image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
        sample = dataset.match(
            {"filepath": os.path.join(images_dir, image_info["file_name"])}
        ).first()

        if sample is None:
            print(f"Sample for image ID {image_id} not found")
            continue
        # Convert bbox to FiftyOne format
        x, y, width, height = bbox
        rel_box = [
            x / sample.metadata.width,
            y / sample.metadata.height,
            width / sample.metadata.width,
            height / sample.metadata.height,
        ]

        # Create detection
        detection = fo.Detection(
            label=classes[category_id],
            bounding_box=rel_box,
            confidence=annotation.get(
                "score", 1.0
            ),  # Default to 1.0 if score is not provided
        )
        # Store detection in dictionary
        if sample.id not in detections_dict:
            detections_dict[sample.id] = [detection]
        else:
            detections_dict[sample.id].append(detection)

        if segmentation:
            mask = Image.new("L", (int(width), int(height)), 0)
            draw = ImageDraw.Draw(mask)
            for seg in segmentation:

                seg = np.array(seg, dtype=np.float64).reshape(-1, 2)
                # seg = np.array(seg).reshape(-1, 2)
                seg[:, 0] -= x
                seg[:, 1] -= y
                seg[:, 0] = np.clip(seg[:, 0], 0, width - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, height - 1)
                draw.polygon([tuple(point) for point in seg], outline=1, fill=1)
            mask = np.array(mask, dtype=bool)

            segmentation = fo.Detection(
                label=classes[category_id],
                bounding_box=rel_box,
                iscrowd=0,
                mask=mask,
                confidence=annotation.get(
                    "score", 1.0
                ),  # Default to 1.0 if score is not provided
            )

            # Store segmentation in dictionary
            if sample.id not in segmentations_dict:
                segmentations_dict[sample.id] = [segmentation]
            else:
                segmentations_dict[sample.id].append(segmentation)

# Assign detections and segmentations to samples
for sample in dataset:
    image_id = sample.id

    if image_id in detections_dict:
        sample["predictions_bbox"] = fo.Detections(detections=detections_dict[image_id])
        print(f"{image_id} - {len(detections_dict[image_id])}")

    if image_id in segmentations_dict:
        sample["predictions_mask"] = fo.Detections(
            detections=segmentations_dict[image_id]
        )  # "predictions_mask"
        print(f"Seg: {image_id} - {len(detections_dict[image_id])}")
        # for detection in sample["predictions_mask"].detections:
        # print(f"PRED Segmentation mask size: {detection.mask.shape}")
        # print(detection)
    sample.save()
sample = dataset.first()

# gt_segmentations = sample["gt_segmentations"]
# for detection in gt_segmentations.detections:
#    print(f"GT Segmentation mask size: {detection.mask.shape}")
sample_with_segmentation = next(
    (s for s in dataset if s.id in segmentations_dict), None
)

if sample_with_segmentation:
    # Get the first segmentation mask and bounding box from the selected sample
    first_segmentation = segmentations_dict[sample_with_segmentation.id][0]
    first_bbox_detection = detections_dict[sample_with_segmentation.id][0]

    # Create binary image showing the segmentation mask with the size of the bounding box
    binary_mask_image_path = "binary_mask_image.png"
    binary_mask_image = Image.fromarray(first_segmentation.mask.astype(np.uint8) * 255)
    binary_mask_image.save(binary_mask_image_path)

print("Binary mask image saved successfully.")
print("Fields in dataset:", dataset.get_field_schema().keys())
dataset.rename_sample_field("predictions_mask", "mask_yolo11s_seg_raynoise_blurr")
dataset.rename_sample_field("predictions_bbox", "bbox_yolo11s_seg_raynoise_blurr")
results_y11s_ray_mask = dataset.evaluate_detections(
    "mask_yolo11s_seg_raynoise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks5",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_mask_yolo11s_seg_raynoise_blurr:", results_y11s_ray_mask.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y11s_ray_mask.print_report()

results_y11s_ray_bbox = dataset.evaluate_detections(
    "bbox_yolo11s_seg_raynoise_blurr",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks6",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_bbox_yolo11s_seg_raynoise_blurr:", results_y11s_ray_bbox.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_y11s_ray_bbox.print_report()

full_report = results_y11s_ray_bbox.report()
with open("results_y11s_ray_bbox_full.csv", "w") as f:
    f.write(str(full_report))

############################################################################################################################
# Inspect the first sample to find the correct attribute name
sample = dataset.first()

# Load your COCO predictions JSON
with open(mask50_r_cnn) as f:
    coco_data = json.load(f)

# Get class list from COCO categories
classes = {category["id"]: category["name"] for category in coco_data["categories"]}
# Verify the sample IDs
sample_ids = [sample.id for sample in dataset]

# Ensure the annotations match the sample IDs
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    # if image_id not in sample_ids:
    #    print(f"Image ID {image_id} not found in dataset")

# Initialize dictionaries to store detections and segmentations
detections_dict = {}
segmentations_dict = {}

# Add predictions to dictionaries using filepaths
with fo.ProgressBar() as pb:
    for annotation in pb(coco_data["annotations"]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation.get("segmentation", None)

        # Find the corresponding sample using its filepath
        image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
        sample = dataset.match(
            {"filepath": os.path.join(images_dir, image_info["file_name"])}
        ).first()

        if sample is None:
            print(f"Sample for image ID {image_id} not found")
            continue
        # Convert bbox to FiftyOne format
        x, y, width, height = bbox
        rel_box = [
            x / sample.metadata.width,
            y / sample.metadata.height,
            width / sample.metadata.width,
            height / sample.metadata.height,
        ]

        # Create detection
        detection = fo.Detection(
            label=classes[category_id],
            bounding_box=rel_box,
            confidence=annotation.get(
                "score", 1.0
            ),  # Default to 1.0 if score is not provided
        )
        # Store detection in dictionary
        if sample.id not in detections_dict:
            detections_dict[sample.id] = [detection]
        else:
            detections_dict[sample.id].append(detection)

        if segmentation:
            mask = Image.new("L", (int(width), int(height)), 0)
            draw = ImageDraw.Draw(mask)
            for seg in segmentation:

                seg = np.array(seg, dtype=np.float64).reshape(-1, 2)
                # seg = np.array(seg).reshape(-1, 2)
                seg[:, 0] -= x
                seg[:, 1] -= y
                seg[:, 0] = np.clip(seg[:, 0], 0, width - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, height - 1)
                draw.polygon([tuple(point) for point in seg], outline=1, fill=1)
            mask = np.array(mask, dtype=bool)

            segmentation = fo.Detection(
                label=classes[category_id],
                bounding_box=rel_box,
                iscrowd=0,
                mask=mask,
                confidence=annotation.get(
                    "score", 1.0
                ),  # Default to 1.0 if score is not provided
            )

            # Store segmentation in dictionary
            if sample.id not in segmentations_dict:
                segmentations_dict[sample.id] = [segmentation]
            else:
                segmentations_dict[sample.id].append(segmentation)

# Assign detections and segmentations to samples
for sample in dataset:
    image_id = sample.id

    if image_id in detections_dict:
        sample["predictions_bbox"] = fo.Detections(detections=detections_dict[image_id])
        print(f"{image_id} - {len(detections_dict[image_id])}")

    if image_id in segmentations_dict:
        sample["predictions_mask"] = fo.Detections(
            detections=segmentations_dict[image_id]
        )  # "predictions_mask"
        print(f"Seg: {image_id} - {len(detections_dict[image_id])}")
        # for detection in sample["predictions_mask"].detections:
        # print(f"PRED Segmentation mask size: {detection.mask.shape}")
        # print(detection)
    sample.save()
sample = dataset.first()

# gt_segmentations = sample["gt_segmentations"]
# for detection in gt_segmentations.detections:
#    print(f"GT Segmentation mask size: {detection.mask.shape}")
sample_with_segmentation = next(
    (s for s in dataset if s.id in segmentations_dict), None
)

if sample_with_segmentation:
    # Get the first segmentation mask and bounding box from the selected sample
    first_segmentation = segmentations_dict[sample_with_segmentation.id][0]
    first_bbox_detection = detections_dict[sample_with_segmentation.id][0]

    # Create binary image showing the segmentation mask with the size of the bounding box
    binary_mask_image_path = "binary_mask_image.png"
    binary_mask_image = Image.fromarray(first_segmentation.mask.astype(np.uint8) * 255)
    binary_mask_image.save(binary_mask_image_path)

print("Binary mask image saved successfully.")
print("Fields in dataset:", dataset.get_field_schema().keys())
dataset.rename_sample_field("predictions_mask", "mask_mask50_r_cnn")
dataset.rename_sample_field("predictions_bbox", "bbox_mask50_r_cnn")
results_mask50_r_cnn_mask = dataset.evaluate_detections(
    "mask_mask50_r_cnn",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks7",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_mask50_r_cnn:", results_mask50_r_cnn_mask.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_mask50_r_cnn_mask.print_report()

results_mask50_r_cnn_bbox = dataset.evaluate_detections(
    "bbox_mask50_r_cnn",
    # "predictions_mask",
    # gt_field="gt_segmentations",
    gt_field="gt_detections",
    eval_key="eval_masks8",
    method="coco",
    compute_mAP=True,
    iou=0.5,
)

print("mAP50_bbox_mask50_r_cnn:", results_mask50_r_cnn_bbox.mAP())
# print("Precision range:", dataset.bounds("eval_masks_precision"))
# print("Recall range:", dataset.bounds("eval_masks_recall"))
results_mask50_r_cnn_bbox.print_report()

full_report = results_mask50_r_cnn_bbox.report()
with open("results_mask50_r_cnn_bbox_full.csv", "w") as f:
    f.write(str(full_report))


############################################################################################################################
session = fo.launch_app(dataset)
session.wait()
print("FiftyOne App URL:", session.url)
