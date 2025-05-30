# Setup detectron2 logger
import wandb

wandb.login(key="******************************")
name_session = "mask_rcnn_R_101_FPN"
session = wandb.init(project="new_dataset", name=name_session)

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("eval", exist_ok=True)
            output_folder = "eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Paths
json_path = "/home/student/Documents/Project7.v1-no_augmentation.coco/train/_annotations.coco.json"
image_dir = "/home/student/Documents/Project7.v1-no_augmentation.coco/train"
output_json = "/home/student/Documents/Project7.v1-no_augmentation.coco/train/_annotations.cleaned.json"

# Load original JSON
with open(json_path) as f:
    coco = json.load(f)

# Keep only images that exist
valid_images = []
valid_ids = set()

for img in coco["images"]:
    img_path = os.path.join(image_dir, img["file_name"])
    if os.path.isfile(img_path):
        valid_images.append(img)
        valid_ids.add(img["id"])
    else:
        print("❌ Missing:", img["file_name"])

# Filter annotations
valid_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in valid_ids]

# Save cleaned JSON
coco["images"] = valid_images
coco["annotations"] = valid_annotations

with open(output_json, "w") as f:
    json.dump(coco, f)

print(f"✅ Cleaned annotation file saved to:\n{output_json}")

# Register the custom datasets
register_coco_instances(
    "my_dataset_train",
    {},
    "/home/student/Documents/Project7.v1-no_augmentation.coco/train/_annotations.cleaned.json",  # ✅ cleaned version
    "/home/student/Documents/Project7.v1-no_augmentation.coco/train",
)

register_coco_instances(
    "my_dataset_val",
    {},
    "/home/student/Documents/Project7.v1-no_augmentation.coco/valid/_annotations.coco.json",
    "/home/student/Documents/Project7.v1-no_augmentation.coco/valid",
)


os.makedirs(
    "/home/student/Documents/Project7.v1-no_augmentation.coco/train_samples_101",
    exist_ok=True,
)

# Load the dataset and metadata for visualization
dataset_dicts = DatasetCatalog.get("my_dataset_train")  # Correct dataset name
metadata = MetadataCatalog.get("my_dataset_train")  # Get metadata for the dataset

# Visualize 15 random samples from the dataset
for d in random.sample(dataset_dicts, 15):
    img = cv2.imread(d["file_name"])
    if img is None:
        print(f"⚠️ Could not read image: {d['file_name']}")
        continue

    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    output_filename = f"/home/student/Documents/Project7.v1-no_augmentation.coco/train_samples_101/output_101_{os.path.basename(d['file_name'])}"
    cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])

print(f"Dataset created!!!!")
custom_output_dir = (
    "/home/student/Documents/Project7.v1-no_augmentation.coco/train/train_samples_101"
)
os.makedirs(custom_output_dir, exist_ok=True)

# Configuration - cfg part needs to be changed for experimets
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
# cfg.MODEL.DEVICE = "cpu"
cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = (
    2  # This is the real "batch size" commonly known to deep learning people
)
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = 500  # evaluate every 500 iterations

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 0.7   # Set a custom testing threshold
cfg.OUTPUT_DIR = "/home/student/Documents/output_101"

trainer = MyTrainer(cfg)
# trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# best_model_ap = 0
# 1f > best_model_ap:
#    best_model_ap =
#    cfg.MODEL.

cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # Path to the model we just trained

# inference
predictor = DefaultPredictor(cfg)

# Get the dataset and metadata for visualization
dataset_dicts = DatasetCatalog.get("my_dataset_val")  # Your custom dataset
metadata = MetadataCatalog.get("my_dataset_val")  # Metadata for your dataset

# Loop through 3 random samples from the dataset
for d in dataset_dicts:
    # Read the image
    im = cv2.imread(d["file_name"])

    # Run the predictor on the image
    outputs = predictor(im)  # Get predictions from the model
    print(outputs)

    # Visualize the predictions
    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,  # Remove the colors of unsegmented pixels for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the output image to the custom output directory
    output_filename = os.path.join(
        custom_output_dir, f"prediction_{d['file_name'].split('/')[-1]}"
    )
    print("Saving to:", output_filename)

    cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_filename}")

# evaluator = COCOEvaluator("my_dataset_val", output_dir="./output_101")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")

evaluator = COCOEvaluator(
    "my_dataset_val",
    tasks=["bbox", "segm"],  # <--- Explicitly define what to evaluate
    distributed=False,
    output_dir="./output_101",
)
metrics_101 = inference_on_dataset(predictor.model, val_loader, evaluator)

flat_metrics_101 = {}
for task in ("bbox", "segm"):
    if task in metrics_101:
        for k, v in metrics_101[task].items():
            flat_metrics_101[f"{task}_{k.replace(' ', '_')}"] = v
# 1f > best_model_ap:
#    best_model_ap =
#    cfg.MODEL.
# Log with step=0 to ensure visibility in the dashboard
if flat_metrics_101:
    wandb.log(flat_metrics_101, step=0)

for k, v in flat_metrics_101.items():
    print(f"{k}: {v}")
wandb.config.update(cfg)
wandb.finish()

df = pd.DataFrame([flat_metrics_101])
df.to_csv("detectron2_metrics_101.csv", index=False)
