# Setup detectron2 logger
import wandb

wandb.login(key="dc8c2951cc5c8a1cf40b20b238eb1c3c5477a4ae")
name_session ="mask_rcnn_R_50_FPN_best_NoiseBlur"
session =wandb.init(project="new_dataset", name=name_session)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances

import os
import cv2
import random

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer

import json
import os

from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.utils.events import get_event_storage
from detectron2.engine.hooks import HookBase

from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage

from detectron2.evaluation import verify_results


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("eval_best_nb", exist_ok=True)
            output_folder = "eval_best_nb"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
class EvalHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        self.val_loader = build_detection_test_loader(cfg, "my_dataset_val")

    def after_step(self):
        if self.trainer.iter % self.cfg.TEST.EVAL_PERIOD == 0 and self.trainer.iter > 0:
            results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
            storage = get_event_storage()
            storage.put_scalar("bbox/AP", results["bbox"]["AP"], smoothing_hint=False)

            print(f"✅ EvalHook triggered at iteration {self.trainer.iter}, bbox/AP={results['bbox']['AP']}")


# Paths
json_path = "/home/student/Documents/Project7.v4-noise_blur.coco/train/_annotations.coco.json"
image_dir = "/home/student/Documents/Project7.v4-noise_blur.coco/train"
output_json = "/home/student/Documents/Project7.v4-noise_blur.coco/train/_annotations.cleaned.json"

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
register_coco_instances("my_dataset_train", {}, 
    "/home/student/Documents/Project7.v4-noise_blur.coco/train/_annotations.cleaned.json",  # ✅ cleaned version
    "/home/student/Documents/Project7.v4-noise_blur.coco/train")

register_coco_instances("my_dataset_val", {}, 
    "/home/student/Documents/Project7.v4-noise_blur.coco/valid/_annotations.coco.json", 
    "/home/student/Documents/Project7.v4-noise_blur.coco/valid")

print("############# Dataset is ready ################")

os.makedirs("/home/student/Documents/Project7.v4-noise_blur.coco/train_samples_50_best_nb", exist_ok=True)

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
output_filename = f"/home/student/Documents/Project7.v4-noise_blur.coco/train_samples_50_nb/output_50b_nb_{os.path.basename(d['file_name'])}"
cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])

print(f"Dataset created!!!!")
custom_output_dir = "/home/student/Documents/Project7.v1-no_augmentation.coco/train/train_samples_50_best_nb"
os.makedirs(custom_output_dir, exist_ok=True)

# Configuration - cfg part needs to be changed for experimets
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
#cfg.MODEL.DEVICE = "cpu"
cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = 500  # evaluate every 500 iterations

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #0.7   # Set a custom testing threshold

cfg.OUTPUT_DIR = "/home/student/Documents/output_50_best_nb"

# Create your trainer
# Hook to save the best model based on bbox AP

# Create your trainer
trainer = MyTrainer(cfg)

# Define EvalHook
eval_hook = EvalHook(cfg)

# Define BestCheckpointer (runs AFTER eval_hook)
best_checkpointer = BestCheckpointer(
    cfg.TEST.EVAL_PERIOD,
    trainer.checkpointer,      # ✅ Correct: a DetectionCheckpointer object
    "bbox/AP",
    mode="max",
    file_prefix="model_best"
)


# Register hooks in reverse order: EvalHook should run BEFORE BestCheckpointer
trainer.register_hooks([best_checkpointer, eval_hook])

# Run training
trainer.resume_or_load(resume=False)
trainer.train()


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
# inference
#predictor = DefaultPredictor(cfg)
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
predictor = DefaultPredictor(cfg)

# Get the dataset and metadata for visualization
dataset_dicts = DatasetCatalog.get("my_dataset_val")  # Your custom dataset
metadata = MetadataCatalog.get("my_dataset_val")  # Metadata for your dataset

# Loop through all samples from the dataset
for d in dataset_dicts:
    # Read the image
    im = cv2.imread(d["file_name"])
    
    # Run the predictor on the image
    outputs = predictor(im)  # Get predictions from the model
    
    # Visualize the predictions
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW  # Remove the colors of unsegmented pixels for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the output image to the custom output directory
    output_filename = os.path.join(custom_output_dir, f"prediction_{d['file_name'].split('/')[-1]}")
    print("Saving to:", output_filename)

    cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_filename}")


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#evaluator = COCOEvaluator("my_dataset_val", output_dir="./output_50")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
#trainer.test(evaluators=[COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output_50")])

evaluator = COCOEvaluator(
    "my_dataset_val",
    tasks=["bbox", "segm"],  # <--- Explicitly define what to evaluate
    distributed=False,
    output_dir="./output_50_best_nb"
)


metrics_50_best_nb = inference_on_dataset(predictor.model, val_loader, evaluator)

flat_metrics_50_best_nb = {}
for task in ("bbox", "segm"):
    if task in metrics_50_best_nb:
        for k, v in metrics_50_best_nb[task].items():
            metrics_50_best_nb[f"{task}_{k.replace(' ', '_')}"] = v

# Log with step=0 to ensure visibility in the dashboard
if flat_metrics_50_best_nb:
    wandb.log(flat_metrics_50_best_nb, step=0)
   
#for k, v in flat_metrics_50.items():
    #print(f"{k}: {v}")
print(f"Evaluating model from: {cfg.MODEL.WEIGHTS}")    
wandb.config.update(cfg)
    
wandb.finish()
import pandas as pd

df = pd.DataFrame([flat_metrics_50_best_nb])
df.to_csv("detectron2_metrics_50_best_nb.csv", index=False)

