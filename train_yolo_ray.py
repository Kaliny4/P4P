from ultralytics import YOLO
import matplotlib.pyplot as plt
import wandb

# Log in to W&B
wandb.login(key="dc8c2951cc5c8a1cf40b20b238eb1c3c5477a4ae")

datasets = ["/home/student/Documents/Project7.v1-no_augmentation.yolov11/data.yaml", "/home/student/Documents/Project7.v4-noise_blur.yolov11/data_noise_blur.yaml"]

for d_id, dataset in enumerate(datasets):
    if d_id == 1:
        dataset = "noise_blurr"
    else: 
        dataset = "standard"

    print(f"################################################")
    print(f"--------------{dataset}") 
    """   
    session_names = [f"yolo12s_std_seg_{dataset}", f"yolo12n_std_seg_{dataset}"]
    models = ["yolo12s-seg.yaml", "yolo12n-seg.yaml"]


    for idx, session in enumerate(session_names):
        print(f"----------------------------------")
        print(session)
        name_session =session
        session =wandb.init(project="new_dataset", name=name_session)

        model = YOLO(models[idx]) 


        # Train the model
        results = model.train(data=datasets[d_id], 
            epochs=200,
            imgsz=1024,  
            project="new_dataset", 
            name=name_session, 
            batch = 8,
            hsv_h= 0.0,
            hsv_s= 0.0,
            hsv_v= 0.0,
            degrees= 0.0,
            translate= 0.0,
            scale= 0.0,
            shear= 0.0,
            perspective= 0.0,
            flipud= 0.0,
            fliplr= 0.0,
            bgr= 0.0,
            mosaic= 0.0,
            mixup= 0.0,
            copy_paste= 0.0,
            erasing= 0.0)
        
   """
    ##Augmentation#
    session_names_aug = [f"yolo11n_seg_ray_{dataset}"]#f"yolo12s_det_ray{dataset}"]
    models_ray = ["yolo11n-seg.pt"] #"yolo12s.pt"]

 
    for idx, session in enumerate(session_names_aug):
        name_session =session
        session =wandb.init(project="new_dataset", name=name_session)

        model = YOLO(models_ray[idx]) 


        # Train the model
        results = model.train(data=datasets[d_id], 
            epochs=200,
            imgsz=1024,    
            project="new_dataset", 
            name=name_session, 
            batch = 8,

            #hyperparameters from Ray Tune
            augment=False,
            auto_augment="randaugment",
            bgr=0.17144103555296397,
            box=0.08817850033506973,
            cache=False,
            cfg=None,
            classes=None,
            close_mosaic=10,
            cls=0.5622498484679996,
            conf=None,
            copy_paste=0.10317581183787627,
            copy_paste_mode="flip",
            cos_lr=False,
            degrees=19.942638903812952,
            deterministic=True,
            device=None,
            dfl=1.5,
            dnn=False,
            dropout=0,
            dynamic=False,
            embed=None,
            erasing=0.4,
            exist_ok=False,
            fliplr=0.7304661190873054,
            flipud=0.05151453312448617,
            format="torchscript",
            fraction=1,
            freeze=None,
            half=False,
            hsv_h=0.07972545215246145,
            hsv_s=0.19441995744142604,
            hsv_v=0.34365364115021596,
            int8=False,
            iou=0.7,
            keras=False,
            kobj=1,
            line_width=None,
            lr0=0.08648041528822659,
            lrf=0.2577678778331252,
            mask_ratio=4,
            max_det=300,
            mixup=0.16922757207427708,
            momentum=0.6955836270997371,
            mosaic=0.14161120028271856,
            multi_scale=False,
            nbs=64,
            nms=False,
            opset=None,
            optimize=False,
            optimizer="auto",
            overlap_mask=True,
            patience=100,
            perspective=0.000793131060488489,
            plots=True,
            pose=12,
            pretrained=True,
            profile=False,
            rect=False,
            resume=False,
            retina_masks=False,
            save=True,
            save_conf=False,
            save_crop=False,
            save_frames=False,
            save_period=-1,
            save_txt=False,
            scale=0.6302500558154542,
            seed=0,
            shear=0.7376147317358128,
            show=False,
            simplify=True,
            single_cls=False,
            source=None,
            split="val",
            stream_buffer=False,
            task="segment",
            time=None,
            tracker="botsort.yaml",
            translate=0.31678908467738254,
            val=True,
            verbose=True,
            vid_stride=1,
            visualize=False,
            warmup_bias_lr=0.1,
            warmup_epochs=0.5991914039761309,
            warmup_momentum=0.9286956482757457,
            weight_decay=0.0007117574947074985,
            workers=0
        )

