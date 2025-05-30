from ultralytics import YOLO
import matplotlib.pyplot as plt
import wandb

# Log in to W&B
wandb.login(key="dc8c2951cc5c8a1cf40b20b238eb1c3c5477a4ae")

##No augmentation#

datasets = ["/home/student/Documents/Project7.v1-no_augmentation.yolov11/data.yaml", "/home/student/Documents/Project7.v4-noise_blur.yolov11/data_noise_blur.yaml"]

for d_id, dataset in enumerate(datasets):
    if d_id == 1:
        dataset = "noise_blurr"
    else: 
        dataset = "standard"

    print(f"################################################")
    print(f"--------------{dataset}")    
    session_names = [f"yolo11s_std_seg_{dataset}", f"yolo11n_std_seg_{dataset}", f"yolo12s_std_det_{dataset}", f"yolo12n_std_det_{dataset}"]
    models = ["yolo11s-seg.pt", "yolo11n-seg.pt", "yolo12s.pt", "yolo12n.pt"]


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
        

    ##Augmentation#
    session_names_aug = [f"yolo11s_aug_seg_{dataset}", f"yolo11n_aug_seg_{dataset}", f"yolo12s_aug_det_{dataset}", f"yolo12n_aug_det_{dataset}"]


    for idx, session in enumerate(session_names_aug):
        name_session =session
        session =wandb.init(project="new_dataset", name=name_session)

        model = YOLO(models[idx]) 


        # Train the model
        results = model.train(data=datasets[d_id], 
            epochs=200,
            imgsz=1024,  
            project="new_dataset", 
            name=name_session, 
            batch = 8)

