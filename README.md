# kaggle-physionet-ecg-image-digitization
Many team use my solution at: https://www.kaggle.com/code/hengck23/demo-submission.  
This repo contains the training code for:  
 - stage0 : homography recification of input image
 - stage1 : detection of grid points 

## 1. Hardware  
- GPU: 2x Nvidia Ada A6000 (Ampere), each with VRAM 48 GB
- CPU: Intel® Xeon(R) w7-3455 CPU @ 2.5GHz, 24 cores, 48 threads
- Memory: 256 GB RAM

## 2. OS 
- ubuntu 22.04.4 LTS


## 3. Set Up Environment
- Install Python >=3.10.9
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── <solution_dir>
    ├── src  
    ├── LICENSE 
    ├── README.md 
```

- Download kaggle dataset from:  
[https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data)

- Download processed dataset (hand-annotated gridpoint) from:  
https://drive.google.com/drive/folders/1pxC707J5uuBjWKOfBPbm6VfwxapVkWkR?usp=sharing


## 4. Training the model

Since there is no ground truth annotation data for both models, there is only training and no validation in my code. Instead, we use the regression results of stage2 to identify the wrong prediction of stage0/1. By visual inspection, the error of stage0 is 0% and stage1 is 1%.

## stage0
- Go to src/stage0-train/run_train.py. Locate configuration variable cfg to set your input and output paths.
- Note that we use cfg.num_train_id = 300 for training.
- Run command:
```
python run_train.py
```
- Similarly, to collect stage0 predictions to train stage1, edit the cfg paths of src/stage0-train/run_collect_data.py, then run command:
```
python run_collect_data.py
```
## stage1
- Go to src/stage1-train/run_train.py. Locate the configuration variable cfg to set your input and output paths.
- Note that we use cfg.num_train_id = 500 for training.
- Run command:
```
python run_train.py
```

## 6. Reference trained models and validation results
- Reference training results can also be found in the share google drive below. It includes train log files, models, etc.  
https://drive.google.com/drive/folders/1pxC707J5uuBjWKOfBPbm6VfwxapVkWkR?usp=sharing

## Authors

- https://www.kaggle.com/hengck23

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

"We extend our thanks to HP for providing the Z8 Fury-G5 Data Science Workstation, which empowered our deep learning experiments. The high computational power and large GPU memory enabled us to design our models swiftly."
