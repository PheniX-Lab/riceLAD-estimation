# riceLAD-estimation
An end-to-end rice canopy leaf angle distribution (LAD) estimation model based on DINOv2 model was developed using 21,6000 sim2real images.  
We can high-throughput extract field rice LAD from UAV captured RGB images using this model.  
This is a well-trained model that can be directly used, just configure the required virtual environment.  

# Model  Architecture
The model consists of two main components: a SOTA feature extraction module and a lightweight regression module.   
1) The feature extraction module uses DINOv2, an open-source visual foundation model developed by Meta AI Laboratory.
2) The feature regression module employs a classic multi-layer perceptron (MLP) structure.

![end-to-end LAD estimation model](https://github.com/user-attachments/assets/c7e6800c-42ce-4901-8033-b1002e4cf418)

# Hardware platform
CPU: Intel(R)xeon(R)Platinum 8358P CPU@ 2.0 GHz  
GPU: NVIDIA GTX 4090 Ti  
CUDA version: 12.0

# virtual environment configuration
Python verion: 3.8  
Before using this model, please check your virtual environment (See environment.yml).  

# LAD estiamtion
Run the script inference_LAD_LoRA.py  
please change the test_dataset_path (Line 128), model_path (Line 134), and output csv_path (Line 182).  
Well-trained LAD estimation model, please download from the Google Drive:  
(https://drive.google.com/file/d/1Jert1xF1emxIw6ytJVBPHeN0G6O-uAxO/view?usp=sharing)

Notes: In the script, we pre-load the DINOv2-large model from the huggingFace website. We have downloaded, please downlaod it from the Google Drive and put it into the folder (47b73eefe95e8d44ec3623f8890bd894b6ea2d6c), then modify the pretrained_model_path in script (Line 39).  
(https://drive.google.com/file/d/1aAa74J3sMCOshOVehldklWyyvURGDBsQ/view?usp=drive_link)

# citation
If our code or models help your work, please cite our paper:  

@online{,  
  title={},  
  author={},  
  Journal={},  
  pages={},  
  year={}  
}
