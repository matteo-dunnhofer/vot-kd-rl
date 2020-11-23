# REPOSITORY STILL UNDER DEVELOPMENT


# Tracking-by-Trackers
Official implementation of the tracking-by-trackers concept proposed in the paper
**"Tracking-by-Tracker with a Distilled and Reinforced Model"**.

## Trackers
The repository contains the official implementation of the **TRAS**, **TRAST**, **TRASFUST** (ACCV 2020), **A3CT**, **A3CTD** (ICCVW 2019) trackers, including trained models. 

### TRAS, TRAST, TRASFUST
**[[Paper]](https://arxiv.org/abs/2007.04108)  [[Qualitative results]](https://youtu.be/uKtQgPk3nCU) [[Raw results]](https://drive.google.com/drive/folders/1Ppj9VIQ6n0KavnaZ2E1S-pKFSrRjQGuW?usp=sharing) [[Pretrained Model]](https://drive.google.com/file/d/1-ijK1kIqpBlSFTbPYNA9Ddfkgn3qrgSI/view?usp=sharing)**
    

![ACCV2020](./accv2020.jpg)

### A3CT, A3CTD
**[[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Dunnhofer_Visual_Tracking_by_Means_of_Deep_Reinforcement_Learning_and_an_ICCVW_2019_paper.html)  
[[Qualitative results]](https://youtu.be/jSGLafk4-G4) [[Raw results]]()
  [[Pretrained Model]]()**
    
![ICCVW_qualex_video](https://youtu.be/jSGLafk4-G4)



## Installation

Code has been developed and tested on Ubuntu 18.04 with python 3.6, PyTorch 1.4.0, and CUDA 10.

#### Clone the GIT repository.  
```bash
git clone https://github.com/dontfollowmeimcrazy/vot-kd-rl.git
```

#### Set paths to checkpoint. 
Download the pretrained weights file from here, put wherever you want in your machine, and set the variable ```CKPT_PATH``` variable (contained in file ```track/config_track_accv.py```) to point it.
   
## Test

#### Set path to benchmark datasets.  
In the file ```track/config_track_accv.py``` set the ```DATA_PATH``` variable to path of where the benchmark datasets are stored.

Run the script track/run_test.py by specifing the tracker with the ```--tracker``` option and the dataset with the ```--dataset```.
```bash
cd track
python run_test.py --tracker TRAS --dataset OTB2015  
```  

For TRAST and TRASFUST trackers you either need to:
	* provide the implementation of "teacher" trackers according to the [GOT-10k toolkit]() tracker definition, and initialize them in lines of the ```track/Trackers.py``` file.
	* use the precomputed results of the "teacher" trackers

## Training
Training code will be released soon!


## References
If you find this work useful please cite
```
@InProceedings{Dunnhofer_2020_ACCV,
author = {Dunnhofer, Matteo and Martinel, Niki and Micheloni, Christian},
title = {Tracking-by-Trackers with a Distilled and Reinforced Model},
booktitle = {Asian Conference on Computer Vision (ACCV)},
month = {Dec},
year = {2020}
}

@InProceedings{Dunnhofer_2019_ICCVW,
author = {Dunnhofer, Matteo and Martinel, Niki and Luca Foresti, Gian and Micheloni, Christian},
title = {Visual Tracking by Means of Deep Reinforcement Learning and an Expert Demonstrator},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2019}
}   
``` 

## Acknowledgments 