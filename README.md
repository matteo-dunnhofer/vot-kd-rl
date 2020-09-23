# REPOSITORY STILL UNDER DEVELOPMENT


# Tracking-by-Trackers
Official implementation of the **TRAS**, **TRAST**, **TRASFUST** (ACCV 2020), **A3CT**, **A3CTD** (ICCVW 2019) trackers, including complete **training code** and trained models.

## Trackers
The repository contains the implementation of the following trackers.  

### TRAS, TRAST, TRASFUST
**[[Paper]](https://arxiv.org/abs/2007.04108)  [[Raw results]]() [[Pretrained Model]]() **
    

![ACCV_qualex_video](https://youtu.be/uKtQgPk3nCU)

### A3CT, A3CTD
**[[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Dunnhofer_Visual_Tracking_by_Means_of_Deep_Reinforcement_Learning_and_an_ICCVW_2019_paper.html)  [[Raw results]]()
  [[Pretrained Model]]() **
    
![ICCVW_qualex_video](https://youtu.be/jSGLafk4-G4)
<iframe width="560" height="315" src="https://www.youtube.com/embed/jSGLafk4-G4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/dontfollowmeimcrazy/vot-kd-rl.git
```
   
## Test
Run the script track/run_test.py by specifing the tracker with the ```--tracker``` option and the dataset with the ```--dataset```.
```bash
cd track
python run_test.py --tracker TRAS --dataset GOT-10-test    
```  

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