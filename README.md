# Recent Face Recognition

* For implementation of recent face recognition papers.

  

## Datasets

**Training**

- [x] Faces-emore
  

**Test**

- [x] LFW
- [x] Agedb-30
- [x] cfp-fp



## Implemented Networks & Loss

- [ ] [CVPR 2015] Facenet - Triplet loss 
- [ ] [ECCV 2016] A Discriminative Feature Learning Approach for Deep Face Recognition - Center loss
- [ ] [CVPR 2017] Sphereface - A-Softmax loss
- [ ] [CVPR 2018] Cosface - Large margin cosine loss
- [ ] [CVPR 2018] Ring loss - Ring loss
- [x] [CVPR 2019] Arcface - Cosine margin loss



## How to train ?

1. Set-up virtual environment.

   ```
   conda create -n [your environment] python=3.6
   
   source activate [your environment]
   
   conda install pytorch=0.4.1 cuda90 torchvision -c pytorch
   
   pip install -r requirements.txt
   ```


2. Download train, test datasets
   ( https://github.com/TreB1eN/InsightFace_Pytorch —> 3.1 Data Preparation )

3. Let's train
   `python train.py --train_root [your data root] --epochs [epochs] --batch_size [batch_size] --save_root [root for saving weights, log file]`



## References

1. <https://github.com/TreB1eN/InsightFace_Pytorch>