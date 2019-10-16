# Awesome Face Recognition

[2019.10.14 - Debugging Log] 

* ArcFace
  * In training phase, I am trying to setting some parameter (feature_scale in ArcFace).
    (If the parameter is not set properly, the loss value changes to NaN)

* Decoupled Network
  * I am trying to implementation DecoupledNetwork on https://github.com/ChoSungMan/DCNets-pytorch
  * If the test is done, I will merge the two repos into one.
    
* RegularFace
  * Will be updated..

---





* For reviewing and implementing of recent face recognition papers.
* Paper review will be updated.
* Add google drive link for sharing my presentation file about face recognition paper.
  (https://drive.google.com/drive/folders/1CEbBRLORD03BZJfekhRJF8tjzloHmOHa?usp=sharing)



## Papers (novel loss function)

| Conf.    | Title                                                        | Images   | LFW(%) | YTF(%) | Link |
| -------- | ------------------------------------------------------------ | -------- | ------ | ------ | ---- |
| 15, CVPR | [FaceNet: A Unified Embedding for Face Recognition and Clustering (Triplet Loss)](https://arxiv.org/abs/1503.03832) | 200M     | 99.63  | 95.12  |      |
| 16, ECCV | [A Discriminative Feature Learning Approach for Deep Face Recognition (Center Loss)](https://ydwen.github.io/papers/WenECCV16.pdf) | 0.7M     | 99.28  | 94.9   |      |
| 16, ICML | [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) | WebFace  | 98.71  | -      |      |
| 17, CVPR | [SphereFace: Deep Hypersphere Embedding for Face Recognition (Angular Softmax Loss)](https://arxiv.org/pdf/1704.08063.pdf) | WebFace  | 99.42  | 95.0   |      |
| 18, CVPR | [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf) | 5M       | 99.73  | 97.6   |      |
| 18, CVPR | [Ring loss: Convex Feature Normalization for Face Recognition](https://arxiv.org/pdf/1803.00130.pdf) | MS-Celeb | 99.52  | -      |      |
| 19, CVPR | [AdaptiveFace: Adaptive Margin and Sampling for Face Recognition](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/papers/2019adaptiveface.pdf) | 5M       | 99.62  | -      |      |
| 19, CVPR | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf) | WebFace  | 99.53  | -      |      |
| 19, CVPR | [RegularFace: Deep Face Recognition via Exclusive Regularization](http://mftp.mmcheng.net/Papers/19cvprRegularFace.pdf) | WebFace  | 99.33  | 94.4   |      |

  

## Papers (various)

| Conf.    | Title                                                        | About                                                        |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 17, NIPS | [Deep Hyperspherical Learning](https://arxiv.org/pdf/1711.03189.pdf) | Feature representation (hyper spherical convolution)         |
| 18, CVPR | [Decoupled Networks](https://arxiv.org/pdf/1804.08071.pdf)   | Feature representation (decoupled network)                   |
| 18, CVPR | [Towards Pose Invariant Face Recognition in the Wild](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Towards_Pose_Invariant_CVPR_2018_paper.pdf) | Pose invariant model for face recognition                    |
| 19, CVPR | [Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition](https://arxiv.org/pdf/1809.00338.pdf) | Age invariant model for face recognition                     |
| 19, CVPR | [R3 Adversarial Network for Cross Model Face Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_R3_Adversarial_Network_for_Cross_Model_Face_Recognition_CVPR_2019_paper.pdf) | Cross model face recognition                                 |
| 19, CVPR | [Feature Transfer Learning for Face Recognition with Under-Represented Data](http://cvlab.cse.msu.edu/pdfs/Yin_Yu_Sohn_Liu_Chandraker_CVPR2019.pdf) | Augment the feature space of under-represented subjects from the regular subjects |
| 19, CVPR | [Low-Rank Laplacian-Uniform Mixed Model for Robust Face Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Low-Rank_Laplacian-Uniform_Mixed_Model_for_Robust_Face_Recognition_CVPR_2019_paper.pdf) | Robustness in occlusion, pixel corrpution, disguise          |
| 19, CVPR | [Unequal-training for Deep Face Recognition with Long-tailed Noisy Data](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Unequal-Training_for_Deep_Face_Recognition_With_Long-Tailed_Noisy_Data_CVPR_2019_paper.pdf) | Unequal-training for long-tailed noisy data                  |

  




## Implementation 

### Datasets

**- Training**
- [x] Faces-emore

  

**- Evaluation**
- [x] LFW
- [x] Agedb-30
- [x] CFP-FP



### Implemented Networks

- [ ] [CVPR 2018] Decoupled Networks
- [x] [CVPR 2019] Arcface 
- [ ] [CVPR 2019] Regularface



### How to train ?

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



### Modules (TO DO)

- [ ] Decoupled Operators
- [ ] Feature Visualization Module
- [ ] Simple Test GUI (face registration, face detection, face recognition)





## References

1. <https://github.com/TreB1eN/InsightFace_Pytorch>
2. https://github.com/ronghuaiyang/arcface-pytorch
3. https://github.com/wy1iu/DCNets

