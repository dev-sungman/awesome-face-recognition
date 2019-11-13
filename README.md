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

