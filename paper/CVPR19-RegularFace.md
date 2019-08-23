## Authors

Kai Zhao, Jingyi Xu, Ming-Ming Cheng

TKLNDST, CS, Nankai University



## Introduction

* 기존 연구(Center loss, SphereFace, CosFace, ArcFace 등)들은 Intra-class 의 compactness 를 맞추는데 중점이 되었다. 
* 본 논문에서는 inter class 를 잘 구분하도록 네트워크를 설계하는 방법에 대하여 제안하였고, 이 방법이 기존 연구들에서는 제안되지 않았던 형태라고 언급한다. 
* '*exclusive regularization*' 을 classification layer 에 적용해서, 서로 다른 클래스들 간의 간격을 크게 만들어 준다. 



## Obeservation and Motivation

* Center-Loss 와 SphereFace 에서 영감을 받아 만들어졌다.

* **Center-Loss**

  * embedding 된 features 와 그 중심을 제한하여, intra-class 를 compactness 하게 만들어준다. 
    
    $$
    L_{center} = \frac 1 2 \sum _{i=1}^{N}{ \left\| x_i - c_{y_i} \right\|^2_2  }
    $$

  

  * $$ x_i \in \R^K$$ 는 샘플 $$i$$ 의 feature embedding 을 나타내고, $$c_{y_i}$$ 는 label이 $$y_i$$ 인 샘플의 embedding center 를 나타낸다. 
    

* **Softmax Loss & Angular Softmax Loss**

  *  Softmax Loss
    
    $$
    p_c(x_i) = \frac {e^{W^T_cx_i + b_c}}{\sum _{j=1}^{C}{ e^{W^T_jx_i + b_j}}}
    $$
    

  * Angular Softmax Loss

    
    $$
    p_c(x_i) = \frac {e^{\left\| x_i \right\|cos(\phi_{i,c})}}{\sum _{j=1}^{C}{ e^{\left\| x_i \right\|cos(\phi_{i,j})}}}
    $$
    

  * $$\phi_{i,j}$$ 은 feature embedding $$x_i$$ 와 weight vector $$W_j$$ 의 각도를 나타낸다.
    softmax-loss 를 최소화시키는 것은 $$\phi_{i,j}$$ 를 최소화시키는 문제와 동일하기 때문에, $$W_j$$ 는 모든 $$x_i$$ 의 클러스터 중심으로 생각할 수 있다. ($$y_i =j$$ 일때)



* **A-Softmax Loss **

  * SphereFace 는 angular softmax loss 에서 angular margin 을 추가한 개념이다. 수식은 아래와 같다. 
    
    $$
    p_c(x_i) = \frac {e^{\left\| x_i \right\|cos(m\cdot\phi_{i,y_i})}}{e^{\left\| x_i \right\|cos(m\cdot\phi_{i,y_i})}+\sum _{j\neq y_i}{ e^{\left\| x_i \right\|cos(\phi_{i,j})}}}
    $$

  

  * $$m \in Z_+ = \left\{ 1, 2,...\right\}$$ 은 margin 을 조절하는 파라미터이다. $$m$$ 이 1일 경우에, Angular Softmax Loss 와 식이 같아진다. 
    

* Feature Embeddings
  <img src="../images/RegularFace/feature_embedding.png" width="100%" height="50%">
* 

