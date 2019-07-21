# Decoupled Networks

Weiyang Liu, Zhen Liu, Zhiding Yu, Bo Dai, Rongmei Lin, Yisen Wang, James M.Rehg, Le Song

Georgia Institute of Technology, NVIDIA, Emory University, Tsinghua University, Ant Financial



## Abstract

* 이 논문에서는 기존 $$w$$ 와 $$x$$ 의 내적을  'decouple'하여 intra-class variation 과 semantic difference 에 대하여 조금 더 학습을 잘 하기 위한 방법을 제안하였다.
* decouple 을 수행하기 위한 다양한 operators 를 제안하였다.   
* 이렇게 decouple 하게 되면 신경망이 조금 더 빠르게 수렴하고, 강인해지도록 만드는 효과가 있다. 



## Introduction

* 기존 convolution 신경망의 $$<w,x> = w^Tx$$  는 intra-class variation 과 semantic difference(inter-class variation) 를 하나의 couple 형태로 측정하게 된다. 
  

* 위의 결과처럼 couple 형태로 측정하게 되면 다음과 같은 질문에 대하여 알기 어려워진다. 2개의 샘플의 내적 결과값이 크다면, '2개의 샘플의 semantic/label difference 를 가진것인지?' 'large intra-class variation을 가진것인지?' 
  (즉, 같은 클래스 내부에서 분리가 된것인지 , 다른 클래스의 성격을 갖고 있는 것인지에 대한 판단이 어렵다)

* 따라서, 기존 $$w^Tx$$ 의 식을 $$ \left\| w \right\| \left\| x \right\| \cos ({ { \theta  }_{ (w,x) } }) $$ 로 분리하였다. 
  angle은 semantic/label difference 를 나타내게 되고, feature의 norm은 intra-class variation을 나타내게 된다.
  (그림 1 참조)


* 그림 1
  
* $$\left\| w \right\| \left\| x \right\| \cos ({ { \theta  }_{ (w,x) } }) $$ 의 수식에서 $$h(\left\| w \right\| \left\| x \right\|)$$ 을 magnitude function,  $$ g({ \theta  }_{ (w,x) })$$  을 anuglar function 으로 정의한다.   이 때, $$ h(\left\| w \right\| \left\| x \right\|)$$  = $$\left\| w \right\| \left\| x \right\|$$  ,  $$ g({ \theta  }_{ (w,x) })$$ = $$\cos ({ { \theta  }_{ (w,x) } }) $$  을 나타낸다. 
  * manitude function 은 intra-class variation 을 나타내고, angular function 은 semantic difference 를 나타낸다.
    
* Decoupling 관점에서 바라보면, 기존 CNN은 norm의 곱형태로 intra-class variation 을 선형적으로 모델링할 수 있고, semantic difference를 cosine angle으로 설명할 수 있다는 강력한 가정을 만들어낸다. 그렇지만, 모든 측면에서 이 모델링이 최적으로 동작하지는 못한다. 
  
* DCNet(Decoupled Network) 은 다음과 같은 4가지 측면에서 장점을 가진다.
  * intra class variation 과 semantic difference를 더 잘 모델링 할 수 있을 뿐만 아니라 이러한 특징들을 바꾸지 않고 직접적으로 학습할 수 있도록 해준다.
  * bounded magnitude function 을 사용함으로써, 수렴속도를 빠르게 할 수 있다. 
    (bounded magnitude function 은 뒤에서 설명할 예정이지만, softmax 처럼 단순히 값의 범위를 정해놓는다고 생각하면 편하다.) 
  * Adversarial attacks 에 강하다. bounded 된 magnitude 에서 각 클래스의 feature를 사용하기 때문에 강인하다.
  * decoupled operators 는 매우 유연하게 사용 가능하며, architecture-agnostic 하다. 
    (VGG, GoogleNet등 어디서든 사용가능하다.)

* 이 논문에서는 2가지의 decoupled operators를 제안한다. 
  * Bounded operators
    * 조금 더 빠르게 수렴하며, adversarial attack 에 강인하다.
  * Unbounded operators
    * 더 많은 representational power 를 가진다. 
  * 추가적으로, operator radius 라는 개념을 제안한다.
    * operator radius 는 $$\left\| x \right\|$$ 의 입력에 따라 변화하는 magnitude function $$h()$$ 의 미분의 critical change를 나타낸다. 



## Decoupled Networks

### Reparametrizing Convolution via Decoupling

* Conventional : $$f(w, x) = <w, x> = w^Tx$$

* Decoupled form : $$f(w,x) = \left\| w \right\| \left\| x \right\| \cos ({ { \theta  }_{ (w,x) } })$$  

* Decoupled general form : $$f_d(w,x) = h(\left\| w \right\|, \left\| x \right\|) \cdot  g({ \theta  }_{ (w,x) })$$



### On Better Modeling of the Intra-class Variation

* angular function 은 오직 angle 만 입력으로 받기 때문에, 설계 하기에 상대적으로 쉽다.  
* magnitude function은 w, x 의 norm을 입력으로 받기 때문에, 설계 하기에 상대적으로 어렵다. 
* $$\left\| w \right\|$$ 은 kernel 그 자체의 가중치이기 때문에, 입력의 intra-class variation 보다는 kerenl 자체에 중요성을 가진다. 따라서, 우리는$$ \left\| w \right\|$$ 를 magnitude function 에 포함시키지 않는다. 
  ( 모두 같은 중요도로 할당 시킨다)
* $$\left\| w \right\|$$ 을 모두 같은 중요도로 할당시키면, 가능한 많은 kernel에 기반하여 network 가 decision을 하기 때문에, 일반화 성능이 증가한다. 하지만, representational power 는 감소할 수 있다. 
*  $$\left\| w \right\|$$ 을 다시 magnitude function 으로 가져와서 사용하면, weighted decoupled operators 로 사용가능하다. 
  (weighted decoupled operators 를 만드는 부분에 대해서는 뒷 절에서 다룰 예정이다.)



### Bounded Decoupled Operators

* $$|f_d(w,x)| \le c$$  where $$c$$ : *positive constant* 처럼 bounded 되어있는 decoupled operators 에 대하여 설명한다.
  (설명의 편의성을 위하여 magnitude function 에서 weight의 norm 은 제외한다.)

* **Hyperspherical Convolution**

  * $$ h(\left\| w \right\|, \left\| x \right\|) = \alpha $$  라고 가정하면, 우리는 아래 수식처럼 decoupled 된 hyperspherical convolution(SphereConv) 를 얻게된다. 

    ​													$$f_d(w,x) = \alpha \cdot g(\theta_{(w,x)})$$   $$,\alpha > 0$$ 

    

  * $$\alpha$$ 는 output의 scale을 조절하며, $$g(\theta_{(w,x)})$$ 은 unit hypersphere 의 geodesic distance 에 의해 결정된다. 일반적으로 결과물은 -1 부터 1 까지의 값으로 결정된다. 따라서, 최종 결과물은 [$$-\alpha, \alpha$$] 로 정해진다. 주로, $$\alpha$$ 의 값으로 1을 사용하는데, 이때는 SphereConv와 유사해진다.

  * 기하학적으로, SphereConv는 $$w$$ 와 $$x$$ 를 hypersphere 로 사영시킨 후에 ($$g(\theta) = cos(\theta)$$ 일 때) 내적을 수행한다.

  * Sphereconv에 따르면, 네트워크의 수렴속도가 증가한다. 
    

* **Hyperball Convolution**

  * Hyperball Conv. 는  $$h(\left\| w \right\|, \left\| x \right\|) = \alpha min(\left\| x \right\|, \rho)/\rho$$ 를 사용한다. 

      												$$ f_d(w,x) = \alpha \cdot \frac {min(\left\| x \right\|, \rho)}{\rho} \cdot g(\theta_{w,x})$$  

    

  * $$\rho$$ 는 saturation threshold 를 조정하게 된다. 

    * 만약, $$\left\| x \right\|$$ 가 $$\rho$$ 보다 커지게 된다면, magnitude 는 saturation 되면서 $$\alpha$$ 값을 출력한다. 
    * 반대의 경우에, magnitude function 은 $$\left\| x \right\|$$ 를 따라서 선형적으로 증가하게 된다. 

  * BallConv 는 SphereConv 에 비해 조금 더 유연하고 강인하다. SphereConv는 angle값만을 이용하기 때문에 $$w$$ 와 같은 방향의 $$x$$ 일 경우에 무조건 maximum output이 나오게 된다. 만약 매우 norm이 매우 작은 $$x$$ 가 입력으로 들어오게 될 경우 이를 amplify 하는 형태가 되는데, 이는 perturbation 에 취약한 형태가 될 수 있다는 것을 의미한다. 

  * 반면에 BallConv는 $$x$$ 의 norm이 작을 경우에, output도 작아질 수 있도록 한다. 또한, $$x$$ 의 norm이 작다는 것은 local patch가 정보가 별로 없고, 강조되면 안좋을 것을 뜻하기도 한다. 
    

* **Hyperbolic Tangent Convolution**

  * TanhConv는 BallConv에서 사용하였던 stepfunction 을 hyperbolic tangent function 으로 대체하였다. 

    ##### 														$$ f_d(w,x) = \alpha \,tanh (\frac {\left\| x \right\|}{\rho}) \cdot g(\theta_{w,x})$$  

  * $$\rho$$ 의 값은 decay curve를 조정하는데 사용된다. 

  * BallConv의 smooth version 으로 바라볼 수 있다.



### Unbounded Decoupled Operators

* **Linear Convolution**

  * 가장 간단한 Unbounded decoupled operator의 형태중 하나는 LinearConv 이다.
    

    ​														 $$f_d(w,x) = \alpha \, \left\| x \right\| \cdot g(\theta_{w,x}) $$

    

  * LinearConv는 weights를 hypersphere에 사영시키고, slope를 컨트롤 할 수 있는 파라미터가 있다는 점에서 기존 convolution 과 다르다. 
    

* **Segmented Convolution**

  * ㅇ
    
  * ㅇ
  * 

