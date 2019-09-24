# 论文阅读笔记
一些论文笔记，整理

## 目录
- [RNN](#rnn)
- [CNN](#cnn)
- [GAN](#gan)
  - [分辨率提升](#分辨率提升)
  - [3DGAN](#3dgan)
- [知识图谱](#知识图谱)

### RNN
- 流场可视化：预测粒子跟踪中的数据访问模式（LSTM）

  "Access pattern learning with long short-term memory for parallel particle tracing"
 
- 预测内存存取模式，预取数据(LSTM)
  [Learning Memory Access Patterns](https://arxiv.org/pdf/1803.02329.pdf)（LSTM）

  
### CNN

### GAN

最新发展和应用 https://www.zhihu.com/question/52602529/answers/updated

### 分辨率提升
对于低分辨率的LIC图像或是低分辨率的streamlines数据进行分辨率提升，避免精细的插值运算

- SRGAN

  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)（2017 CVPR）
- ESRGAN

  [Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) （2018 ECCV）
- 无监督分辨率提升（CNN）

  [Unsupervised Learning for Real-World Super-Resolution](https://128.84.21.199/pdf/1909.09629.pdf) （2019 ICCV）
- 单图像无监督的退化学习 （GAN）

  [Unsupervised Degradation Learning for Single Image Super-Resolution](https://arxiv.org/abs/1812.04240v1) (2018 CVPR)

  学习了cyclegan（风格迁移）中image to image的思想

  [知乎笔记](https://zhuanlan.zhihu.com/p/52237543)

- 流场可视化：时序的超分辨率(RNN+GAN)

  [TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization ](https://www3.nd.edu/~cwang11/research/vis19-tsr.pdf)


### 3Dgan 

- 生成3D物体的体数据
  [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](http://3dgan.csail.mit.edu/)

### 知识图谱
