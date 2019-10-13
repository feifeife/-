# 论文阅读笔记
一些论文笔记，整理

## 目录
- [RNN](#rnn)
- [CNN](#cnn)
	- [经典网络结构](#经典网络结构)
	- [风格迁移](#风格迁移)
	- [目标检测](#object-detection)
	- [结构分析](#cnn结构分析)
	- [视频](#视频预测未来帧)
	- [streamline](#提取流场streamline特征)
	- [涡识别](#提取流场vortex)
- [GAN](#gan)
	- [Volume render](#volume)
	- [图像生成](#gan图像生成)
  - [超分辨率](#超分辨率)
  - [3DGAN](#3dgan)
- [NLP](#nlp)
	- [Memory机制](#memory)
- [HPC](#hpc)
	- [并行粒子追踪](#并行粒子追踪)

### RNN
- 流场可视化：预测粒子跟踪中的**数据访问**模式（LSTM）

  >*Hong, Fan & Zhang, Jiang & Yuan, Xiaoru. (2018). Access Pattern Learning with Long Short-Term Memory for Parallel Particle Tracing. 76-85. 10.1109/PacificVis.2018.00018.*

- <span id="tsr">流场可视化：TSR-TVD：流场时序的高分辨率</span>
  >*Han, J., & Wang, C. (2019). [TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization ](https://www3.nd.edu/~cwang11/research/vis19-tsr.pdf). IEEE TVCG.*
  
  * 目标：对于两个时间步t,k之间的体数据，找到一个映射Φ，得到中间时间步的体数据
  
  * 网络结构RGN：RNN+GNN
    * Generate：一个生成网络G(类似于bilstm的思想)
  
		a. 预测模型（biRNN）：前向预测（从t开始）和后向预测（从k开始）
    
		b. 融合模型：将前向后向的各个对应的时间步融合
    * Discriminator：一个判别模型D（二分类的CNN）
  
		区分出fake的体数据V
    * 3个loss funtion：（1）生成对抗loss（2）体数据和GroundTruth对比的损失（生成器尽可能的还原GT的体数据）（3）CNN中间层特征损失：		提取判别器D中的卷积层参数，使生成器生成的体数据在每一层都尽可能的贴近GroundTruth

- 流体可视化：预测流体**压力场p**的变化(RNN)
  >*Wiewel, S., Becher, M., & Thürey, N. (2018). [Latent Space Physics: Towards Learning the Temporal Evolution of Fluid Flow.](https://arxiv.org/abs/1802.10123) Comput. Graph. Forum, 38, 71-82.*
  
  由[TSR-TVD](#tsr)引用
  
  RNN预测3d物理函数的时间演变
  
  Navier-stokes方程：
  ![](https://cdn.mathpix.com/snip/images/CTOYuXYlH8iRbLGOmp50PTqIovm4q9IrrIGlzxtYiRo.original.fullsize.png)

  - Step 1:cnn对**p**的降维，autoencoder的模式，提取其中的latent vector作为特征向量**p**
  - Step 2:将序列**c**输入LSTM+CNN的混合架构（对于高维物理输出比较重要），输出未来h个时间步的预测值

- 预测内存存取模式，预取数据(LSTM)
  >*Hashemi, M., Swersky, K., Smith, J.A., Ayers, G., Litz, H., Chang, J., Kozyrakis, C.E., & Ranganathan, P. (2018). [Learning Memory Access Patterns](https://arxiv.org/pdf/1803.02329.pdf) ArXiv, abs/1803.02329.*
  
### CNN
#### 经典模型
- ResNet
	>*He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)](https://arxiv.org/abs/1512.03385)[(code)](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), 770-778.*

详解：https://zhuanlan.zhihu.com/p/56961832

https://blog.csdn.net/lanran2/article/details/79057994

解决了什么问题：https://www.zhihu.com/question/64494691/answer/786270699

其他变体：https://www.jiqizhixin.com/articles/042201

- ResNeXt
>*Xie, S., Girshick, R.B., Dollár, P., Tu, Z., & He, K. (2016). [Aggregated Residual Transformations for Deep Neural Networks.](https://arxiv.org/abs/1611.05431) [(code)](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/resnext.py) 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5987-5995.*

详解：https://zhuanlan.zhihu.com/p/78019001

https://blog.csdn.net/u014380165/article/details/71667916

- DenseNet
	>*Huang, G., Liu, Z., Maaten, L.V., & Weinberger, K.Q. (2016). [Densely Connected Convolutional Networks.](https://arxiv.org/abs/1608.06993) [(code)](https://github.com/liuzhuang13/DenseNet) 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261-2269.*
	
	
#### 风格迁移
- <span id="perceptual">Perceptual Loss:风格迁移</span>
	>*Johnson, J., Alahi, A., & Fei-Fei, L. (2016). [Perceptual Losses for Real-Time Style Transfer and Super-Resolution.](https://arxiv.org/abs/1603.08155) ECCV.*
- <span id="deepae">deep VAE学习图像语义特征</span>
	>*Hou, X., Shen, L., Sun, K., & Qiu, G. (2016). [Deep Feature Consistent Variational Autoencoder.](https://arxiv.org/abs/1610.00291) 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), 1133-1141.*
	
#### Object Detection
- Cascade R-CNN
	>*Cai, Z., & Vasconcelos, N. (2017). [Cascade R-CNN: Delving Into High Quality Object Detection.](https://arxiv.org/abs/1712.00726) 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6154-6162.*
	
	在提出候选框anchor box时，由于IoU（和GT的重叠部分）阈值设为0.5偏低，造成很多close false positive（稍微高于0.5，但是是负样本）重复无用的候选框，因此采用级联的方式逐步提升，在不同的阶段设置不同的阈值
	
论文笔记：https://wanglichun.tech/algorithm/CascadeRCNN.html

	 https://arleyzhang.github.io/articles/1c9cf9be/
	 
	 https://github.com/ming71/CV_PaperDaily/blob/master/CVPR/Cascade-R-CNN-Delving-into-High-Quality-Object-Detection.md
	 
- CBNet：多模型特征融合
	>*Liu, Yudong & Wang, Yongtao & Wang, Siwei & Liang, TingTing & Zhao, Qijie & Tang, Zhi & Ling, Haibin. (2019). [CBNet: A Novel Composite Backbone Network Architecture for Object Detection.](https://arxiv.org/abs/1909.03625v1)*

#### CNN结构分析
- 利用CNN中间层作为loss
  >*Zhang, R., Isola, P., Efros, A.A., Shechtman, E., & Wang, O. (2018). [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2Fcs%2FCV+%28ArXiv.cs.CV%29). 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 586-595.*
	

#### 视频预测未来帧
- Deep Voxel Flow 
  >*(oral)Liu, Z., Yeh, R.A., Tang, X., Liu, Y., & Agarwala, A. (2017). Video Frame Synthesis Using Deep Voxel Flow. 2017 IEEE International Conference on Computer Vision (ICCV), 4473-4481.*
  
  利用现有的video进行无监督的学习
#### 提取流场streamline特征
- 流场可视化：Flownet：streamlines的聚类和选择（去噪）
  更好的呈现可视化的效果，防止重叠现象等
  >*Han, J., Tao, J., & Wang, C. (2018). FlowNet: [A Deep Learning Framework for Clustering and Selection of Streamlines and Stream Surfaces.](https://www3.nd.edu/~cwang11/research/tvcg19-flownet.pdf) IEEE transactions on visualization and computer graphics.*
  
  将streamlines转化成体数据->输入3DCNN->encoder->latent value vector(特征向量)->decoder->生成体数据->与原数据计算loss
  
  得到**中间的特征向量表示**后，用t-SNE降维，用DBSCAN聚类
  
  如何生成中间的特征表示？？？（[3Dshape gan生成](#3dgan)）
#### 提取流场vortex
* 流场可视化：学习unsteady场的最优参考系参数

	> 1. *Kim, B., & Günther, T. (2019). [Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks.](https://arxiv.org/abs/1903.10255) Comput. Graph. Forum,Eurovis 38, 285-295.*

	* input：2D向量场
	* output：参考系的旋转和平移参数θ
	
	由[TSR-TVD](#tsr)引用
	
	通过对steady的向量场进行变换获得train data，输入unsteady的向量场得到其形变的参数，就可以准确识别其vortex。

- 流场可视化：R-CNN识别分类vortex，先检测再分类
	> 2. *Strofer, C.M., Wu, J., Xiao, H., & Paterson, E.G. (2018). [Data-Driven, Physics-Based Feature Extraction from Fluid Flow Fields using Convolutional Neural Networks.](https://arxiv.org/abs/1802.00775) Commun. Comput. Phys.*
	* input:流场被分为三维网格，每个point都带有一个特征向量（类似RGB）：具有伽利略不变性的物理特征
	* Step 1：Region Proposal：目标检测，得到候选框
	* Step 2：分类,识别不同特性的流场
- 流场可视化：CNN识别分类vortex（逆时针、顺时针）直接检测
	> 3. *Bin, T.J. (2018). [CNN-based Flow Field Feature Visualization Method.](https://pdfs.semanticscholar.org/de16/9148f9c8484d175f92463af461a2bdfb3605.pdf) IJPE*
	* input：用9×9×2的向量场，u,v双通道
	* output：对vortex分类，分成顺时针、逆时针和普通场
- 流场可视化：CNN识别分类vortex，对每个点分类
	> 4. *Deng, Liang & Wang, Yueqing & Liu, Yang & Wang, Fang & Li, Sikun & Liu, Jie. (2018). [A CNN-based vortex identification method.](https://sci-hub.tw/10.1007/s12650-018-0523-1#) Journal of Visualization. 22. 10.1007/s12650-018-0523-1. *
	- Step 1：用IVD方法给每个点打上标签，vortex：1，non-vertex：0
	- Step 2：对于每个点，取一个15×15的patch，放进cnn训练，二分类确定这个点是否属于vortex
### GAN

最新发展和应用 https://www.zhihu.com/question/52602529/answers/updated
- <span id="gan">综述 : 损失函数、对抗架构、正则化、归一化和度量方法</span>
	>*Kurach, K., Lucic, M., Zhai, X., Michalski, M., & Gelly, S. (2018). [A Large-Scale Study on Regularization and Normalization in GANs.](https://arxiv.org/abs/1807.04720) ICML.*
	
	资料：https://daiwk.github.io/posts/cv-gan-overview.html
	
- 条件转移GAN：更少的数据要求+更多分类
	>*Wu, C., Wen, W., Chen, Y., & Li, H. (2019). [Conditional Transferring Features: Scaling GANs to Thousands of Classes with 30% Less High-quality Data for Training.] (https://www.arxiv-vanity.com/papers/1909.11308/) CVPR*

#### VOLUME
- INSITUNet:流场可视化图片生成(探索参数空间)
  >*He, W., Wang, J., Guo, H., Wang, K., Shen, H., Raj, M., Nashed, Y.S., & Peterka, T. (2019). [InSituNet: Deep Image Synthesis for Parameter Space Exploration of Ensemble Simulations.](https://arxiv.org/abs/1908.00407) IEEE transactions on visualization and computer graphics.*
simulation, vi-sual mapping, and view parameters

  输入模拟参数(simulation, visual mapping, and view parameters)，实时生成可视化图像（RGB）->可以探索参数空间对可视化图像的影响
  * in&out对于不同的**视觉参数**生成**可视化图像**，作为输入输出输入模型
  * 模型结构：
	
	* GAN:卷积回归模型（生成图像）+判别器D（对比GroundTruth）:内部都是resnet
		
	GAN结构来自于已有GAN模型（[SN-GAN](#sngan)和[ganlandsacpe](#gan)）
		
	* 特征提取网络：VGGnet
		
	思路来自图像合成[Perceptual Losses](#perceptual)（CNN）和[SRGAN](#srgan)和[Deep feature VAE](#deepae)
	
	
- 体数据渲染生成
  >*Berger, M., Li, J., & Levine, J.A. (2017). [A Generative Model for Volume Rendering.](https://arxiv.org/abs/1710.09545) IEEE Transactions on Visualization and Computer Graphics, 25, 1636-1650.*

#### GAN图像生成
- <span id="sngan">SN-GAN：谱归一化，解决训练不稳定</span>
	>*Miyato, Takeru & Kataoka, Toshiki & Koyama, Masanori & Yoshida, Yuichi. (2018).(oral) [Spectral Normalization for Generative Adversarial Networks.](https://arxiv.org/abs/1802.05957) IEEE ICLR*
#### 超分辨率
对于低分辨率的LIC图像或是低分辨率的streamlines数据进行分辨率提升，避免精细的插值运算
##### 体数据的超分辨率
- 体数据的上采样(CNN)
  >*Zhou, Z., Hou, Y., Wang, Q., Chen, G., Lu, J., Tao, Y., & Lin, H. (2017). Volume upscaling with convolutional neural networks. CGI.*

- 时序流体数据tempoGAN
  >*Xie, You & Franz, Erik & Chu, Mengyu & Thuerey, Nils. (2018). [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow.](https://arxiv.org/abs/1801.09710) ACM Transactions on Graphics. 37. 10.1145/3197517.3201304.*
  
  input：单时间步的低分辨率流体数据
  
- Point cloud 点云图的上采样(GAN)
  >*Li, R., Li, X., Fu, C., Cohen-Or, D., & Heng, P.A. (2019). [PU-GAN: a Point Cloud Upsampling Adversarial Network](https://liruihui.github.io/publication/PU-GAN/). ArXiv, abs/1907.10844.*

##### 图片的超分辨率
- SRCNN（CNN）
  >*Dong, Chao & Loy, Chen Change & He, Kaiming & Tang, Xiaoou. (2014). [Image Super-Resolution Using Deep Convolutional Networks.](https://arxiv.org/pdf/1501.00092.pdf) IEEE Transactions on Pattern Analysis and Machine Intelligence. 38. 10.1109/TPAMI.2015.2439281.*
- <span id="srgan">SRGAN</span>
  >*Ledig, C., Theis, L., Huszar, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., & Shi, W. (2016). P[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)2017 IEEE (CVPR), 105-114.*
- ESRGAN
  >*Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., & Loy, C.C. (2018). [Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219). ECCV Workshops.*
- GAN生成高分辨率的突破工作
  >*Wang, T., Liu, M., Zhu, J., Tao, A., Kautz, J., & Catanzaro, B. (2017). [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs.](https://arxiv.org/abs/1711.11585) 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8798-8807.*
  
  知乎笔记：https://zhuanlan.zhihu.com/p/35955531
- 无监督分辨率提升（CNN）
  >*Lugmayr, Andreas et al. (2019).[Unsupervised Learning for Real-World Super-Resolution](https://128.84.21.199/abs/1909.09629) ICCV*

- 单图像无监督的退化学习 （GAN）
  >*Zhao, T., Ren, W., Zhang, C., Ren, D., & Hu, Q. (2018). [Unsupervised Degradation Learning for Single Image Super-Resolution](https://arxiv.org/abs/1812.04240v1). CVPR, abs/1812.04240.*
  学习了cyclegan（风格迁移）中image to image的思想

  [知乎笔记](https://zhuanlan.zhihu.com/p/52237543)

- 流场可视化：时序的超分辨率(RNN+GAN)
  >*Han, J., & Wang, C. (2019). [TSR-TVD: Temporal Super-Resolution for Time-Varying Data Analysis and Visualization ](https://www3.nd.edu/~cwang11/research/vis19-tsr.pdf). IEEE TVCG.*


#### 3Dgan 

- 生成3D物体的体数据
  >*Wu, J., Zhang, C., Xue, T., Freeman, W.T., & Tenenbaum, J.B. (2016). [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](http://3dgan.csail.mit.edu/) NIPS.*
  
  >*Mo, K., Guerrero, P., Yi, L., Su, H., Wonka, P., Mitra, N., & Guibas, L.J. (2019). [StructureNet: Hierarchical Graph Networks for 3D Shape Generation.](https://cs.stanford.edu/~kaichun/structurenet/) ArXiv, abs/1908.00575.*
  
#### 预测视频的未来帧

>*Lotter, W., Kreiman, G., & Cox, D.D. (2015). [Unsupervised Learning of Visual Structure using Predictive Generative Networks](https://arxiv.org/abs/1511.06380). ArXiv, abs/1511.06380.*

### NLP
#### Attention
- 可解释性
	>*Vashishth, S., Upadhyay, S., Tomar, G.S., & Faruqui, M. (2019). [Attention Interpretability Across NLP Tasks.](https://www.arxiv-vanity.com/papers/1909.11218/)*
#### Memory
- Memory机制：attention中加入存储器层(Facebook)
	>*Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., & J'egou, H. (2019). [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242). ArXiv, abs/1907.05242.*

### HPC

- 对HPC中的异常进行可视化(转化成向量表示)
  >*Xie, C., Xu, W., & Mueller, K. (2018). A Visual Analytics Framework for the Detection of Anomalous Call Stack Trees in High Performance Computing Applications. IEEE Transactions on Visualization and Computer Graphics, 25, 215-224.*
#### 并行粒子追踪
- 并行粒子追踪综述
	>*Zhang, Jiang & Yuan, Xiaoru. (2018). [A survey of parallel particle tracing algorithms in flow visualization.](http://vis.pku.edu.cn/research/publication/jov18-pptsurvey.pdf) Journal of Visualization. 21. 10.1007/s12650-017-0470-2. *
