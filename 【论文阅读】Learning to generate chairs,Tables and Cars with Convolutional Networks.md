## INTRODUCTION
如何生成一个自然图像：1.学习要生成的图像的分布；2.学习一个生成器：生成服从该分布的图像的； （本文主要研究第二步）  
**问题概括**   
给定high-level descriptions of a set of images 训练出 geneerator  
**本文的方法**   
‘up-convolutional’ 生成网络  
**目标**  
1. 知识迁移（同一类）:由于数据量小，网络可以利用从其他椅子中学得的知识，推断剩余的视点（viewpoint）
2. 知识迁移（不同类）：网络可以将从桌子中学得的知识，转移到椅子上
3. 特征的加减法
4. 在类和类之间的不同对象之间的插值
5. 随机生成全新的椅子

## Related Work
**RBM ：Boltzmann machines**   
> G. E. Hinton and R. R. Salakhutdinov, “Reducing the
dimensionality of data with neural networks,” Science, pp.
504–507, 2006.  

**DBM ：Deep Boltzmann Machines**
> R. Salakhutdinov and G. E. Hinton, “Deep boltzmann
machines,” in AISTATS, 2009, pp. 448–455

**CDBNs : Convolutional Deep Belief Networks** (making use of “unpooling”)
> H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng, “Convolutional
deep belief networks for scalable unsupervised
learning of hierarchical representations,” in ICML, 2009,
pp. 609–616.

**ShapeNets** :(training a 3D variant of CDBN to generate3D models of furniture) 
> Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang,
and J. Xiao, “3D ShapeNets: A deep representation for
volumetric shapes,” in CVPR, 2015, pp. 1912–1920.

*另一种方法是对数据分布的定向图形模型进行训练*  
**GAN : generative adversarial networks**  (using a ”deconvolutional” generative network)
==与本文的体系结构相似==
> I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu,
D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio,
“Generative adversarial nets,” in NIPS, 2014, pp. 2672–2680.

**conditional RBMs :**  缺点是：推导过程很复杂 an expensive inference
procedure  
> S. Reed, K. Sohn, Y. Zhang, and H. Lee, “Learning to
disentangle factors of variation with manifold interaction,”
in ICML, 2014, pp. 1431–1439.

## Model Description

**数据集D** ={(c1,v1,θ1),...,(cN,vN,θN)}  
c: model id   
v: azimuth and elevation of camera position  #相机位置的方位和高度  
θ: parameters of additional artificial transformations  #人工转化的参数    
包括 ：旋转(12%)、位移(10%)、放大(135%)、水平(垂直)拉伸(10%)、色调(random)、饱和度(25% -400%)、亮度(35% -300%)   
**目标集合O** ={(x1,s1),...,(xN,sN)}  
x: 输出的三通道图像  
s：一通道的obejct 范围 （使得 很容易将object 和背景区分开）  
![image](https://note.youdao.com/yws/api/personal/file/12A5074E3443415E84652456AD2078B4?method=download&shareKey=d6b28f180ef961ee3671e934c1c58db4)

**Datasets**  
1. 1393个对齐的椅子模型，每一个有62个视点：31 azimuth angels（step = 11°） and  2 elevation angels（20° & 30°）   
azimuth：方位  
elevation：海拔高度  
2. 剔除一些相似的，最后还剩809个模型。
3. resize 128 *128 保持长宽比，空白的地方用白色填充


## Training Parameters
 caffe CNN  
 adam β1 =0.9  β2 =0.99   
 learning_rate =0.0005 for  250 000 mini batch iterations  
 
 ## Experiments
 
 1. Modeling transformations 
 红色框的是原图，两边都是对不同的θ的改变
![image](https://note.youdao.com/yws/api/personal/file/3758BA27FC8248B09F0043F498F3FE6A?method=download&shareKey=44101be1cd3f6943f76d4e78c2f83cde)  
2. ==Interpolation between viewpoints==水平
 ![image](https://note.youdao.com/yws/api/personal/file/55734D6FDD524E0D8F16BC139AB7B96C?method=download&shareKey=10340dfc0a9b7f07dfea4c3ea2df0220)

3. ==Elevation transfer and extrapolation==垂直
![image](https://note.youdao.com/yws/api/personal/file/7CBAB66D4509408382540AA15242C30F?method=download&shareKey=e97cab6d04ab45f641613e728fd41622)

4. Interpolation between styles  也支持多个
![image](https://note.youdao.com/yws/api/personal/file/0F6757F2A9E1476DB3F4944D64E176E2?method=download&shareKey=598f0aed7fae6a427a727a1e3643bec7)
5. Feature space ==arithmetics==  
6. Correspondences 
![image](https://note.youdao.com/yws/api/personal/file/56000BCA8D1143258DCA2965D6D96CCC?method=download&shareKey=e53893502e96afab84aa2c1d797b0c57)