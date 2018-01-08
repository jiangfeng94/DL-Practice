## Introduction
 In 2014, Alexey Dosovitsky 给定view point 和class 合成3D模型   
在任何两个图像类之间进行插值，创建一组变形图像 如从扶手椅子到凳子的变化
> Dosovitskiy A., Springenberg J. T., Brox T. Learning to Generate Chairs with Convolutional
Neural Networks 
> CoRR. — 2014. — Vol. abs/1411.5928. — URL: http://arxiv.org/
abs/1411.5928.

[DCGAN] In 2015,Alec Radford showed that it is possible to build deep convolutional models using GAN.
 > Radford A., Metz L., Chintala S. Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks
> CoRR.—2015.—Vol. abs/1511.06434.—URL:
http://arxiv.org/abs/1511.06434.


但是根据DCGAN的限制条件，GAN没法用于Dosovikiy 的生成器：全链接的隐藏层应该被删除，但是这些层是必要的。  
==it is indicated in the restrictions that the fullyconnected
hidden layers should be removed wherever possible. But such layers are necessary, so that
the model be able to build a powerful internal representation==

本文提出一种更为强大的GAN方法，通过生成模型和判别模型的最佳实践（best practice）来处理图片。

## Related Work
1.  Generative models
![image](https://note.youdao.com/yws/api/personal/file/12A5074E3443415E84652456AD2078B4?method=download&shareKey=d6b28f180ef961ee3671e934c1c58db4)
   这就是 《Learning to Generate Chairs with CNNs》论文，作者希望通过给定的视点和对象类，生成三维物体投影。给出的方法是：1.将对象类和视点参数，通过全连接的隐藏层，折叠成一个特征向量；2.通过一个反卷积层，将特征向量展开成生成的图像。   
**缺点**：该模型在生成三维物体图像的任务中表现出良好的效果，并显示出以类和角度插值对象的能力。但是这个模型产生的==图像相当模糊==。
2. Normalization for GANs  
**Instance normalization**    每批独立批处理的实例中对标准化的统计量进行了评估。这种归一化通常用于处理图像。在风格转移问题上表现出了较高的效率   
**Weight normalization**   通过理论计算的均值进行归一化  
==so we can normalize the data without using the statistics of the batch== （why？）
3. Generative Adversarial Nets   
GAN的相关介绍 略
## Approach and Model Architecture
1. generator for absolutely conditional synthesis  
这个生成器是在Dosovitskiy生成器的基础上的改进：  
a. 在图像的反卷积之前，我们使用instance normalization   
b.在每一层反卷积之后，我们使用weight normalization   
c. 最后我们将tanh用于rgb image  sigmoid 用于mask
![image](https://note.youdao.com/yws/api/personal/file/A317C97574E84D75AB105B4A2E599B94?method=download&shareKey=a8a73aa5cbff1fbde00dad808f62e034)

2. generator for partially conditional synthesis  
添加了随机变量Z：特定标签的各种图像 作为生成器的输入  
我们发现，在有变量z的情况下，当我们reshape之后，用==concatenate internal== 表示更有效（就是concat()函数）  
![image](https://note.youdao.com/yws/api/personal/file/3479A9FB78DB41A6A1BABA79F4FEBBE4?method=download&shareKey=2627cc735b0170b4a7cf83163eac05da)
3. discriminator  
![image](https://note.youdao.com/yws/api/personal/file/3151DF74B46043C68F46D651ED56F011?method=download&shareKey=e9241406fba5f9c319b3c824b28087ff)