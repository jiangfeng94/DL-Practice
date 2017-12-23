## SGD
SGD全名stochastic gradient descent，即随机梯度下降。
```
optimizer =tf.train.GradientDescentOptimizer(learning_rate =self.learning_rate)
```
具体的实现：  
1. 从训练集中随机抽取一批容量为m的样本，以及相关的输出y
2. 计算梯度和误差 并更新参数   
     `$\hat{g} \leftarrow +\frac{1}{m}\nabla \theta \sum\limits_{i} L(f(x_i;\theta),y_i) $`   
    `$ \theta \leftarrow \theta - \epsilon \hat{g} $`

优点：  
训练速度快，对于很大的数据集也有很快的收敛速度  
缺点：
由于是抽取方式，使得得到的梯度有一定的误差，因此学习率需要逐渐的减小，不然无法收敛。  
解决方法：在实际操纵中使得学习率线性衰减：  
`$ \epsilon_k = (1-\alpha)\epsilon_0 +\alpha\epsilon_\gamma $`  
`$ \alpha = \frac{k}{\gamma}$`

## Momentum
```
tf.train.MonmentumOptimizer(learning_rate)
```
当前权值会受到上一个权值的影响。  
`$ v  =mu*v-learning\_rate *dx $`  
`$ x + = v$`
## NAG
Nesterov Accelerated gradient 与momentum相比，NAG知道自己的目标，它知道自己的下一个目标的大致位置，然后提前计算下一个位置的梯度。  
`$ v\_prev =v$`  
`$ v =mu * v -learning\_rate *dx$`  
`$ x+=-mu*v\_prev +(1+mu)*v$`
## Adagrad
核心思想：对常见的数据给予小的学习率去调整，而对不常见的数据给予大的学习率

## Adam 
把之前衰减的梯度和梯度平方保存起来，使用RMSprob，Adadelta相似的方法更新数据。
```
tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,name ='Adam')
```
learning_rate :学习率  
beta1:一阶段的衰减因子  
beta2:二阶段的衰减因子  
epsilon ：大于但接近0的数，放在分母上避免除0  
`$ m =beta1 * m  +(1-beta1)*dx$`  
`$ v=beta2 * v + (1-beta2)*(dx**2)$`  
`$ m_b =m/(1-beta1**t)$`  #t=step number      
`$ v_b =v/(1-beta2**t)$`  
`$ x += -\frac{learning\_rate *m_b }{\sqrt {v_b}+espilon}$`
