# 机器学习模型实现笔记

**机器学习的一般步骤**

1. 准备数据
2. 选择数学模型
3. 建立损失函数
4. 对损失函数进行优化
5. 模型训练(前馈、反馈、调整超参数)

## 预测模型

### linear regression

**理论推导**

**选择模型**

$y=\omega^T x+b$

**损失函数**
$$
\begin{align}
loss&=(\hat{y}-y)^2
\\
\begin{bmatrix}
loss_1\\
loss_2\\
loss_3\\
\end{bmatrix}
&=(\begin{bmatrix}
\hat{y_1}\\
\hat{y_2}\\
\hat{y_3}\\
\end{bmatrix})
-
(\begin{bmatrix}
{y_1}\\
{y_2}\\
{y_3}\\
\end{bmatrix})^2
\end{align}
$$
### logistic_regression

**理论推导**
$$
\begin{aligned}
\hat{y}&=\sigma(x^T*w)+b \\ \text{其中}\quad
\sigma(x)&=\frac{1}{1+e^{-x}}
\end{aligned}
$$

![Logistic函数](.\assets\format,f_jpg.jpeg)

**损失函数**

​		损失函数需要计算两个概率分布的差距，此处采用交叉熵作为损失函数
$$
\begin{aligned}
loss(p,q)&=-\sum_X^n p(x)\log q(x)
\\ &=-(y\log\hat{y}+(1-y)\log(1-\hat{y}))
\end{aligned}
\text{其中} y,\hat{y}\in\{0,1\}
$$


## 损失函数

​		**KL 散度（Kullback-Leibler Divergence**）：也叫KL 距离或相对熵(Relative Entropy)，**是用概率分布 𝑞 来近似 𝑝 时所造成的信息损失量**．KL 散度是按照概率分布𝑞的最优编码对真实分布为𝑝的信息进行编码，其平均编码长度（即交叉熵）𝐻(𝑝, 𝑞) 和 𝑝 的最优平均编码长度（即熵）𝐻(𝑝) 之间的差异．对于离散概率分布𝑝和𝑞，从𝑞到𝑝的KL散度定义为
$$
\begin{align}
KL(𝑝, 𝑞) & = 𝐻(𝑝, 𝑞) − 𝐻(𝑝)\\
& =\sum_i^np(x_i)\log \frac{p(x_i)}{q(x_i)}
\end{align}
$$

​		**交叉熵**：对于分布为𝑝(𝑥)的随机变量，熵𝐻(𝑝)表示其最优编码长度．交叉熵（Cross Entropy）是按照概率分布𝑞的最优编码对真实分布为𝑝的信息进行编码的长度，定义为
$$
H(p,q)=-\sum_Xp(x)\log q(x)
$$

​		给定$q$，如果$p,q$越接近，交叉熵越小；如果$p,q$越远，交叉熵越大；**由此，交叉熵可用来衡量两个变量分布的差异。**

​		以分类为例，设真实分布$p_r(y|x)$，预测分布$p_\theta(y|x)$，假设$y^*$为$x$的真实标签，其中$p_r(y^*|x)=1,p_r(y|x)=0$ 如何衡量两个分布之间的差异？
$$
\begin{align}
KL(p_r(y|x),p\theta(y|x)) & =\sum_y p_r(y|x)\log\frac{p_r(y|x)}{p_\theta(y|x)}\quad \text{KL散度} \\
& = \alpha-\sum_y p_r(y|x)\log{p_\theta(y|x)}\quad\text{交叉熵} \\
& = -\log p_\theta(y^*|x) \  \text{负对数似然}
\end{align}
$$
​		在机器学习中，KL散度与交叉熵在损失函数中应用广泛。最小化真实分布与预测分布之间的差异=最小化交叉熵=最大化对数似然，从概率论角度可理解为最大似然估计。

## 优化算法

### 梯度下降 Gradient Descent

​		对某一个参数计算梯度，沿梯度下降算法求解极小值,设损失函数$L(\omega_1)$,权重$\omega_1$,则$\omega_1$的更新公式为$\omega_1=\omega_1-\alpha*\nabla L(\omega_1)$,其中$\nabla L(\omega_1)$称为$L$对$\omega_1$的梯度，$\alpha$为学习率。

**批量梯度下降 Batch Descent**

​		批量梯度下降指将每一个样本应用到梯度下降过程中，其计算得到的是一个标准梯度，对于凸优化一定能达到全局最优。但算法时间复杂度很高。

**随机梯度下降**

​		指在所有样本中随机选取一个，进行梯度下降。其效率最高但不能保证得到的参数就是全局最优解，可能陷入到鞍点、局部最优点。

**mini-batch梯度下降**

​		即在前述两个梯度下降算法中折中，随机选取小批量样本参与到更新权重的过程中，在降低时间复杂度的同时尽量得到全局最优解。

**动量梯度下降法 Gradient descent with Momentum**

Todo

## 激活函数

​		激活函数是神经网络中引入非线性因素的重要工具，通过激活函数神经网络就可以拟合各种曲线。激活函数主要分为饱和激活函数（Saturated Neurons）和非饱和函数（One-sided Saturations）。

![图片](D:\code\study\ml\doc\assets\激活函数示例.png)

