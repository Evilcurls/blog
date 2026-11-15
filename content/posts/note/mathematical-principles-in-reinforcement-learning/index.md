---
title: "mathematical Principles in Reinforcement Learning"
date: 2026-05-30T16:24:25+08:00
draft: false
categories: ["笔记"]
tags: ["强化学习"]
lightgallery: true
---

### 强化学习小记

> [!note]
>
>  我当时记录这个的原因是我对于RL这个过程再到更新梯度这一块有些不了解

一般来说，模型训练的过程就是  前向传播，计算Loss，反向传播，计算梯度，其中梯度使用符号

$\nabla$   \(\\nabla）来表示 

然后参数通过 $w_{new}=w_{old}-\eta * \frac{\partial loss}{\partial a}
$ 其中a是参数来完成更新，\\eta是学习率

然后在强化学习中，我们的目标是最大化 $J(\theta)= \sum_y\pi_{\theta}(y)Reward(y)$，其中该式子与外在的黑盒Reward有关，不便计算，所以我们要求梯度的话，就要考虑把Reward移除式子

存在微积分公式   $\nabla f(x）=f(x)\nabla log(f(x))$

所以 $\nabla J(\theta) = Reward(y) \sum_y log{\pi_{\theta}(y)} $



因为感觉自己对llm训练这一块不熟悉，又重新看了一下

无论是什么模型训练，都是遵从的

前向传播Forward  \-\-\-\-\-\-\>得到Loss\-\-\-\-\-\-\-\> 反向传播 Backward\-\-\-\-\-\-\-\>梯度回传

这个梯度就是根据Loss上升的判断量，如果梯度大于0意味着参数增大 loss上升，所以都是沿着梯度的反方向走来降低loss也就是  $w_{new}=w_{old}-\eta * \frac{\partial loss}{\partial w}
$

我之前在思考两个问题

> 1\.如果梯度是要考虑每一个参数，那么一个7b的模型不是要计算70次 因为要把其他的参数固定住
>
> 

答：通过链式法则，你从后往前一步步算，比如 $\frac {\partial L} {\partial w1}= \frac {\partial L} {\partial w3} \frac {\partial w3} {\partial w2}\frac {\partial w2} {\partial w1}
$  然后其中的 $\frac 
{\partial L} {\partial w3} 
$ 是第三层需要的梯度，然后 $\frac {\partial L} {\partial w2}= \frac {\partial L} {\partial w3} \frac {\partial w3} {\partial w2}$ 是第二层需要的梯度，所以这样一步步算过来，准确来说可以复用中间结果，

> 2\.是怎么更新梯度的呢，是一起的吗，如果是分开的那岂不是要算一次梯度要再更新一次，重复n次
>
> 

答：是一块更新的，然后由于偏导的性质，每一次更新的就只需要考虑自己的梯度就可以了，是准备一个巨大的张量，形状与模型要更新的参数一致，然后每一个参数开始运行 $w_{new}=w_{old}-\eta * \frac{\partial loss}{\partial w}
$

> 3\.LLM是存在采样这一步，似乎无法回传参数
>
> 

答：采样是采样，但是采样可以考虑概率，而概率在交叉熵等损失函数中变成了Loss，这个loss在logits计算的时候开始回传计算

### 全流程：如何用不可微的Reward(y)完成梯度回传

#### 强化学习的优化目标

对于一个输入 $x$，LLM 生成一条完整回答：

$y=(y_1,y_2,\ldots,y_T)
$

其中，整条回答的生成概率为：

$\pi_\theta(y\mid x)= \prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}) 
$

由于 Token 概率需要连乘，因此对于包含上千个 Token 的回答，其生成概率通常非常小。

实际计算时使用 Log Probability，将连乘变为累加：

$\boxed{ \log\pi_\theta(y\mid x) = \sum_{t=1}^{T} \log\pi_\theta(y_t\mid x,y_{<t}) }$

强化学习希望最大化所有可能回答的期望 Reward：

$\boxed{ J(\theta)=\sum_y\pi_\theta(y\mid x)R(y) }  $

其中，$\sum_y$ 表示所有可能的完整回答，而不是对一条回答中的 Token 求和。

#### 为什么不能直接使用负 Reward 作为 Loss？

理论上，我们希望最小化：

$L_{\text{ideal}}(\theta)=-J(\theta) 
$ 

但存在两个问题：

- 不可能枚举 LLM 所有可能的回答，直接计算完整的 $J(\theta)$。

- Reward 可能来自 CER、人工打分或其他黑盒评估器，不能直接通过采样过程反向传播。

因此，我们需要构造一个可以通过有限采样计算梯度的代理 Loss。

注意：对于固定的一条回答，$-\pi_\theta(y)R(y)$ 在数学上可以对 $\theta$ 求导，但它不等于完整的 $-J(\theta)$，也不是通常使用的采样策略梯度损失。

#### Log\-Derivative Trick 如何解决这个问题？

从目标函数出发：

$J(\theta)=\sum_y\pi_\theta(y)R(y) 
$

对参数求梯度（省略固定输入 $x$）：

$\nabla_\theta J(\theta) = \sum_y R(y)\nabla_\theta\pi_\theta(y)
$

这里假设对于固定的 $y$，Reward 不直接依赖模型参数 $\theta$，因此无需计算 Reward 的梯度。

利用恒等式：

$\nabla_\theta\pi_\theta(y) = \pi_\theta(y)\nabla_\theta\log\pi_\theta(y)$

得到：

$ \boxed{ \nabla_\theta J(\theta) = \sum_y\pi_\theta(y)R(y) \nabla_\theta\log\pi_\theta(y) }
$

根据期望的定义：

$\boxed{ \nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta} \left[ R(y)\nabla_\theta\log\pi_\theta(y) \right ]}$

这一步的意义：无需对 Reward 求导，只需要 Reward 的数值，以及模型对生成结果的 Log Probability 的梯度。



#### 如何变成实际训练的 Loss？

因为无法枚举所有回答，所以从模型中采样 $G$ 条回答：

$y_1,\ldots,y_G\sim\pi_\theta$  利用蒙特卡洛方法估计：   $ \nabla_\theta J(\theta) \approx \frac1G\sum_{i=1}^{G} R(y_i)\nabla_\theta\log\pi_\theta(y_i) $

因此，可以构造代理损失：

$\boxed{ L_{\text{policy}} = -\frac1G\sum_{i=1}^{G} R(y_i)\log\pi_\theta(y_i) } $

这里采样结果与 Reward 在本次反向传播中被视为固定值。

又由于：

$\log\pi_\theta(y_i) = \sum_t\log\pi_\theta(y_{i,t}\mid y_{i,<t},x) $

所以实际训练可以通过各 Token 的 Log Probability 计算 Loss，然后执行：

$\text{Loss}\rightarrow\text{Backward}\rightarrow \text{Optimizer Step}$

策略梯度训练流程

LLM 采样 G 条回答

黑盒评估器计算 Reward

计算采样 Token 的 Log Probability

Reward × Log Probability → Policy Loss

Backward → 更新 LLM 参数



#### PPO 和 GRPO 为什么使用 Advantage？

基础策略梯度可以直接使用 Reward，但梯度估计的方差可能较大。

因此，引入 Advantage：

$A(y)=R(y)-b 
$

其中 $b$ 为基线，用来衡量当前结果相对于参考水平的好坏。

在适当条件下，减去不依赖当前采样动作的基线不会改变期望策略梯度，但可以降低方差。

PPO 通常通过价值函数（Critic）估计 Advantage；GRPO 则使用同一 Prompt 下的一组输出的相对 Reward 来计算 Advantage。

例如 GRPO 中常见的形式：

$ A_i= \frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)} {\operatorname{std}(R_1,\ldots,R_G)+\epsilon}$

因此，实际用于策略更新的信号从直接使用 Reward 变为了使用 Advantage。

但还需要注意：PPO 和 GRPO 不只是把 Reward 换成 Advantage。 它们通常还会使用新旧策略概率比值、裁剪等机制，控制策略更新幅度。

#### 用一段话总结

强化学习的目标是最大化模型生成回答的期望 Reward。由于 LLM 的回答由多个 Token 自回归生成，整条回答的概率是各 Token 条件概率的连乘，通常非常小，因此计算时使用 Log Probability，将连乘转化为求和。

对期望 Reward 求梯度时，可以通过 Log\-Derivative Trick，将问题转换为利用 Reward 加权模型输出的 Log Probability 梯度，无需对黑盒 Reward 或离散采样过程求导。实际训练通过有限 Rollout 估计梯度，并构造可微的 Policy Loss，使用反向传播更新模型参数。

PPO 和 GRPO 在基础策略梯度上引入 Advantage，并利用概率比值和裁剪等机制提高训练稳定性。Reward 并没有被丢弃，而是用于计算 Advantage，进而影响模型更新。

> [!IMPORTANT]
>
>  “Log Probability 替代了 Reward”，而是“Reward（或 Advantage）决定梯度的权重和方向，Log Probability 提供可以对 LLM 参数求导的路径”。 

### 关于课程：《强化学习中的数学原理》

研究中的焦虑大部分来自于对于时间速成的胡乱分配，说一天看完论文，可能一天Intro都看不完，但是稳扎稳打可能两三天就看完了。

Time:course 10h  其他基础和编程加起来大概40小时  



### Lecture I  basic concept in reinforcement learning

1.state:status of agent with respect to  environment

2.state space : set of state （空间是定义了规则的集合）

3.Action：
