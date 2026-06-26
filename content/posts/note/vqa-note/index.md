---
title: "视频质量评估笔记"
date: 2026-06-12T16:00:53+08:00
draft: false
categories: ["笔记"]
tags: ["视频质量评估"]
lightgallery: true
---

### 视频质量评估笔记

### 初学乍练

选择了两个小短视频进行入门![](https://raw.githubusercontent.com/Evilcurls/image4blog/main/20260612025116086.png)

#### 传知播客-视频质量评价SimpleVQA

介绍UGC视频的意义，以ACM2023的论文：《A Deep Learning based No-reference Quality Assessment Model

for UGC Videos》为切入点开始介绍：在广义上$$Q = f(C, S, D)$$，其中**$Q$ (Quality)**: 视频质量，**$C$ (Content)**: 视频内容（或内容丰富度/复杂度），**$S$ (Stability)**: 视频稳定性，**$D$​ (Distortion)**: 视频失真程度，视频质量由这些所决定

该论文针对时间-空间进行建模 完成评价，突出特点在于对于UCG视频，传统的视频评估中叫做有参考，也就是把压缩到流媒体上的视频和原视频进行比对，但是UCG视频天生就不存在完美的原版，所以引入了无参考的方式进行评价。

![](https://raw.githubusercontent.com/Evilcurls/image4blog/main/20260612103106706.png)

方法如下：

先将输入的视频进行切块，把切块的视频片段分为，1.每秒的第一帧作为关键帧，然后一共8张，输入到空间流，2.整个视频的所有帧，但是要进行压缩，输入到时间流

在时间流方面，判断的是失真 所以需要对相邻的多帧进行判断，对分辨率不敏感，所以选用了视频的所有帧作为输入，使用3D CNN获得了时间上的特征向量

在空间流方面，判断的是单帧画面中是否出现光点/伪影/噪声/模糊问题，对时间不敏感，所以变成了八张关键帧图片进行判断（每秒取第一帧），使用Resnet50，获得了空间上的特征向量

将两向量拼接在一起，输入MLP进行评分，获得了单视频块的得分，将多个视频块进行时间平均池化，得到了最终的质量得分

作者为拓展工作量，针对不同视频在多分辨率的屏幕下的得分也进行了考量，（比如在4k分辨率下看1080的视频就觉得一般，在1080p分辨率下看1080就觉得还可以，距离是人要是离屏幕太远就只看低频信息）作者限定了距离和尺寸，这样不同分辨率的视频在当前情况下能看到的最高频率就可以计算，然后比如540p 720p 1080p，能看到的最高频率分别是a1 a2 a3，那么将这些频率带入，引入心理学的CSF函数积分，获得了权重比例，将这些权重比例用于几何加权获得最终质量评分

#### 小红书REDtech来了 | 无参考视频质量评估算法研发及落地实践

不看  基于业务的太多了 我日了

### 初窥门径

选择看两篇综述类论文《VIDEOSCORE: Building Automatic Metrics to Simulate Fine-grained HumanFeedback for Video Generation》 对于AI生成视频，使用自动化指标完成评估  

《EvalCrafter: Benchmarking and Evaluating Large Video Generation Models》
