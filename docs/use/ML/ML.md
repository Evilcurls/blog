# Machine Learning

# 基础

## 目标

 机器学习的目标就是拟合一个函数能够完成预测问题

三要素：模型 学习准则 优化算法

学习算法一般分为三种

​	监督学习

​		回归问题	标签y是连续值	f(x,θ)输出也是连续值

​		分类问题	y是离散值

​		结构化学习问题	实质是特殊分类问题 标签Y通常是结构化的对象 序列 /树/图之类的

​	

​	无监督学习:从不包含目标标签的训练样本中自动学到有价值的信息

​		聚类 	密度估计	特征学习 	降维

​	强化学习：通过交互来学习的机器学习算法 	在和环境的交互中不断学习并且调整策略

### 特征表示

​	首先机器学习核心应该是数学运算，所以要把各种特征转化为向量来表示，方便计算

​		图像：M*N的图像 变为 M * N维的向量，每维的数值是灰度，如果是彩色图就RGB三层叠加

​		文本： 简单方式是用词袋（Bag of words)但好像由于不考虑词序，不能精确表示文本信息

​		特征学习 又称表示学习   让机器自动学习出有效特征

​				方式：用人为设计准则  通过准则选出需要的特征

### **梯度消失（Vanishing Gradient）**

#### **定义**

在反向传播时，梯度值随着网络层数增加呈指数级衰减，导致浅层（靠近输入层）的权重几乎不更新，训练停滞



 

**梯度爆炸（Exploding Gradient）**

#### **定义**

在反向传播时，梯度值随着网络层数增加呈指数级增长，导致权重更新幅度过大，模型参数发散（NaN 或溢出）。

## 数据增广

扩充数据集：比如对已有图片加入噪声 镜像旋转 

​			可以遇到更加多样化的情况提升降低过拟合风险

## 微调

通过大数据集训练出来的模型用在小数据集上

因为大的数据集训练出来的初始模型泛化能力一般会强一点，所以用来初始化参数（反正肯定比随机设参数强）

那么在复制参数的时候，是不是一模一样复制呢：在神经

## 两条定律

没有免费午餐定律：炉石传说没有完爆

奥卡姆剃刀原理：简单的模型泛化能力更强	没必要不要增实体

# 各种问题

### 一元

predict	

![0](一元模型三要素.png)

​	在计算损失函数的时候，loss=yi-(wxi+b)，由于只是把标签值yi和输入值xi带入了，所以loss是关于两个参数的方程，可以计算最小值梯度等问题，当使用线性模型的时候会出现离群点，损失函数可以换成别的函数均方误差受到的影响太大

### 二分类问题

​	样本空间中找到一个超平面 它将特征空间一分为二 一半是我愿意 一半是我拒

![sigmoid函数](sigmoid函数.png)

​	在分类问题中不能用线性模型的输出，或者说不能直接用，因为一般线性模型的输出是连续的，所以要将线性模型的输出使用sigmoid函数转化为概率值	 把一整段连续值挤压到空间[0,1]，赋予概率值，0或1哪个超过0.5就分类结果是哪个

sigmoid函数=$ \frac{1} {1+e^{-x}} $

 ![二分类问题的步骤](二分类问题.png)

​	像这样的思想可以推广到多分类问题，就像之前那个手写体数字识别，就是用softmax激活函数完成的多分类问题

​	其中softmax=$\frac {e^x_i} {\sum_i {e^{xi}}}$

### SVM 支持向量机

模型＆目的：找一个超平面把标记了正负的样本空间里的样本给一分为二   就是之前的二分类问题，有一个名词叫支持向量，表示距离超平面最近的点，一共有两个，超平面移动到这两个点形成的平面分别叫正负超平面

如果一次性将所有的点都分开，那么叫做硬间隔，但尽善尽美的事情很少，往往最优的超平面会分出绝大多数的点，但是有那么一两个

点会分隔异常，所以此时把这些异常的点用损失因子记录，通过考虑损失因子和间隔的收益，这种方法叫做软间隔

学习准则：最大化间隔    间隔为2/|w|		所以是argmin|w|^2

优化策略：凸二次优化

问题：超平面又支持向量决定，支持向量又很少，导致解具有稀疏性

## 神经网络

​	神经元是最小单位，有轴突和树突，树突用于获取信息，轴突用于发送信息，神经元只有两种状态 ，激活和抑制

​	激活函数在模拟神经元的触发机制，当满足某种条件变成激活状态，

### sigmoid函数

​	长得像S型的函数  (两端饱和函数)一般有logstic 和 tanh

​	logstic函数的长这样   $\frac 1 {1+exp(-x)}$

​	tanh函数长这样  $\frac {exp(x)-exp(-x)} {exp(x)+exp(-x)}$





![ ](logistic&tanh.png)

logstic函数优势在于简单，劣势在于没有是非零中心化的，会影响后来的神经元，有偏置偏移

### Relu函数

长相：$f(x)=\begin{cases}1     x\ge 0 ，0      x<0 \end{cases}$

优点：计算高效，兴奋程度高

缺点：ReLU函数的输出是非零中心化的，给后一层的神经网络引入偏置偏移， 会影响梯度下降的效率． 此外，ReLU神经元在训练时比较容易“死亡”．在训 练时，如果参数在一次不恰当的更新后，第一个隐藏层中的某个ReLU神经元在 所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是 0， 在以后的训练过程中永远不能被激活．这种现象称为死亡ReLU问题

有好几种变种去解决这些问题：泄露的relu(leaky relu) 带参数的relu(parametric Relu) 还有好多什么ELU，什么GELU

### 网络结构

### 前馈神经网络

就是拟合一个函数，有向无环，单向传播

![](前馈神经网络.png)

上图示例就是将输入x输入到神经网络$\phi (x)$ 中去，分类器的参数是一堆$\theta$ ,在通过比如softmax分类器 对$\phi (x)$的输出数据分类

### 卷积神经网络

假设我直接将一个100*100的图片输入到全连接网络

假设有N个神经元，那么在后续的全连接层中1.输入参数过多不好计算 2.会丢失某些图像特征

所以卷积应运而生：

1.部分连接：

​     在全连接层中，第二层的任意神经元都要连接前面所有的神经元，那么就需（要维数*N+维数）个参数，但在卷积神经网络中，第二层中的所有神经元都各司其职，只与前一层的部分神经元连接

2.权重共享：

​	虽然部分连接决定了某些神经元去处理图像的不同区域，但是比如人脸识别这样的工作，人脸可能在上下左右的任意地方出现，所以可以用一套参数完成任务

![](卷积作用图.png)

## 期中作业 Dog-breed-identifacation

1.读取数据集

2.数据增广

### 加载别人现有的模型

`class Net(torch.nn.module）`继承了torch.nn.module 公式写法

`super().__init__() `使用pytorch设定好的连接方式

`self.mymode=models.resnet50(pretrained=False)` 

self.xx是实例化指自己，有点像c++里面的this指针   self.model是新增一个类里面的一个属性，models是torchvision里面的一个包，models.resnet50是加载了resnet50这个网络的骨架，pretrained=False表示不用加载预训练数据,因为我本地有resnet50的参数，如果选了True就要从网上开始下了

`    self.mymode.load_state_dict(torch.load('resnet50-19c8e357.pth'))` 

​				**`state_dict`** ：

​					在 PyTorch 中，每个模型都有一个 `state_dict`，它是一个 Python 字典，存储了模型的所有可学习参数（如权						重和偏置）。

​					例如，对于一个简单的线性层 `nn.Linear(10, 2)`，其 `state_dict` 可能包含两个键值对：`'weight'` 和 						`'bias'`。

​				**`torch.load`** ：

​						`torch.load` 是 PyTorch 提供的函数，用于从文件中加载保存的模型参数（通常是 `.pth` 或 `.pt` 文件）。

​				**`load_state_dict`** ：

​					这个方法将加载的参数字典应用到当前模型中，从而恢复模型的状态。

### 构建数据集

`class dog_dataset(torch.utils.data.Dataset):`**为什么需要继承 `Dataset`？**

​			PyTorch 的 `DataLoader` 是一个强大的工具，用于高效地加载和批量处理数据。`DataLoader` 的工作依赖于 `Dataset` 类提				供的接口。具体来说：

​				**`Dataset`** ：

​							负责定义如何加载和访问数据。

​					提供了一个统一的接口，使得数据可以被 `DataLoader` 使用。

​				**`	DataLoader`** ：

​					负责将数据分批次加载到模型中。

​					支持多线程、数据打乱、并行处理等功能。 

​					有参数num_worker的时候，在windows下会报错

```   def __init__(self,csvfile,imgdir):
    def __init__(self,csvfile,imgdir):
        self.result_csv=pd.read_csv(csvfile)
        self.imgdir=imgdir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图片大小
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
                ])
```

 		`pd.read_csv`:用于读取csv 表格文件

​		 **为什么需要定义这个 `transform`？**

​			在深度学习中，尤其是卷积神经网络（CNN）中，输入数据通常需要满足以下要求：

​			**尺寸一致** ：所有图片的尺寸必须相同，以便可以堆叠成一个批次（batch）。

​			**数值范围一致** ：图片像素值通常被归一化到一个特定的范围（例如 `[0, 1]` 或均值为 0、标准差为 1 的分布），以加速模				型训练并提高收敛性。

​			**格式统一** ：图片需要转换为张量（Tensor）格式，才能被 PyTorch 模型处理。

 		`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化`  

​			其中的mean和标准差是通过ImageNet计算而来的  我感觉狗种类识别可以从他训练集里面算			  

##



`model.train()` 是 PyTorch 中的一个方法，用于将模型设置为**训练模式**

验证集的主要作用是评估模型在**未见过的数据** 上的性能



假设 `batch_size = 32`，我们来看一个具体的例子：

#### 输入数据：

- `input`: 形状为 `[32, input_dim]` 的张量，表示 32 张图片。
- `label`: 形状为 `[32]` 的张量，表示这 32 张图片的真实类别标签。

#### 前向传播：

- 模型输出 `output` 的形状为 `[32, num_classes]`，表示每张图片在每个类别上的预测分数。
- 使用 `_, predicted = output.max(1)` 得到形状为 `[32]` 的张量 `predicted`，表示每张图片的预测类别。

#### 计算正确数：

- 假设 `predicted = [0, 1, 2, ..., 0]`（长度为 32），`label = [0, 1, 3, ..., 0]`。
- `(predicted == label)` 会生成一个布尔张量 `[True, True, False, ..., True]`。
- `(predicted == label).sum()` 计算为 28，表示当前 batch 中有 28 张图片预测正确。

#### 计算损失：

- 假设损失函数为交叉熵，计算得到 `loss = 0.5`。
- `loss.item()` 提取标量值 0.5，并将其累加到 `loss_total` 中。



## 注意力机制

一般分为两种：

​	聚焦式注意力，要去完成某种特定任务：比如在人群中寻找某个人就开始注意脸部，统计人数时就注意轮廓

​	显著式的注意力：比如在正常任务时，被某种刺激所导致的开始聚焦  （上课突然有人开始狗叫 大家都看向狗叫的人）

对于这两种注意力 也就催生出了两种自注意力的实现方式 ，主要集中在Q值（query）的来源上

​	当q是我们设定（即外部提供的） 就是我们想要聚焦某种特征，比如在seq2seq中 解码器通过一个注意力分布来选择性的观察编码器生成的隐藏状态

```
假设我们有一个翻译任务，目标是将英语句子翻译成法语。在解码阶段：

解码器生成一个查询向量 q，表示当前需要翻译的单词。
编码器生成一组键值对 (k,v)，表示源语言句子中的每个单词。
解码器通过计算 q 和 k 的相似性来决定应该关注源语言句子中的哪些单词。
```

​	当显著式也就是自动生成q时：比如transformer就会计算元素与元素之间的相关性，可以动态决定应该关注哪些部分。特点就是qkv都是根据输入数据生成的（比如线性变换），

``` 
在 Transformer 模型中：

输入序列中的每个 token 会通过线性变换生成自己的 q、k 和 v。
每个 token 的 q 会与其他所有 token 的 k 计算相似性，得到注意力权重。
最终，这些权重用于加权求和 v，生成新的表示。
```





# 传统机器学习模型补漏

## 随机森林

随机森林的设计过程主要分为三步

### 设定超参数

使用几颗决策树？每个决策树分几层？

### 决定使用的数据

对于总体数据 N个样本 每个样本D个特征

每棵树 选用nxd  其中n<<N  d<<D

### 获得结果 

Regression选用平均值

classification选用多数投票



Day 1 
回顾线性回归和逻辑回归的工作原理（L1/L2 正则化概念即可）。重点理解它们如何处理特征的权重。
导入项目数据（如 Kaggle 的房屋数据），使用 Pandas 进行初步探索（.info(), .describe()）。	完成数据读取，识别缺失值和类别特征。
使用 Scikit-learn 实现 LinearRegression 或 LogisticRegression（根据项目是回归或分类）。

## 使用sklearn这个包 来完成线性回归

!jupyter nbconvert --to markdown week1.ipynb


```python
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter
```


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# 1. 生成模拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100个样本，1个特征
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5  # y = 4 + 3x + 噪声
```


```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print("Sklearn 参数：")
print(f"截距: {model.intercept_[0]:.3f}")
print(f"斜率: {model.coef_[0][0]:.3f}")
```

## 以kaggle上经典数据集 House price predict 来练习


```python
import pandas as pd 
train_csv_path="houseprice/train.csv"
df=pd.read_csv(train_csv_path)
df.info()      #诊断工具，看看数据的大致情况 有几行 几列分别是什么有没有缺省值 数据类型
df.describe()  #最大最小平均数等各种情况

missing=df.isnull().sum()   # 返回一个pd.Serial类型的数据 第一列是原来的列名  第二行是该列总缺省值 /不加sum()是原表格显示每一格是否缺省
missing[missing>0]    

objectclass=df.select_dtypes(include=['object']).columns  #筛选数据

Bsm=df[['BsmtFinSF1']]
houseprice=df['SalePrice']
model=LinearRegression()
model.fit(Bsm,houseprice)
print("Sklearn 参数：")
print(f"截距: {model.intercept_:.3f}")
print(f"斜率: {model.coef_[0]:.3f}")

```

9月17日 (周二)
使用 Matplotlib/Seaborn 绘制特征分布图、特征与目标变量的关系图（如散点图）。识别异常值和强相关特征。	发现至少 3 个有价值的数据洞察。
实现至少 3 种特征工程技巧：数值特征对数转换（处理偏态）、类别特征 One-Hot 编码、创建交互特征（如面积 * 楼层）。
将数据加载和清洗代码封装成函数或类，提高代码可读性和复用性。	代码结构化，符合工程规范。


```python
import matplotlib.pyplot as plt
import seaborn as sns
matrix=df.pivot_table(index='Neighborhood',columns='LandSlope',values='SalePrice',aggfunc='mean')
sns.scatterplot(data=df,x='1stFlrSF',y='GrLivArea')
plt.show()
sns.boxplot(data=df)
plt.show()
sns.histplot(data=df,x='SalePrice')
plt.show()
sns.heatmap(matrix)
plt.show()
```

进行独热编码 


```python
one_hot_encoded=pd.get_dummies(df['BldgType'],prefix='')  # 用传统方式表示平级的类别，比如1，2，3，4会使得机器学到不应该的关系，比如红蓝黄设置为123，那么会学到黄大于蓝
print(one_hot_encoded)

#创建交互特征
df['new_feature']=df['GrLivArea']*df['1stFlrSF']
print(df['new_feature'])
```


```python
class data_prepare():
    def __init__(self,csv_path,save_percent):
        self.csv_path=csv_path
        self.df=pd.read_csv(self.csv_path)
        self.threshold=len(self.df)*save_percent
        print(f"数据原形状:{df.shape}")
        self.df_clean=df.dropna(thresh=self.threshold,axis=1)
        print(f"数据现形状{self.df_clean.shape},缺省率超过{save_percent*100}的数据已经被删除")
train_csv_path="houseprice/train.csv"
price='SalePrice'
a=data_prepare(train_csv_path,0.05)
```

快速理解决策树（CART）的分裂准则（如 Gini/Entropy）。
重点掌握 Bagging (Bootstrap Aggregating) 思想，即随机森林如何通过“多数投票”或“平均”来降低过拟合。
使用 Scikit-learn 的 RandomForestRegressor 或 RandomForestClassifier 训练模型。
掌握 K-Fold Cross-Validation 的必要性。   #没做 就是
使用 cross_val_score 或 KFold 手动实现交叉验证，评估随机森林的稳定性能。


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd

train_csv_path="houseprice/train.csv"
houseprice=data_prepare(train_csv_path,0.01) 
X=houseprice.df_clean.drop("SalePrice",axis=1)
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded,houseprice.df_clean["SalePrice"], test_size=0.2, random_state=42)
forest_model=RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train,y_train)
error=forest_model.score(X_test,y_test)
print(f"结果是{error}")


```


```python
from sklearn.model_selection import cross_val_score
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用 cross_val_score 执行 5-折交叉验证
# estimator: 你要评估的模型
# X, y: 完整的特征数据和标签
# cv: 折数 (k的值)
# scoring: 评估指标，例如 'accuracy' (分类), 'neg_mean_squared_error' (回归)


scores = cross_val_score(rf_model, X_encoded, houseprice.df_clean['SalePrice'], cv=5, scoring='r2')

# 打印每次验证的结果
print(f"每次: {scores}")

# 打印平均分和标准差，以评估模型的稳定性能
print(f"平均r2: {np.mean(scores):.4f}")
print(f"r2标准差: {np.std(scores):.4f}")
```

## ·对于GBDT的一些梳理

要从GBDT开始说起，他是一个不确定的函数，可能是十颗树 一百颗树，这跟我们深度学习里面设定了模型结构然后调整参数不太一样
其中的B指的是boost，比起随机森林这种bagging采用多数投票的并行树的方式，boosting采用的是串行树，也就是第n颗树的所预测的目标由前n-1颗树来决定，去预测负梯度
其中的G是指Gradient，梯度，作者是希望有一个统一的框架来完成一个目标，那就是靠一种方法把分类，回归等任务来完成于是使用负梯度来完成这一步，后一棵树是拟合的是函数梯度下降的方向，如果损失函数恰好是MSE，那么残差就是负梯度
新增决策树的过程是采用贪心算法，以第N颗树设立距离，前N-1颗树为函数F_N-1 用标签y和F_N-1把样本都过一遍，用损失函数得出负梯度，然后负梯度和样本x_i成为一个新的数据集，新的树再拟合这个数据集

## day4

DMatrix是XGboost团队专门开发的数据类型，在XGboost上表现会好一点
#使用houseprice预测遇到的问题：
XGBoost 默认不支持 object（字符串）类型的分类变量，直接传入会报错： 
把 object 列转成 category 类型：
for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
创建 DMatrix 时开启分类支持：
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)




```python
import  xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


class data_prepare():
    def __init__(self,csv_path,save_percent):
        self.csv_path=csv_path
        self.df=pd.read_csv(self.csv_path)
        self.threshold=len(self.df)*save_percent
        print(f"数据原形状:{self.df.shape}")
        self.df_clean=self.df.dropna(thresh=self.threshold,axis=1)
        print(f"数据现形状{self.df_clean.shape},缺省率超过{save_percent*100}的数据已经被删除")

        
train_csv_path="houseprice/train.csv"
houseprice=data_prepare(train_csv_path,0.01) 
X=houseprice.df_clean.drop("SalePrice",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,houseprice.df_clean["SalePrice"], test_size=0.2, random_state=42)

# 5. 把 object 列转成 category（关键！）
cat_cols = X_train.select_dtypes(include='object').columns
print(cat_cols)
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

xg_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
xg_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
params = {
    'objective': 'reg:squarederror',  # 回归任务：最小化平方误差
    'eval_metric': 'rmse',            # 评估指标：均方根误差
    'seed': 42
}
model=xgb.train(
    params=params,
    dtrain=xg_train,
    num_boost_round=100,           # 树的数量（默认100）
    evals=[(xg_test, 'eval')],         # 用测试集验证
    verbose_eval=10                  # 每10轮打印一次进度
)
y_result=model.predict(xg_test)
print(y_result)
```

## Day 5

LightGBM   也是一套用GBDT框架做的模型  用了一些小技巧加速，比如bin-way split /GOSS(单边梯度采样)/Exclusive feature bunding   /leaf-wise等等方法来保证效率
导入 lightgbm 库，并在同一数据集上运行 LGBM 模型,并能够解释 LightGBM 为什么比 XGBoost 快（基于直方图的决策树算法）。

2. 学习使用 Scikit-learn 的 GridSearchCV 或 RandomizedSearchCV 对 XGBoost 模型进行简单超参数调优。重点关注 max_depth 和 colsample_bytree。掌握模型调优的基本方法，找到一组优于默认参数的配置。
   3.提取 XGBoost 的特征重要性分数。绘制柱状图，分析哪些特征对模型预测的贡献最大。识别出对项目结果影响最大的 Top 5 特征，并尝试解释其合理性。


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

```


```python
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
gbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,            # 由 num_leaves 控制复杂度
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

gbm_model.fit(X_train,y_train)

predict=gbm_model.predict(X_test)
acc=accuracy_score(y_test,predict)
print(acc)
```

很多超参数是需要设定来寻找最优值的，所以使用RandomSearchCV来寻找最优参数，不必手动调参,cv是crossvalidation,交叉验证，你想 可能你这次训练的好只是因为你测试集走狗运划分的好，不能说明一个稳定的情况，五折交叉验证，就是说你将把训练集分成五份，然后每份轮流当测试集，其他四份当训练集，当然不会让测试集泄露，这样可以保证结果是稳定的


```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
gbm_Base=LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',  
    eval_metric='logloss'
)

param_dist = {  #设定参数范围
    'max_depth': randint(2, 10),  #从2-9的整数   
    'colsample_bytree': uniform(0.4, 0.6), #均匀采样
}

search=RandomizedSearchCV(
    estimator=gbm_Base,  #使用哪个基准模型，在上面定义了的
    param_distributions=param_dist,
    n_iter=20,   # 试多少组
    scoring='roc_auc',
    n_jobs=-1,
    cv=5,   #采用几折的方式
    random_state=42,
    verbose=1
)
search.fit(X_train,y_train)

print('best score',search.best_score_)
print('best param',search.best_params_)

best_xgb = search.best_estimator_  #保存效果最好的模型

from sklearn.metrics import accuracy_score
predict=gbm_model.predict(X_test)
acc=accuracy_score(y_test,predict)
print(acc)
```


```python
#提取 XGBoost 的特征重要性分数。绘制柱状图，分析哪些特征对模型预测的贡献最大。识别出对项目结果影响最大的 Top 5 特征
#先运行一遍day 4的第一个代码块，使用xgboost跑房价分析的

feature_importance=model.get_score(importance_type="gain")

importance_df = pd.DataFrame({
    "特征名": list(feature_importance.keys()),
    "重要性分数": list(feature_importance.values())
}).sort_values(by="重要性分数", ascending=False)

top5_features=importance_df.head(5)


# 接下来是画图

import matplotlib.pyplot as plt


# 创建柱状图
plt.figure(figsize=(10, 6))  # 画布大小（宽10，高6）
bars = plt.bar(
    x=top5_features["特征名"],  # x轴：特征名
    height=top5_features["重要性分数"],  # y轴：重要性分数
    color="#1f77b4",  # 柱子颜色（蓝色）
    alpha=0.8  # 透明度
)

# 美化图表（添加标签、标题、数值）
plt.xlabel("name_of_feature", fontsize=12)
plt.ylabel("Gain", fontsize=12)
plt.title("XGBoost Top5 important feature", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")  # 特征名旋转45度，避免重叠

# 在柱子顶部添加具体数值（保留2位小数）
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,  # x坐标（柱子中心）
        height + max(top5_features["重要性分数"])*0.01,  # y坐标（柱子顶部+1%余量）
        f"{height:.2f}",  # 数值格式
        ha="center", va="bottom", fontsize=10
    )

# 调整布局（避免标签被截断）
plt.tight_layout()

# 显示图片
plt.show()

```



# 使用Hugging face

这是一个使用别人提供的训练好的现成模型的网站，就算是我要修改也只能微调，但是这应该符合我的方向，从头训练一个模型太大了，不如选用别人预训练完的模型，在特定任务上微调。

### 运行

使用pipeline函数把别人上传的模型和封装的方法，使得实例化的对象成为一个可以用的函数

` classifier=pipeline('sentiment-analysis')` 



### Tokenizer

在使用模型时所有这些预处理都需要与模型预训练时的方式完全相同

- 将输入拆分为单词、子单词或符号（如标点符号），称为 **token**（标记）
- 将每个标记（token）映射到一个数字，称为 **input ID**（inputs ID）
- 添加模型需要的其他输入，例如特殊标记（如 `[CLS]` 和 `[SEP]` ）,比如下图Input IDs开头的101 代表进行什么任务	
- - 位置编码：指示每个标记在句子中的位置。
- - 段落标记：区分不同段落的文本。
- - 特殊标记：例如 [CLS] 和 [SEP] 标记，用于标识句子的开头和结尾

这些input IDs 组合而成tensor，这正是Transformer模型需要的输入

![pipeline集成步骤](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline-dark.svg)

将Tensor输入Transformer模型后得到的是高维向量表示，需要使用Head(任务头)来完成具体的任务，得到logits（对数几率）不能直接反应结果，所以需要进行后序处理，使用Softmax函数得到分布，比如

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

具体分数和标签的对应关系可以使用模型的` model.config.id2label ` 查看

Tokenizer是一个大的类，里面有许多小的函数，比如raw text -> token 由`tokenize`完成，token->ids 由`convert_tokens_to_ids`完成，由ids->tensor 由`torch.tensor`完成（这一步如果输入单个句子会报错，因为transformer要求输入多个句子，所以在使用tokenizer一次性完成全部操作时如果输入的是一个句子，他会自动给你升维； 然后tensor是一种形状固定的数据结构，所以如果输入长短不一的句子会自动padding)

但padding什么呢？padding的token会引起什么后果呢？解决方法是什么呢？

tokenizer.pad_token_id。注意力层会因为填充的token不同得到不同的高维向量。注意力掩码层。





### 开始微调

hugging face提供了一个包datasets能下载想要的数据集，比如下载"glue"中"mrpc"任务的数据集`origin_dataset=datasets.load_dataset("glue","mrpc") `，可以print看看结构方便后续处理。

Transformers 提供了一个 `Trainer` 类，可以帮助你在数据集上微调任何预训练模型。

#### 先开始数据预处理

但是如果我要用自己的数据集去微调，但是我数据集的格式只有很小的概率是符合模型的（就比如要编码），所以我们要对数据集进行预处理以达到预期。

 `map()` 方法的工作原理是使用一个函数处理数据集的每个元素。先定义这个函数

`def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True) `

然后在map中使用这个函数一批一批处理数据

`tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) `

众所周知，tensor要求每个句子长度相同，所以我们进行了填充，但是如果将所有句子都填充到数据集最长的那个句子的长度是很浪费空间的，既然输入模型的句子是按batch输入的，所以使用动态填充函数，能将每个batch中的句子自动补全到该batch的最长句子的长度

`data_collator = DataCollatorWithPadding(tokenizer=tokenizer) `

#### 微调

Pytorch是底层训练框架，重新写微调代码还是费劲的，所以Hugging face提供了Trainer函数来完成一键完成。

```
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
