

# Day 1 Sklearn包与预处理数据（Info/is_null)

回顾线性回归和逻辑回归的工作原理（L1/L2 正则化概念即可）。重点理解它们如何处理特征的权重。
导入项目数据（如 Kaggle 的房屋数据），使用 Pandas 进行初步探索（.info(), .describe()）。	完成数据读取，识别缺失值和类别特征。
使用 Scikit-learn 实现 LinearRegression 或 LogisticRegression（根据项目是回归或分类）。

## 使用sklearn这个包 来完成线性回归


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


## Day2 特征工程 

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

# Day3 随机森林 与Cv(交叉验证)

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

## 对于GBDT的一些梳理
要从GBDT开始说起，他是一个不确定的函数，可能是十颗树 一百颗树，这跟我们深度学习里面设定了模型结构然后调整参数不太一样
其中的B指的是boost，比起随机森林这种bagging采用多数投票的并行树的方式，boosting采用的是串行树，也就是第n颗树的所预测的目标由前n-1颗树来决定，去预测负梯度
其中的G是指Gradient，梯度，作者是希望有一个统一的框架来完成一个目标，那就是靠一种方法把分类，回归等任务来完成于是使用负梯度来完成这一步，后一棵树是拟合的是函数梯度下降的方向，如果损失函数恰好是MSE，那么残差就是负梯度
新增决策树的过程是采用贪心算法，以第N颗树设立距离，前N-1颗树为函数F_N-1 用标签y和F_N-1把样本都过一遍，用损失函数得出负梯度，然后负梯度和样本x_i成为一个新的数据集，新的树再拟合这个数据集

## Day4  XGboost

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

## Day 5 LightGBM
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
