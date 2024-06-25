#  Blog实现

好记性不如烂笔头，好多碎碎念的东西过了两天就忘得一干二净了属实可惜。

# docsify

## 过程

### 是啥

网页是随着主人的意愿制作而成的如同一栋房屋，html是其中的骨架，而docsify是装修公司，他能按照主人的意愿完成房屋的装饰过程。 其实就是把你的一堆markdown文档渲染成一个网页了啦。

抄了个介绍，优点轻/易/主题多

> 简单来说，`docsify` 可以动态地将 `Markdown` 文件转换 `html` 文件。所谓动态，即所有转换工作都是在运行时进行。这也就意味着，你如果有需要搭建一个小型项目知识手册之类的网站时，使用 `docsify` 可以让你只需要一个 `index.html` 以及一大堆 `Markdown` 文件即可动态地生成很多 html 文件，让你更加方便地编辑 `Markdown` 文件，而不去操心 `html` 页面的排版问题，专注于内容的创作。

### 用法

1.先安装个node.js

2.然后安装docsify本体

` npm i docsify-cli -g `

3.找个IDE，什么都行，我用pycharm，好像Vscode挺好用，以后试试

后面步骤的什么修改index啊，serve查看网页效果啊，sidebar,coverpage等参照文档([docsify中文文档 (jingping-ye.github.io)](https://jingping-ye.github.io/docsify-docs-zh/#/))



## 插件

当时想安装两个插件，一个是侧边栏折叠，一个是加密文档。后来发现侧边栏折叠没必要，改submaxlevel一样完成美观效果。

装加密文档插件了: 

​	对CSDN同志进行不点名批评啊，真是安禄山进长安唐完了，奶奶的居然能用AI对问题胡说八道煞有介事的编出好像真有一个encrypt插件能用然后堂而皇之地放在公开页面。

​	在github有一个现成的，但是需要php，我打算放在gitee pages上静态页面估计也用不了。果然大神在民间，在别人的博客里找到了能用的([docsify文档加密解密插件 | 云梦 (clboy.cn)](https://www.clboy.cn/archives/docsify文档加密解密插件))

但是和我预想的麻烦了不少：

1.先将要加密的文档内容进行DES加密，DES加密后的结果放在md文件里。我用的是([DES加密/解密 - 在线工具 (toollist.net)](https://toollist.net/zh/des-encrypt)，CBC

2.预装crypto-js和作者根据docsify提供的开发插件接口写的js

3.登记在index



## Markdown语法

[Markdown 基本语法 | Markdown 官方教程](https://markdown.com.cn/basic-syntax/)

### 标题

首先是标题，在字的前面加#，几个#就代表几级标题。注意在docsify的渲染中，第一个一级标题在侧边栏展示的时候是被屏蔽的，我想可能是通常第一个一级标题就是文章的名字吧。

### 引用

使用>号就是代表引用某句话

> 使用>号就是代表引用某句话

在typora中好像是你干什么打一个回车都是处于该状态下的正常换行，打两个回车是退出，如上面的引用和用*显示标题，打一个回车是换行还是在引用/显示标题，打两个回车就回到了编辑正文。

也可以引用链接 语法如下 [name] (url)

[   ]中间是你这条链接的名字  紧跟着的括号是对前面名字提供链接，我是方便写才上面打了个空格。

### 代码

使用`号像双引号一样包裹住代码段

这个反引号在数字1旁边 tab键上面

如果连打三个反引号+语言如c/python，就能代码块上面显示是何语言



# git

## 过程

主要是三个概念，commit repository branch

基础用法 ([改变了世界的软件！程序员的基本功，Git 应该如何使用？_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1u94y1n73L/?spm_id_from=333.337.search-card.all.click&vd_source=3ecda433bad27ee4e395ad1ffebdd84b))

全面用法([Git入门图文教程(1.5W字40图)🔥🔥--深入浅出、图文并茂 - 安木夕 - 博客园 (cnblogs.com)](https://www.cnblogs.com/anding/p/16987769.html))

工作区 暂存区 本地仓库 仓库 

常用的 

` git remote add [name][url]`增加一个远程仓库并命名

` git add [filename] `添加文件到暂存区

` git commit -m "注释"` 提交到本地仓库 

` git push `本地送云端

` git config --global user.name "Your Name" `编辑提交人的名字，这个只是好分辨是谁提交的，安全性靠别的

安全性方面 gitee是点仓库里的code，会提示SSH怎么用，github我用的时候是有个GUI让登账号密码。一开始想用gitee pages部署，国内的可能好连，结果不让用了。

## 遇到的问题

直接用` git commit`会进入像vim编辑器的界面，看最底下一行，按i是进入文本编辑模式，按esc退回，输入:wq是保存并退出。

由于老是挂梯子，会在提交时显示“fail to connect to github.com port"，参照([解决使用git时遇到Failed to connect to github.com port 443 after 21090 ms: Couldn‘t connect to server_git couldn't connect to server-CSDN博客](https://blog.csdn.net/qq_40296909/article/details/134285451))

