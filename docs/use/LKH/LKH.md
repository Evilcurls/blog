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

