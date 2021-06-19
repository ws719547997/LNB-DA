## 基于终身朴素贝叶斯方法的情感分类

​	中文信息学报论文《基于终身朴素贝叶斯方法的情感分类》的代码仓库。论文方法LNB-DA是在LNB的基础上提出了领域注意力机制，为相似的任务赋予更高的权重，改善知识挖掘性能。 本仓库包含以下内容：

1. **LNB-DA**：本文方法。
2. **JD21**：在本文中提出的自建中文数据集 。
3. **对比实验**：分为两个代码库。8种方法。

### LNB-DA

LNB为论文《Forward and Backward Knowledge Transfer for Sentiment Classification》

LNB代码官方实现：https://github.com/cshaowang/lnb

LNB-DA做了如下修改

1. 改写了dataloader，支持了读取中文数据集，名为Reuters10。
2. 编写了领域注意力模块和3中注意力模式 cmdOption.attantionMode = "att" "att_max" "att_percent" or "none".
3. 编写了调参代码 /src/mainloop，注意这个代码仅能用于LNB及LNB-DA的调参。
4. 编写了实验结果处理模块（见小工具）。

```
├── Data
│   ├── DomainToEvaluate
│   │   └── stock.txt  //任务的名称和顺序在这里定义
│   ├── Input
│   │   └── stock_comments  //这里放完整的数据集
│   ├── Intermediate
│   │   ├── Knowledges  //统计出来的知识被放在这里
│   │   ├── TestingDocs  //这里放划分好的测试集
│   │   ├── TrainingDocs  //这里是测试集
│   │   └── inter
│   └── Output  //输出的结果在这里
│       └── SentimentClassificaton
└── src
    └─── main
        ├── CmdOption.java  //部分实验参数
        ├── Constant.java
        ├── MainEntry.java  //源代码的主入口，在这里选择分类器
        └── MainLoop.java  //专门针对LNB-att调参的入口
```

### 语料库  JD21

本项目自建了中文多领域情感分类数据集:

['修复霜.txt', '电动牙刷.txt', '吹风机.txt', '.DS_Store', '平板电脑.txt', '蛋白粉.txt', '电视.txt', '海鲜.txt', '运动鞋.txt', '褪黑素.txt', '洗面奶.txt', '护肤品.txt', '游戏机.txt', '酒.txt', '无线耳机.txt', '智能手表.txt', '红米手机.txt', 'iPhone.txt', 'MacBook.txt', '维生素.txt', '智能手环.txt', '小米手机.txt']

数据采集自JD.com。语料库建设分为两个阶段：第一阶段为2020年7-9月，21个品类的商品每种1000条左右。第二阶段为2021年5月，每个品类商品被扩充到5000条左右，语料库为12万条。负类平均占比为23.3%，结巴分词后，每条的次数众数为11词。详情信息见dataSetINFO，对语料库的简单处理详见小工具。

数据集路径：

LNB：（划分好的版本）放在LNB-DA/Data/Intermediate中的TestingDocs和TrainingDocs中。（完整版）放在LNB-DA/Data/Input/stock_comments中。

神经网络对比实验和SRK：JD21/data/train, dev, test。  

```
├── cutword //分词后的完整版
├── 82word //对数据集进行了8：2的划分 
└── dataSetsINFO.xlsx
```

### 对比实验

1. LNB：http://proceedings.mlr.press/v101/wang19f.html

2. LSC：https://www.aclweb.org/anthology/P15-2123

3. LLV：《Distantly Supervised Lifelong Learning for Large-Scale Social Media Sentiment Analysis

   以上三个实验的代码在LNB代码库中。

4. FastText：《Bag of Tricks for Efficient Text Classification》

5. TextCNN：《Convolutional Neural Networks for Sentence Classification》

6. TextRNN：《Recurrent neural network for text classification with multi-task learning》

7. DPCNN：《Deep Pyramid Convolutional Neural Networks for Text Categorization》

8. Transformer：http://arxiv.org/abs/1706.03762

9. EWC：http://arxiv.org/abs/1612.00796

   以上六个实验的代码在‘神经网络对比实验’代码库中。

10. SRK：https://www.researchgate.net/publication/332580631_Sentiment_Classification_by_Leveraging_the_Shared_Knowledge_from_a_Sequence_of_Domains

    SRK的方法在SRK代码库中。

#### 神经网络对比实验

基于https://github.com/649453932/Chinese-Text-Classification-Pytorch

1. 改写了dataloader，支持本语料库的同时，支持（S\T\ST）数据读取模式；
2. 改写了run.py和train.py来进行论文中的相关实验。
3. 增加了实验结果整理模块

```
├── 21domain_avg.sh  //运行全部模型avg的脚本
├── BWT_avg.sh  //评价BWT的脚本，每学习一个新任务，测试所有学习过的任务
├── JD21  //数据集
│   ├── data
│   │   ├── class.txt  //label的类别
│   │   ├── dev  //10%
│   │   ├── embedding_SougouNews.npz  //根据词表获取的词向量
│   │   ├── test  //20%
│   │   ├── train  //70%
│   │   ├── train.txt  //根据全部的训练集获取词表
│   │   └── vocab.pkl  //词表
│   ├── log  //训练过程中的日志
│   ├── result  //结果文件以excel的形式存在这里~
│   └── saved_dict  //训练过程中的模型保存在这里
├── models  //模型文件
│   ├── DPCNN.py
│   ├── FastText.py
│   ├── TextCNN.py
│   ├── TextRCNN.py
│   ├── TextRNN.py
│   ├── TextRNN_Att.py
│   └── Transformer.py
├── run.py  // 这两个run略有不同，这个是BWT
├── run2.py  //这个是计算avg
├── train_eval.py  //对应的工具也不同
├── train_eval2.py
├── utils.py  // dataloader在这里，也被我修改了,生成词向量的代码需要直接运行这个文件
└── utils_fasttext.py  // 这个是针对fasttext
```

#### SRK

基于https://github.com/ZixuanKe/LifelongSentClass/tree/master/reference/SRK。

改写了dataloader。

### 小工具

小工具主要实现对原始数据的预处理，划分。对实验代码结果的处理与分析。

```
├── acc_counter.py  //统计LNB实验的实验结果到一个excel表格里
├── makedata_jd21.py  //把原始数据处理成数据集
├── sent_lenth_counter.py  //对语料库频率和句子长度的统计
└── split55.py  //可以对处理好的数据集进行划分
```

### 致谢

感谢Hao Wang，Guangyi Lv， Zixuan Ke在持续学习情感分类领域的出色工作和无私分享；在email交流中给予我的帮助。

### 联系我

email：719547997@qq.com

王松

2021-06