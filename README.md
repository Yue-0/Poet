# AI诗人

## 目录

1. [简介](#简介)
2. [效果展示](#效果展示)
3. [文件结构](#文件结构)
4. [快速开始](#快速开始)
   1. [PaddlePaddle版本](#1.PaddlePaddle版本)
   2. [PyTorch版本](#2.PyTorch版本)

## 简介

本项目是我大二的深度学习课程期中作业，项目内容包括：
* 构建一个古诗数据集；
* 使用PyTorch搭建一个LSTM网络；
* 在构建的古诗数据集上训练LSTM；
* 实现一个能够 __续写诗句__ 和 __创作藏头诗__ 的AI。

本项目的[PaddlePaddle版本](https://aistudio.baidu.com/aistudio/projectdetail/5313721)已经在百度AI Studio开源。

## 效果展示

* 续写诗句
```
[Input]  时维九月
[Output] 时维九月上蓬莱，天上星辰照斗魁。
         万里长江千里月，一天明月两峰梅。
         风吹玉笛吹残梦，月照金波照碧哉。
         不是金陵旧游客，一时分付与君来。

[Input]  九重环佩艳琳琅
[Output] 九重环佩艳琳琅，
         一夜祥风入舜裳。
         天上玉皇香案吏，
         人间金粟玉麒麟。

[Input]  他强由他强，清风拂山岗
[Output] 他强由他强，清风拂山岗。
         不知何处去，空有一枝芳。
         露重风初定，风高月正长。
         谁知此时意，独自倚阑干。
```

* 创作藏头诗
```
[Input]  人工智能
[Output] 人生何处不相逢，一日相逢一笑中。
         工部未能忘世俗，文章犹自在天工。
         智愚不必论三益，忧患须知有一同。
         能使此心无所用，尧夫非是爱吟诗。

[Input]  深度学习
[Output] 深山无处觅，
         度岭有人行。
         学道无人到，
         习池多日生。


[Input]  循环神经网络
[Output] 循吏归来日，
         环山望故乡。
         神仙归故里，
         经术付诸郎。
         网罟方为累，
         络车犹未央。
```

## 文件结构

```
Poet
├── data              # 存放数据集代码和数据
   └── __init__.py
├── model             # 存放网络结构代码和网络参数
   └── __init__.py
├── config.py         # 超参数定义文件
├── LICENSE           # LICENSE文件
├── main.ipynb        # 主程序
├── poet.py           # API代码
├── README.md         # 说明文件
├── requirements.txt  # 依赖库列表
└── train.py          # 训练程序
```

## 快速开始

### 1.PaddlePaddle版本

__PaddlePaddle版本无需克隆、无需配置环境、无需训练，[点击这里](https://aistudio.baidu.com/aistudio/projectdetail/5313721)即可快速体验！__

1. 前往[百度AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/5313721)，fork项目。
2. 打开run.ipynb，无需训练，即可快速体验。
3. 此外，main.ipynb提供了使用PaddlePaddle从零开始训练的全部教程。

### 2.PyTorch版本

#### 1.克隆项目

```shell
git clone https://github.com/Yue-0/Poet.git
cd ./Poet
```

#### 2.安装依赖

本项目的依赖库包括：
* tqdm
* torch

```shell
pip install -r requirements.txt
```

#### 3.下载数据集

项目数据集共包含250435首古诗，每首古诗都是五言诗或七言诗，只包含正文部分，且长度小于100个字符。
数据集已经上传到百度AI Studio，[前往下载](https://aistudio.baidu.com/aistudio/datasetdetail/131509)。

下载数据集后，请放置在data目录下。

#### 4.开始训练

开始训练前，须 __保证model/parameters文件夹里没有.pt文件，否则可能失败！__
默认使用GPU训练。

```shell
python train.py
```

可以修改[config.py](config.py)中的参数来更改网络结构和训练超参数，默认参数如下：
```python
lr: float = 1e-3  # 学习率
batch: int = 8    # Batch size
epoch: int = 12   # Epochs

max_len: int = 95  # 最长诗句的长度

num_layers: int = 3       # LSTM层数
hidden_dim: int = 1024    # LSTM隐藏层神经元个数
embedding_dim: int = 512  # 词嵌入维度
```

#### 5.开始体验

打开[main.ipynb](main.ipynb)，按照提示即可开始体验。
