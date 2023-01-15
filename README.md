### 创建环境

    conda env create -f environment.yaml
    source activate SLU

### 运行

在根目录下运行 `python scripts/slu_baseline.py -h` 查看参数说明。

示例：

- 训练：使用Bi-LSTM，不使用JIEBA分词，使用tencent_word2vec-200.txt初始化embedding layer：

  ```
  python scripts/slu_baseline.py --embedding tencent_word2vec-200.txt --embed_size 200
  ```
- 训练：使用Bi-LSTM，使用JIEBA分词，使用sgns.wiki.bigram-char初始化embedding layer：

  ```
  python scripts/slu_baseline.py --embedding sgns.wiki.bigram-char --embed_size 300 --segmentation
  ```
- 训练：使用预训练BERT，使用JIEBA分词

  ```
  python scripts/slu_baseline.py --embedding bert_model --segmentation --lr 1e-5
  ```
- 训练：使用固定的预训练BERT作为embedding

  ```
  python scripts/slu_baseline.py --embedding bert_as_embed
  ```
- 训练（当前的best model）：

  ```
  python script/slu_baseline.py -e bert_model --lr 3e-5 --weight_decay
  ```
- 增强数据

  ```
  python utils/data_augmentation/augmentation.py   # delexicalisation + clustering + ranking + generating translation pairs
  python utils/data_augmentation/train_aug.py      # 准备模型
  python utils/data_augmentation/generation.py     # 生成增强数据集
  ```
- 验证脚本

  ```
  python scripts/slu_baseline.py --testing --model best_model.bin --embedding bert_model
  ```
- 测试脚本

  ```
  python scripts/slu_baseline.py --inference --model best_model.bin --embedding bert_model
  ```
- 可视化脚本

  > 需要先生成训练结果（每次训练完成后自动生成），并在visualization里设置可视化对象。
  >

  ```
  python script/visualization.py
  ```


### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成

  python scripts/slu_baseline.py --`<arg>` `<value>`
  其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `utils/data_augmentation/`:seq2seq数据增强相关
+ `utils/visualization.py`:可视化代码
+ `model/slu_baseline_tagging.py`:baseline模型
+ `model/bert_tagging.py`:BERT模型
+ `scripts/slu_baseline.py`:主程序脚本
+ `embedding/`:各种embedding文件
