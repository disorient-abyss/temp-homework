# 1  Bert 分类模型搭建的实现过程

- 1  构建 build_dataset 函数

  - 1.1  传入 bert 预训练模型的 config.json 文件

  - 1.2  构建 load_dataset 函数

    - 1.2.1  传入 train_path，pad_size（默认值为 32）参数

    - 1.2.2  切分数据集为 content（正文内容），label（标签）

    - 1.2.3  对 content 进行 tokenizer.tokenize 分词成 tokens

    - 1.2.4  将 tokens 大小截断留出 2 个位置给 [CLS] 和 [SEP]

    - 1.2.5  拼接 BERT 首尾标记 tokens = [cls_token] + tokens + [sep_token]

    - 1.2.6  通过 tokenizer.convert_tokens_to_ids 将汉字/单词列表转换成一串唯一的数字 ID，并赋值到 token_ids

    - 1.2.7  统一生成 mask 和补齐 token_ids（用 0 补齐）

    - 1.2.8  拼接 contents 成 (token_ids, int(label), seq_len, mask) 元组

    - 1.2.9  返回经历 for 循环后完整的 contents

  - 1.3  对训练集 train，验证集合 dev，测试集 test 进行 load_dataset 封装并返回对应结果

- 2  构建 DatasetIterater 工具类迭代器

  - 2.1  构造函数 \__init\__() 初始化传入的参数

    - 2.1.1   dataset（build_dataset 返回的结果，形状为 (token_ids, int(label), seq_len, mask) 元组）

    - 2.1.2  batch_size（单批次大小）

    - 2.1.3  model_name（预训练模型名称）

  - 2.2  类内部构造的参数

    - 2.2.1  n_batches（分割完整的批次数）

    - 2.2.2  risidue（分割不完全剩余的数据）

    - 2.2.3  index（迭代器内部的位置计数器，从 0 开始）

  - 2.3  构造 \_to\_tensor 类内部函数

    - 2.3.1  将传入的 dataset 中 4 元组转换为 4 个 tensor 张量 x, y, seq_len, mask

    - 2.3.2  返回 (x, seq_len, mask) 和 y 这 2 个张量

  - 2.4  构造 \__next\__ 类内部函数

    - 2.4.1  计算 total_batches 即总共有多少个 Batch（如果有余数，总批次数要在整除基础上加 1）

    - 2.4.2  判断 index >= total_batches 用于抛出标准迭代终止异常，通知外部的 for 循环停止当前轮次的遍历

    - 2.4.3  由 index * batch_size 计算出 start_idx

    - 2.4.4  由 start_idx * batch_size 与 dataset 比较取出最小值作为 end_idx

    - 2.4.5  得到 batch_slice 切片结果为[start_idx:end_idx]

    - 2.4.6  index 自增 1

  - 2.5  构造 \__iter\__ 类内部函数，用于 for 时触发初始化 index 置 0

  - 2.6 n_batches + 构造 \__len\__类内部函数，用于判断 residue 是否为 0，是则返回 n_batches + 1，否则返回 n_batches

- 3  构建 Config 类

  - 3.1  初始化 \__init\__ 构造函数，引入数据集路径 data_path

  - 3.2  读取训练集 train，验证集 dev，测试集 test 以及分类文件

  - 3.3  设置模型保存路径

  - 3.4  引入硬件环境与超参数

    - 3.4.1  设置 device 为 cuda

    - 3.4.2  require_improvement 限制个 batch 的训练量

    - 3.4.3  设置 num_epochs

    - 3.4.4  设置 batch_size 大小

    - 3.4.5  设置 pad_size 大小，用于每句话截断/填充的目标长度

    - 3.4.6  设置 learning_rate，最好小于 5e-5

  - 3.5  处理预训练模型相关内容

    - 3.5.1  设置 bert_path

    - 3.5.2  通过 BertTokenizer 类自动找寻 vocab.txt 生成 tokenizer 对象

    - 3.5.3  通过 BertConfig 类自动找寻 config 文件生成 bert_config 对象

    - 3.5.4  通过 bert_config 动态读取隐藏层维度并赋值给 hidden_size（bert 一般为 768 维，用于后续分类制作混合矩阵）

- 4  构建 Model 类

  - 4.1  初始化 \__init\__ 构造函数，传入 config 参数

    - 4.1.1  继承 nn.Module 父类并重写 \__init\__ 方法

    - 4.1.2  通过 BertModel 类引入 bert_path, bert_config 参数用来加载预训练模型 bert

    - 4.1.3  引入 dropout 防止全连接过拟合

    - 4.1.4  通过 nn.Linear 将 768 维的 hidden_size 转换成 num_classes 分类问题

  - 4.2  构建 forward 函数，引入张量 x

    - 4.2.1  因为张量形状为 (context, seq_len, mask)

    - 4.2.1.1 取 x[0] 为 context

    - 4.2.1.2 去 x[2] 为 mask

    - 4.2.2  传入 context 和 mask 进入预训练模型 bert 输出 outputs

    - 4.2.3  通过对 outputs 进行 pooler_output，即全句信息的 [CLS] 向量经过内置 Dense + Tanh 变换后，输出形状为 [batch_size, hidden_size] 的 pooled_out 张量，简单来说，就是提取出整合了全句语义的句表征向量，专用于供后续分类层使用。

    - 4.2.4 将 pooled_out 张量通过 dropout 处理后通过线性层分类，即先升维到 hidden_size 大小，再降维到 num_classes 分类数量并返回结果

- 5  构建 train_eval.py 训练、验证以及测试代码

  - 5.1  构造 \_to\_device 函数，并递归调用自身，将传入的数据迁移到指定设备并启用异步传输，即 CPU 将拷贝任务交给后台的 DMA（直接内存访问）通道后，立刻继续往下执行代码（比如进行下一轮数据加载或准备），数据传输与 CPU 计算实现并行重叠（Overlap）

  - 5.2  编写训练函数 train

    - 5.2.1  传入 config, model, train_iter, dev_iter 参数

    - 5.2.2  区分需要权重衰减和不需要权重衰减的参数写入 no_decay 中

    - 5.2.3  设置非 no_decay 的组 weight_decay 数据为 0.01，no_decay 组 weight_decay 数据为 0.0

    - 5.2.4  通过 AdamW 对分组后的模型根据 lr 执行参数更新

    - 5.2.5  设置模型训练前初始化参数

      - 5.2.5.1  total_batch，记录进行到多少 batch（跨 epoch 持续累加）初始化置 0

      - 5.2.5.2  dev_best_loss，记录验证集最佳 loss，初始设置为无穷大

      - 5.2.5.3  last_improve，记录上一次验证集 loss 下降的 batch 数

      - 5.2.5.4  early_stop，早停标志位（布尔开关），验证是否过了 1000 batch 没有提升

    - 5.2.6  执行 model.train()

      - 5.2.6.1  根据 num_epochs 数值建立 for epoch 循环训练

      - 5.2.6.1  按 batch_size 通过第 2 层 for 循环逐批遍历训练迭代器 train_iter，将迭代器每次产出的二元组 ((x, seq_len, mask), y) 拆开，把特征部分赋给 trains（供模型做前向传播），把标签部分赋给 labels（供损失函数对比误差）。

      - 5.2.6.2  通过 \_to\_device 函数将模型张量迁入 GPU (支持 non_blocking 异步重叠)

      - 5.2.6.3  通过 model(trains) 进入训练输出 outputs 结果

      - 5.2.6.4  对 outputs 和 labels 进行交叉商得到损失函数得到 loss

      - 5.2.6.5  梯度清零

      - 5.2.6.6  反响传播

      - 5.2.6.7  跟新参数

      - 5.2.6.8  判断是否训练满足 200 个 batch，满足就跑一轮验证集

        - 5.2.6.8.1  把 labels 拿出来作为真实标签 true

        - 5.2.6.8.2  选出 outputs 最大的值为 predic

        - 5.2.6.8.3  调用 sklearn 的包，通过 true， predic，计算 train_acc

        - 5.2.6.8.4  通过 evaluate 对 model 跑一次验证集得到 dev_acc, dev_loss

        - 5.2.6.8.5  如果验证集的损失降低了就保存一次 model 的状态字典

        - 5.2.6.8.6  评估完成后将模型重新置于训练模式 model.train()

  - 5.3  编写测试函数 test

    - 5.3.1  传入 config, model, test_iter

      - 5.3.2  开启模型评估模式，model.eval()，避免测试时 dropout 生效

    - 5.3.3  调用 evaluate 函数

      - 5.3.3.1 传入 config, model, test_iter, test 参数

      - 5.3.3.2 初始化 loss_total = 0, predict_list = [], labels_list = []

      - 5.3.3.3 通过 torch.nograd 封锁反向传播

        - 5.3.3.3.1  for 循环通过 data_iter 数据迭代器，拿出 texts（测试文本），labels（标签）

        - 5.3.3.3.2  通过数据集 texts 拿到模型输出 outputs

        - 5.3.3.3.3  对模型输出 outputs 和 labels 求交叉熵得到损失函数 loss

        - 5.3.3.3.4  累计所有 loss 得到 loss_total

        - 5.3.3.3.5  同理拿到所有的标签 labels_list，所有的模型的预测结果 predict_list

        - 5.3.3.3.6  计算模型 acc 和 平均损失 avg_loss 并返回

# 2 Bert 分类模型搭建的测试用例

- 1  构建 run.py 

  - 1.1  固定全局随机种子

  - 1.2  通过 model_name 动态加载模型与配置模块

  - 1.3  构建数据集与批次迭代器

    ```python
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    ```

  - 1.4  初始化模型并迁移至目标设备 (GPU / CPU)

    ```python
    model = model_module.Model(config).to(config.device)
    ```

  - 1.5  启动训练与最终测试

    ```python
    train(config, model, train_iter, dev_iter)
    test(config, model, test_iter)
    ```

- 2  执行命令，并得出训练与评估结果

  - 2.1  执行命令

    ```zsh
    python run.py --model bert --dataset toutiao
    ```

  - 2.2  得出训练与评估结果

    ```zsh
    python src/bert_pretrain/run.py --model bert --dataset data
    Loading data for [BERT] on [data] dataset...
    Loading train.txt: 180000it [00:04, 38922.39it/s]
    Loading dev.txt: 10000it [00:00, 42622.05it/s]
    Loading test.txt: 10000it [00:00, 43308.65it/s]
    Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 68844.15it/s]
    [transformers] BertModel LOAD REPORT from: ./data/bert_pretrain
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
    cls.predictions.decoder.weight             | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    Epoch [1/3]
    Training Epoch 1:  14%|██████████▍                                                               | 199/1407 [00:27<02:54,  6.92it/s]Iter:    200, Train Loss:  0.22, Train Acc: 92.97%, Val Loss:  0.27, Val Acc: 91.65%, Time: 0:00:31 *
    Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:59<02:22,  7.06it/s]Iter:    400, Train Loss:  0.25, Train Acc: 93.75%, Val Loss:  0.25, Val Acc: 92.44%, Time: 0:01:03 *
    Training Epoch 1:  43%|███████████████████████████████▌                                          | 599/1407 [01:31<01:54,  7.06it/s]Iter:    600, Train Loss:  0.29, Train Acc: 88.28%, Val Loss:  0.24, Val Acc: 92.21%, Time: 0:01:35 *
    Training Epoch 1:  57%|██████████████████████████████████████████                                | 799/1407 [02:03<01:26,  7.03it/s]Iter:    800, Train Loss:  0.17, Train Acc: 95.31%, Val Loss:  0.23, Val Acc: 92.79%, Time: 0:02:07 *
    Training Epoch 1:  71%|████████████████████████████████████████████████████▌                     | 999/1407 [02:34<00:58,  7.02it/s]Iter:   1000, Train Loss:  0.16, Train Acc: 92.19%, Val Loss:  0.22, Val Acc: 92.99%, Time: 0:02:39 *
    Training Epoch 1:  85%|██████████████████████████████████████████████████████████████▏          | 1199/1407 [03:06<00:29,  7.02it/s]Iter:   1200, Train Loss:  0.15, Train Acc: 96.09%, Val Loss:  0.21, Val Acc: 92.73%, Time: 0:03:10 *
    Training Epoch 1:  99%|████████████████████████████████████████████████████████████████████████▌| 1399/1407 [03:38<00:01,  6.99it/s]Iter:   1400, Train Loss:  0.12, Train Acc: 95.31%, Val Loss:  0.20, Val Acc: 93.86%, Time: 0:03:42 *
    Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████| 1407/1407 [03:43<00:00,  6.30it/s]
    Epoch [2/3]
    Training Epoch 2:  14%|██████████                                                                | 192/1407 [00:27<02:53,  7.02it/s]Iter:   1600, Train Loss:  0.12, Train Acc: 97.66%, Val Loss:  0.21, Val Acc: 93.76%, Time: 0:04:14 
    Training Epoch 2:  28%|████████████████████▌                                                     | 392/1407 [00:59<02:24,  7.01it/s]Iter:   1800, Train Loss:  0.10, Train Acc: 96.09%, Val Loss:  0.21, Val Acc: 93.46%, Time: 0:04:46 
    Training Epoch 2:  42%|███████████████████████████████▏                                          | 592/1407 [01:30<01:56,  6.99it/s]Iter:   2000, Train Loss:  0.23, Train Acc: 92.97%, Val Loss:  0.21, Val Acc: 93.38%, Time: 0:05:17 
    Training Epoch 2:  56%|█████████████████████████████████████████▋                                | 792/1407 [02:02<01:27,  7.01it/s]Iter:   2200, Train Loss:  0.14, Train Acc: 94.53%, Val Loss:  0.21, Val Acc: 93.58%, Time: 0:05:49 
    Training Epoch 2:  71%|████████████████████████████████████████████████████▏                     | 992/1407 [02:34<00:59,  6.98it/s]Iter:   2400, Train Loss:  0.17, Train Acc: 93.75%, Val Loss:  0.21, Val Acc: 93.62%, Time: 0:06:21 
    Training Epoch 2:  71%|████████████████████████████████████████████████████▏                     | 993/1407 [02:37<07:45,  1.13s/it]No optimization for a long time, auto-stopping...
    Training Epoch 2:  71%|████████████████████████████████████████████████████▏                     | 993/1407 [02:37<01:05,  6.30it/s]
    Test Loss:  0.20, Test Acc: 93.97%
    Precision, Recall and F1-Score...
                   precision    recall  f1-score   support
    
          finance     0.9169    0.9490    0.9327      1000
           realty     0.9258    0.9730    0.9488      1000
           stocks     0.9134    0.8650    0.8885      1000
        education     0.9612    0.9650    0.9631      1000
          science     0.9178    0.9050    0.9114      1000
          society     0.9173    0.9430    0.9300      1000
         politics     0.9232    0.9140    0.9186      1000
           sports     0.9879    0.9770    0.9824      1000
             game     0.9754    0.9500    0.9625      1000
    entertainment     0.9598    0.9560    0.9579      1000
    
         accuracy                         0.9397     10000
        macro avg     0.9399    0.9397    0.9396     10000
     weighted avg     0.9399    0.9397    0.9396     10000
    
    Confusion Matrix...
    [[949  13  24   0   3   3   7   1   0   0]
     [  8 973   2   1   2   4   5   2   0   3]
     [ 56  33 865   1  26   2  16   0   0   1]
     [  4   1   2 965   1  14   6   0   0   7]
     [  7   5  19   1 905  15  15   1  21  11]
     [  3  11   0  14   4 943  19   0   1   5]
     [  5   6  26  13  13  18 914   1   0   4]
     [  1   3   4   0   0   4   4 977   0   7]
     [  0   3   3   3  26   9   2   2 950   2]
     [  2   3   2   6   6  16   2   5   2 956]]
    Time usage: 0:00:03
    ```
