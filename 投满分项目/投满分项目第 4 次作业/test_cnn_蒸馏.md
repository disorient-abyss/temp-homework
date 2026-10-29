# 1  知识蒸馏模型 TextCNN 构建

- 1  配置 config 类

  - 1.1  初始化参数 \__init\__()

    - 1.1.1  require_improvement（早停步数容忍阈值，1000）

    - 1.1.2  num_epochs（最大训练轮数，20）

    - 1.1.3  batch_size（批处理大小，128）

    - 1.1.4  pad_size（文本对齐长度，32）

    - 1.1.5  learning_rate（学习率，1e-3）

  - 1.2  TextCNN 核心网络架构参数

    - 1.2.1  n_vocab（词表容量，10000）

    - 1.2.2  pad_idx（填充占位符索引）

    - 1.2.3  embedding_dim（词嵌入维度，300）

    - 1.2.4  filter_sizes（卷积核尺寸 / 窗口宽度，(2, 3, 4)）

    - 1.2.5  num_filters（卷积通道数，256）

    - 1.2.6  dropout（随机失活概率，0.5）

- 2  构建 Model 类

  - 2.1  初始化 \__init\__() 传入Config 类构建的 config 对象

    - 2.1.1  构建 embedding 层，传入以下参数

      - 2.1.1.1  n_vocab

      - 2.1.1.2  embedding_dim 

      - 2.1.1.3  padding_idx

    - 2.1.2  构建 Conv2d 对象的 ModuleList

      - 2.1.2.1 构建 filter_sizes 个，即 2 + 3 + 4 = 9

      - 2.1.2.2  每份大小为 filter_sizes * embedding_dim

      - 2.1.2.3  每份数量为 num_filters 个卷积核

    - 2.1.3  定义 fc 全连接层

      - 2.1.3.1  第一层网络数量为 num_filters * len(config.filter_sizes)

      - 2.1.3.2  第二层网络数量为 num_classes 投满分的 10 分类大小

  - 2.2  构建 forward 函数

    - 2.2.1  给 embedding 的 dim = 1 添加单个维度 unsqueeze(1) 方便后续 cat

      - [B, L, D] -> [B, 1, L, D]

    - 2.2.2  进行池化操作与在 dim = 1 维度进行拼接

      - 2.2.2.1  conv_and_pool(x, conv) 池化

        - 2.2.2.1.1  输入: [B, 1, L, D]

        - 2.2.2.1.2  核大小: (k, D) -> 纵向滑动步长为 1，横向正好覆盖整个词向量维度 D

        - 2.2.2.1.3  F.relu(...) 激活不改变形状 -> [B, num_filters, L - k + 1, 1]

        - 2.2.2.1.4  squeeze(3) 挤压掉大小为 1 的最后一维 -> [B, num_filters, L - k + 1]

        - 2.2.2.1.5  池化后形状: [B, num_filters, 1]

        - 2.2.2.1.6  squeeze(2) 挤压掉池化后的时间步维度并返回 -> [B, num_filters]

    - 2.2.3  给输出张量 x 进行 dropout

    - 2.2.4  给输出张量 x 进行线性全连接层映射得到 class 分类结果

# 2  知识蒸馏专属训练函数 train_kd 构建 (BERT 教师指导 TextCNN 学生)

- 1  传参

  - 1.1  cnn_config, 

  - 1.2  bert_model, 

  - 1.3  cnn_model, 

  - 1.4  train_iter, 

  - 1.5  dev_iter, 

  - 1.6  alpha, 

  - 1.7  T

- 2  开启 bert_model.eval() 模式

- 3  关闭所有的 bert_model.parameters() 禁止 bert 模型本身的参数更新

- 4  开启 cnn_model.train() 训练

- 5  设置优化器 optimizer 为 AdamW 传入参数和学习率

- 6  开启双循环 epoch 训练

  - 6.1  通过 bert_model(teacher_inputs) 拿到关闭梯度更新的 teacher_outputs

  - 6.2  通过 cnn_model(student_inputs) 拿到 student_outputs

- 7  根据 alpha 和 T 计算 student_outputs 和 teacher_outputs 的损失 loss

  - $L = (1 - \alpha) \text{CE}(y, p) + \alpha T^2 \text{CE}(q^T, p^T)$

- 8  梯度清零 -> 反向传播 -> 参数更新

# 3  开启蒸馏 bert 训练 textCNN 模型

- 1  进行训练

  ```zsh
  Processing dual-input dataset for distillation...
  Student Vocab size: 4762
  Processing train.txt: 100%|████████████████████████████████████████████████████████████████████████████| 88/88 [00:02<00:00, 41.03it/s]
  Processing dev.txt: 100%|███████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 393.23it/s]
  Processing test.txt: 100%|███████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 17.01it/s]
  Distillation Parameters: Alpha=0.8, Temperature=2.0, LR=0.001
  KD Epoch [1/20]
  Distilling Epoch 1:  14%|██████████▌                                                                | 197/1407 [00:08<00:53, 22.57it/s]Iter:    200, Train Loss:  1.49, Train Acc: 79.69%, Val Loss:  0.61, Val Acc: 81.40%, Time: 0:00:09 *
  Distilling Epoch 1:  28%|█████████████████████▎                                                     | 399/1407 [00:17<00:44, 22.68it/s]Iter:    400, Train Loss:  1.19, Train Acc: 81.25%, Val Loss:  0.52, Val Acc: 83.81%, Time: 0:00:18 *
  Distilling Epoch 1:  42%|███████████████████████████████▊                                           | 597/1407 [00:26<00:36, 22.47it/s]Iter:    600, Train Loss:  1.08, Train Acc: 82.81%, Val Loss:  0.47, Val Acc: 85.48%, Time: 0:00:27 *
  Distilling Epoch 1:  57%|██████████████████████████████████████████▌                                | 799/1407 [00:35<00:27, 22.49it/s]Iter:    800, Train Loss:  1.20, Train Acc: 75.00%, Val Loss:  0.43, Val Acc: 86.83%, Time: 0:00:36 *
  Distilling Epoch 1:  71%|█████████████████████████████████████████████████████▏                     | 997/1407 [00:44<00:18, 22.44it/s]Iter:   1000, Train Loss:  0.95, Train Acc: 83.59%, Val Loss:  0.44, Val Acc: 86.52%, Time: 0:00:45 
  Distilling Epoch 1:  85%|███████████████████████████████████████████████████████████████           | 1199/1407 [00:53<00:09, 22.55it/s]Iter:   1200, Train Loss:  0.84, Train Acc: 85.94%, Val Loss:  0.40, Val Acc: 87.58%, Time: 0:00:54 *
  Distilling Epoch 1:  99%|█████████████████████████████████████████████████████████████████████████▍| 1397/1407 [01:02<00:00, 22.44it/s]Iter:   1400, Train Loss:  0.83, Train Acc: 84.38%, Val Loss:  0.39, Val Acc: 88.14%, Time: 0:01:03 *
  Distilling Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████| 1407/1407 [01:03<00:00, 22.24it/s]
  
    ......
  
  KD Epoch [20/20]
  Distilling Epoch 20:   5%|███▌                                                                       | 66/1407 [00:02<00:59, 22.46it/s]Iter:  26800, Train Loss:  0.28, Train Acc: 98.44%, Val Loss:  0.28, Val Acc: 91.43%, Time: 0:20:11 
  Distilling Epoch 20:  19%|█████████████▉                                                            | 264/1407 [00:11<00:50, 22.50it/s]Iter:  27000, Train Loss:  0.34, Train Acc: 92.97%, Val Loss:  0.28, Val Acc: 91.25%, Time: 0:20:20 
  Distilling Epoch 20:  33%|████████████████████████▌                                                 | 466/1407 [00:20<00:41, 22.47it/s]Iter:  27200, Train Loss:  0.31, Train Acc: 96.09%, Val Loss:  0.29, Val Acc: 91.21%, Time: 0:20:29 
  Distilling Epoch 20:  47%|██████████████████████████████████▉                                       | 664/1407 [00:29<00:33, 22.47it/s]Iter:  27400, Train Loss:  0.38, Train Acc: 96.88%, Val Loss:  0.28, Val Acc: 91.42%, Time: 0:20:38 
  Distilling Epoch 20:  62%|█████████████████████████████████████████████▌                            | 866/1407 [00:38<00:24, 22.47it/s]Iter:  27600, Train Loss:  0.29, Train Acc: 96.09%, Val Loss:  0.29, Val Acc: 91.31%, Time: 0:20:47 
  Distilling Epoch 20:  76%|███████████████████████████████████████████████████████▏                 | 1064/1407 [00:47<00:15, 22.48it/s]Iter:  27800, Train Loss:  0.33, Train Acc: 92.97%, Val Loss:  0.28, Val Acc: 91.42%, Time: 0:20:56 
  Distilling Epoch 20:  90%|█████████████████████████████████████████████████████████████████▋       | 1266/1407 [00:56<00:06, 22.48it/s]Iter:  28000, Train Loss:  0.29, Train Acc: 98.44%, Val Loss:  0.29, Val Acc: 91.43%, Time: 0:21:05 
  Distilling Epoch 20: 100%|█████████████████████████████████████████████████████████████████████████| 1407/1407 [01:03<00:00, 22.22it/s]
  ```

- 2  测试模型训练结果

  ```zsh
   Test Loss:  0.26, Test Acc: 92.06%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9365    0.8990    0.9173      1000
         realty     0.9320    0.9450    0.9384      1000
         stocks     0.8649    0.8900    0.8773      1000
      education     0.9525    0.9620    0.9572      1000
        science     0.8617    0.8850    0.8732      1000
        society     0.8994    0.9120    0.9057      1000
       politics     0.9099    0.8990    0.9044      1000
         sports     0.9766    0.9610    0.9688      1000
           game     0.9337    0.9300    0.9319      1000
  entertainment     0.9438    0.9230    0.9333      1000
  
       accuracy                         0.9206     10000
      macro avg     0.9211    0.9206    0.9207     10000
   weighted avg     0.9211    0.9206    0.9207     10000
  
  Confusion Matrix...
  [[899  14  53   4   8   9   5   2   4   2]
   [  8 945  14   2   3  11   5   3   2   7]
   [ 39  17 890   1  23   2  23   2   1   2]
   [  1   3   1 962   7   7  11   1   0   7]
   [  0   5  34   3 885  15  19   3  30   6]
   [  2  18   2  19  18 912  16   1   5   7]
   [  7   3  24   8  23  29 899   1   2   4]
   [  1   2   4   1   4   8   5 961   1  13]
   [  1   0   5   3  46   3   2   3 930   7]
   [  2   7   2   7  10  18   3   7  21 923]]
  Time usage: 0:00:00
  ```

- 3  训练得到的数据字典大小对比

  ```sh
     8320 -rw-r--r--. 1 disorient disorient    8517016 10月29日 10:26 text_cnn.pt
   399596 -rw-r--r--. 1 disorient disorient  409182765 10月26日 22:54 macbert.pt
    38936 -rw-r--r--. 1 disorient disorient   39869753 10月24日 17:28 albert.pt
   399596 -rw-r--r--. 1 disorient disorient  409182765 10月24日 12:09 roberta.pt
   399580 -rw-r--r--. 1 disorient disorient  409169856 10月24日 10:30 bert.pt
   149472 -rw-r--r--. 1 disorient disorient  153055243 10月23日 23:39 bert_int8.pt
   149068 -rw-r--r--. 1 disorient disorient  152643571 10月23日 23:27 bert_fp8.pt
  1694056 -rw-r--r--. 1 disorient disorient 1734712369 10月19日 11:59 toutiao_fasttext_jieba_1792382370.bin
  ```

> [!Summary] 作业总结
>
> - 训练动态与收敛特征
> - 参数配比合理：蒸馏超参数设为 $\alpha = 0.8, T = 2.0$。给教师模型的软标签（Soft Targets）赋予了 80% 的主导权重，使 TextCNN 学习到了类别间的暗知识（Dark Knowledge，如财经与股票之间的关联相似度）。
> - 收敛迅速且平稳：
>   - 第 1 轮 结束时，验证集准确率即从 81.40% 快速攀升至 88.14%。
>   - 第 20 轮 稳定收敛至 Train Acc ~96-98%，Val Acc 维持在 91.4% 左右，泛化到测试集达到 92.06%，无明显过拟合。
> - 工程吞吐极高：单卡迭代速度约 22.5 it/s，单个 Epoch（1407 iters）仅需 63 秒，整整 20 轮蒸馏总耗时约 21 分钟，训练成本极低。
> - 落地价值
> - 本次实践验证了跨架构蒸馏（Transformer 教师 $\to$ CNN 学生）在短文本多分类任务上的可行性。TextCNN 去除了复杂的自注意力机制，由轻量卷积核配合池化抽取 n-gram 特征，在维持 92%+ 工业级准确率的同时，将体积缩减至 8.5MB，可直接部署于低配边缘设备或高 QPS（如网关分类、实时推荐初筛）的 CPU 纯算力集群。
