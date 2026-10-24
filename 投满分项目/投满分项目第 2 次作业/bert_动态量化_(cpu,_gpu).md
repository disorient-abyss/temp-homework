# bert 模型的动态量化

## 1  bert 模型的动态量化结果

- 1  量化后各模型大小:

  - 1.1  bert 原模型大小为 409.2 MB

  - 1.2  bert 进行 cpu_int8 量化后模型大小为 153.1 MB

  - 1.3  bert 进行 gpu_fp8 量化后模型大小为 152.6 MB

  ```zsh
   149472 -rw-r--r--. 1 disorient disorient  153055243 10月23日 23:39 bert_int8.pt
   149068 -rw-r--r--. 1 disorient disorient  152643571 10月23日 23:27 bert_fp8.pt
   399580 -rw-r--r--. 1 disorient disorient  409169856 10月20日 19:24 bert.pt
  ```

- 1  未量化的 bert 模型跑测试集，得到的 ACC 为 94.34 %

  ```zsh
  Loading test dataset for [BERT] on [toutiao]...
  Loading test.txt: 10000it [00:00, 43082.62it/s]
  Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 64612.67it/s]
  [transformers] BertModel LOAD REPORT from: /home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/data/bert_pretrain
  Key                                        | Status     |  | 
  -------------------------------------------+------------+--+-
  cls.predictions.bias                       | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
  cls.seq_relationship.weight                | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
  cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
  cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
  cls.seq_relationship.bias                  | UNEXPECTED |  | 
  cls.predictions.decoder.weight             | UNEXPECTED |  | 
  
  Notes:
  - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Mode: Testing ORIGINAL float model from [/home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/model_bin/bert.pt]
  Running evaluation on device: cuda...
  Test Loss:  0.18, Test Acc: 94.34%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9368    0.9340    0.9354      1000
         realty     0.9448    0.9580    0.9513      1000
         stocks     0.9119    0.9010    0.9064      1000
      education     0.9530    0.9730    0.9629      1000
        science     0.9170    0.9060    0.9115      1000
        society     0.9363    0.9400    0.9381      1000
       politics     0.9271    0.9160    0.9215      1000
         sports     0.9802    0.9890    0.9846      1000
           game     0.9659    0.9630    0.9644      1000
  entertainment     0.9598    0.9540    0.9569      1000
  
       accuracy                         0.9434     10000
      macro avg     0.9433    0.9434    0.9433     10000
   weighted avg     0.9433    0.9434    0.9433     10000
  
  Confusion Matrix...
  [[934  16  28   2   9   1   7   1   1   1]
   [  8 958   9   1   4   5   6   2   1   6]
   [ 41  13 901   0  19   0  22   1   2   1]
   [  2   2   1 973   1  10   4   0   0   7]
   [  1   3  20   7 906  15  13   2  22  11]
   [  4   8   3  14   6 940  14   0   3   8]
   [  5   8  22  13  10  22 916   1   0   3]
   [  0   3   1   0   1   2   1 989   0   3]
   [  1   1   1   2  26   3   1   2 963   0]
   [  1   2   2   9   6   6   4  11   5 954]]
  Time usage: 0:00:03
  ```

- 2  使用 cpu 量化 bert 后跑测试集得到 ACC 为 94.30 %

  ```zsh
  [Device] 使用 CPU -> 启用 CPU + INT8 动态量化
  Loading test dataset for [BERT] on [toutiao]...
  Loading test.txt: 10000it [00:00, 43389.83it/s]
  Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 69561.34it/s]
  [transformers] BertModel LOAD REPORT from: /home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/data/bert_pretrain
  Key                                        | Status     |  | 
  -------------------------------------------+------------+--+-
  cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
  cls.predictions.decoder.weight             | UNEXPECTED |  | 
  cls.predictions.bias                       | UNEXPECTED |  | 
  cls.seq_relationship.bias                  | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
  cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
  cls.seq_relationship.weight                | UNEXPECTED |  | 
  
  Notes:
  - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Mode: Testing QUANTIZED model from [/home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/model_bin/bert_int8.pt]
  Running evaluation on device: cpu...
  Test Loss:  0.18, Test Acc: 94.30%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9349    0.9340    0.9345      1000
         realty     0.9458    0.9590    0.9523      1000
         stocks     0.9119    0.9010    0.9064      1000
      education     0.9493    0.9740    0.9615      1000
        science     0.9162    0.9080    0.9121      1000
        society     0.9379    0.9370    0.9375      1000
       politics     0.9252    0.9150    0.9201      1000
         sports     0.9812    0.9900    0.9856      1000
           game     0.9678    0.9620    0.9649      1000
  entertainment     0.9586    0.9500    0.9543      1000
  
       accuracy                         0.9430     10000
      macro avg     0.9429    0.9430    0.9429     10000
   weighted avg     0.9429    0.9430    0.9429     10000
  
  Confusion Matrix...
  [[934  17  29   2   8   1   6   1   1   1]
   [  8 959   9   2   3   4   6   2   1   6]
   [ 41  11 901   1  20   0  23   1   1   1]
   [  2   2   0 974   1  10   4   0   0   7]
   [  1   3  20   7 908  14  13   2  21  11]
   [  5   8   3  14   7 937  14   0   3   9]
   [  5   8  22  14  11  22 915   0   0   3]
   [  0   3   1   0   0   2   1 990   0   3]
   [  1   1   1   2  27   3   1   2 962   0]
   [  2   2   2  10   6   6   6  11   5 950]]
  Time usage: 0:00:50
  ```

- 3  使用 gpu 量化 bert 后跑测试集得到 ACC 为 94.21 %

  ```zsh
  [Device] 使用 GPU: NVIDIA GeForce RTX 4070 SUPER (CC 8.9) -> 启用 GPU + FP8 动态量化
  Loading test dataset for [BERT] on [toutiao]...
  Loading test.txt: 10000it [00:00, 44006.09it/s]
  Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 99211.52it/s]
  [transformers] BertModel LOAD REPORT from: /home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/data/bert_pretrain
  Key                                        | Status     |  | 
  -------------------------------------------+------------+--+-
  cls.predictions.decoder.weight             | UNEXPECTED |  | 
  cls.predictions.bias                       | UNEXPECTED |  | 
  cls.seq_relationship.weight                | UNEXPECTED |  | 
  cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
  cls.seq_relationship.bias                  | UNEXPECTED |  | 
  cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
  cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
  
  Notes:
  - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Mode: Testing QUANTIZED model from [/home/disorient/Projects/learning-AI-Projects/nlp-learning/toutiao/model_bin/bert_fp8.pt]
  Running evaluation on device: cuda...
  Test Loss:  0.18, Test Acc: 94.21%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9339    0.9320    0.9329      1000
         realty     0.9409    0.9560    0.9484      1000
         stocks     0.9119    0.9000    0.9059      1000
      education     0.9521    0.9730    0.9624      1000
        science     0.9152    0.9060    0.9106      1000
        society     0.9370    0.9370    0.9370      1000
       politics     0.9278    0.9130    0.9204      1000
         sports     0.9792    0.9890    0.9841      1000
           game     0.9640    0.9640    0.9640      1000
  entertainment     0.9577    0.9510    0.9543      1000
  
       accuracy                         0.9421     10000
      macro avg     0.9420    0.9421    0.9420     10000
   weighted avg     0.9420    0.9421    0.9420     10000
  
  Confusion Matrix...
  [[932  18  29   2   9   0   7   1   1   1]
   [  9 956   9   1   4   5   6   2   2   6]
   [ 41  13 900   0  21   0  21   1   2   1]
   [  2   2   1 973   1  10   4   0   0   7]
   [  1   3  20   7 906  14  13   2  23  11]
   [  4   9   3  15   6 937  13   0   3  10]
   [  6   9  21  13  11  23 913   1   0   3]
   [  1   3   1   0   1   2   0 989   0   3]
   [  1   1   1   2  25   3   1   2 964   0]
   [  1   2   2   9   6   6   6  12   5 951]]
  Time usage: 0:00:03
  ```

## 2  一些量化中的问题

- 1  为什么量化后精度几乎没有下降（仅降 0.04% ~ 0.13%）？

  - 1.1  **动态量化（Dynamic Quantization）的保真度极高**：它在推理时，每一批次（Batch）的激活值都是实时根据当前输入动态计算缩放因子的，因此几乎不会出现类似静态量化那样由于校准集分布偏差导致的截断溢出。

  - 1.2  **BERT 模型的参数冗余**：预训练 Transformer 的注意力权重经过 LayerNorm 归一化后分布非常规则，对 8 位离散化有极强的鲁棒性。

- 2  为什么 INT8 的准确率略高于 FP8？

  - 2.1  尾数精度（Mantissa）的差异：

    - 2.1.1  INT8 是均匀线性量化，在对称区间内拥有整整 256 个等间距格点。

    - 2.1.2  FP8 (E4M3) 是浮点结构（1 位符号 + 4 位指数 + 3 位尾数）。虽然它的动态范围比 INT8 更宽，但其尾数有效精度只有 3 位（相当于局部有效数字只有 8 档）。在数据分布平稳、没有极端离群值（Outliers）的轻量文本分类任务中，INT8 的局部线性采样分辨率其实优于 FP8。

  - 2.2  样本波动本质：10,000 个测试样本中，INT8 只比 FP8 多答对了 9 个样本（9430 vs 9421），这在统计学上属于正常的数值扰动区间。

- 3  耗时表现解析（Float 3s vs CPU 50s vs GPU FP8 3s）

  - 3.1  终端计时精度太粗糙：

    - 3.1.1  当前输出使用的计时格式是 Time usage: 0:00:03（秒级向下取整）。

    - 3.1.2  果 Float 模型跑了 3.8 秒，FP8 跑了 3.1 秒，终端打印出来的结果都会是 0:00:03，看不出毫秒级的差异。

  - 3.2  文本序列较短，计算尚未成为绝对瓶颈：

    - 3.2.1  配置是 pad_size = 32，属于典型的短文本推理。

    - 3.2.2  在测试 10,000 条样本时，大部分耗时消耗在 Python 的 build_iterator 数据解包、CPU 内存向 GPU 显存传输（H2D Transfer）以及分类层开销上，GPU Tensor Core 内部的矩阵乘法（GEMM）耗时占比其实不高。

  - 3.3  缺少编译融合（Eager 模式的额外开销）：

    - 3.3.1  在 PyTorch Eager（默认解释器）模式下，动态量化需要频繁调用额外的动态缩放 Kernel（每次矩阵乘法前先将 FP16/FP32 张量临时量化为 FP8，乘完后再反量化）。

    - 3.3.2  这些额外的 Kernel Launch 开销抵消了 FP8 Tensor Core 带来的计算红利。

    - 3.3.3  如果想让 FP8 展现出明显优势，通常需要开启图模式编译优化：

      ```python
      model = torch.compile(model, mode="max-autotune")
      ```

      - torch.compile 会将量化/反量化节点与矩阵乘法算子直接融合为一个 CUDA 内核，才能真正激发 GPU 的极限吞吐。
