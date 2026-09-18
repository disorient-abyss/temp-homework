# 1  项目文件结构

- phone-price-model/
  ├── data/
  │   ├── 手机价格预测.csv       # 原始数据集 (2000 条样本，20 维特征)
  │   └── best_model.pth         # 训练得到的历史最优模型权重
  ├── price_classify.py          # 数据读取、标准化与 PyTorch Dataset 封装
  ├── phone_price_model.py       # MLP 分类神经网络定义
  ├── train.py                   # Trainer 训练大类封装 (含训练、验证与权重持久化)
  ├── test.py                    # 主程序入口 (支持 train / eval 模式)
  └── README.md                  # 项目技术文档



# 2  构建数据集

- 数据共有 2000 条, 其中 1600 条数据作为训练集, 400 条数据用作测试集。 我们使用 sklearn 的数据集划分工作来完成。 并使用 PyTorch 的 TensorDataset 来将数据集构建为 Dataset 对象，方便构造数据集加载对象。



- 代码详见 pirce_classify.py



# 3  构建分类网络模型

- 我们构建的用于手机价格分类的模型叫做全连接神经网络。它主要由两个个线性层来构建，根据奥卡姆剃刀定律，针对小样本表格数据，摒弃深层臃肿网络，在每个线性层后，我们使用的是 ReLU 激活函数。

- 通过将 20 维特征紧凑压缩到 16 与 8 维，迫使网络重点关注对价格起决定性作用的特征（如 RAM、电池容量），配合双层 Dropout 在测试时起到类似于模型集成的平滑效果。



- 代码详见 phone_class_model.py



# 4  构建训练大类

- 封装了高内聚的 Trainer 训练大类，负责生命周期管理：
  
  - 设备自动检测：自动优选 CUDA $\to$ MPS $\to$ CPU 硬件。
  - 显存与吞吐优化：单轮训练中累计指标完全留在 GPU 端张量（loss.detach() * len(y)），每个 Epoch 仅向 CPU 回拉一次标量。
  - 最优权重归档：在验证准确率创新高时，自动检查并安全创建目标目录，保存 state_dict。
  - 安全反序列化：评估阶段加载权重启用 weights_only=True，符合现代 PyTorch 安全规范。



- 代码详见 train.py



# 5  统一训练与闭卷评估流水线

- 提供可复现性配置（全局锁定 torch、numpy、cuda 随机种子）。

- 内置两种运行模式：
  
    1. run_train()：端到端数据加载、初始化并启动 50 轮训练。
  
    2. run_evaluate()：脱离训练集，独立拉起空模型并载入本地持久化权重，对 400 条未见样本执行闭卷性能评测。



- 代码详见 test.py


