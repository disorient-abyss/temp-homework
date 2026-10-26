# 1  从 0 训练图像识别 cnn 模型效果

- 1  通过 3 层卷积模块和池化模块压缩图像信息

  ```python
          # 卷积块 1
          self.conv1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
          self.bn1 = nn.BatchNorm2d(9)
          self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  
          # 卷积块 2
          self.conv2 = nn.Conv2d(9, 27, kernel_size=3, stride=1, padding=1)
          self.bn2 = nn.BatchNorm2d(27)
          self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
  
          # 卷积块 3: 64 -> 128 (8x8 -> 4x4)
          self.conv3 = nn.Conv2d(27, 81, kernel_size=3, padding=1)
          self.bn3 = nn.BatchNorm2d(81)
          self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
  
          # 全连接分类头
          self.linear1 = nn.Linear(81 * 4 * 4, 324)
          self.dropout = nn.Dropout(p=0.15)
          self.linear2 = nn.Linear(324, 81)
          self.linear3 = nn.Linear(81, 27)
          self.out = nn.Linear(27, 10)
  ```

- 2  得到 image_classification.pt 模型最终效果

  ```zsh
  Device: cuda
  Final Test Loss: 0.5429, Test Acc: 81.19%
  
  Precision, Recall and F1-Score:
                precision    recall  f1-score   support
  
      airplane     0.8146    0.8570    0.8353      1000
    automobile     0.8924    0.9210    0.9065      1000
          bird     0.7852    0.7020    0.7413      1000
           cat     0.6484    0.6140    0.6307      1000
          deer     0.7785    0.8260    0.8016      1000
           dog     0.6995    0.7170    0.7081      1000
          frog     0.8560    0.8680    0.8620      1000
         horse     0.8462    0.8470    0.8466      1000
          ship     0.9119    0.9000    0.9059      1000
         truck     0.8784    0.8670    0.8727      1000
  
      accuracy                         0.8119     10000
     macro avg     0.8111    0.8119    0.8111     10000
  weighted avg     0.8111    0.8119    0.8111     10000
  
  Confusion Matrix:
  [[857  11  25  12  12   4   3  10  38  28]
   [  6 921   2   4   1   0   4   1   7  54]
   [ 67   1 702  37  62  54  44  20   7   6]
   [ 19   4  49 614  52 163  51  31   8   9]
   [ 11   2  37  29 826  25  22  44   3   1]
   [  8   3  27 153  43 717  12  33   1   3]
   [  7   5  24  48  20  18 868   4   4   2]
   [ 10   4  20  33  40  38   2 847   1   5]
   [ 42  21   3   5   3   4   4   6 900  12]
   [ 25  60   5  12   2   2   4   5  18 867]]
  Time usage: 00:00:00
  ```

# 2  将 resnet-18 模型通过 cifar10 数据集合进行迁移学习

- 1  将模型的 512 维输出结果投射到 cifar10 的 10 分类当中

  ```python
        # 模型结构参数（ResNet-18 最终输出维度为 512，CIFAR-10 为 10 类）
        self.hidden_size = 512
        self.num_classes = 10
  ```

- 2  得到 resnet-18-cifar10.pt 模型最终效果，得到 96.08% 的 test 集的 acc

  ```zsh
  Device: cuda
  Final Test Loss: 0.2173, Test Acc: 96.08%
  
  Precision, Recall and F1-Score:
                precision    recall  f1-score   support
  
      airplane     0.9589    0.9800    0.9693      1000
    automobile     0.9714    0.9840    0.9776      1000
          bird     0.9682    0.9440    0.9559      1000
           cat     0.9232    0.9020    0.9125      1000
          deer     0.9550    0.9760    0.9654      1000
           dog     0.9245    0.9300    0.9272      1000
          frog     0.9713    0.9820    0.9766      1000
         horse     0.9808    0.9720    0.9764      1000
          ship     0.9798    0.9690    0.9744      1000
         truck     0.9748    0.9690    0.9719      1000
  
      accuracy                         0.9608     10000
     macro avg     0.9608    0.9608    0.9607     10000
  weighted avg     0.9608    0.9608    0.9607     10000
  
  Confusion Matrix:
  [[980   1   3   1   0   0   0   1  10   4]
   [  0 984   0   0   0   0   0   0   2  14]
   [ 12   0 944   8  12   9  12   2   1   0]
   [  4   1   9 902  14  53  10   4   3   0]
   [  0   0   4   7 976   4   3   6   0   0]
   [  1   0   4  46  10 930   4   5   0   0]
   [  3   0   3   8   1   1 982   1   0   1]
   [  1   0   3   5   9   9   0 972   1   0]
   [ 18   3   4   0   0   0   0   0 969   6]
   [  3  24   1   0   0   0   0   0   3 969]]
  Time usage: 0:00:02
  ```

  # 3  将 resnet-34 模型通过 cifar10 数据集合进行迁移学习

- 1  将模型的 512 维输出结果投射到 cifar10 的 10 分类当中

  ```python
        # 模型结构参数（ResNet-34 最终输出维度为 512，CIFAR-10 为 10 类）
        self.hidden_size = 512
        self.num_classes = 10
  ```

- 2  得到 resnet-34-cifar10.pt 模型最终效果，得到 97.24% 的 test 集的 acc

  ```zsh
  Device: cuda
  Final Test Loss: 0.1769, Test Acc: 97.24%
  
  Precision, Recall and F1-Score:
                precision    recall  f1-score   support
  
      airplane     0.9790    0.9780    0.9785      1000
    automobile     0.9762    0.9850    0.9806      1000
          bird     0.9808    0.9690    0.9748      1000
           cat     0.9304    0.9490    0.9396      1000
          deer     0.9655    0.9800    0.9727      1000
           dog     0.9612    0.9420    0.9515      1000
          frog     0.9870    0.9870    0.9870      1000
         horse     0.9889    0.9780    0.9834      1000
          ship     0.9791    0.9850    0.9821      1000
         truck     0.9769    0.9710    0.9739      1000
  
      accuracy                         0.9724     10000
     macro avg     0.9725    0.9724    0.9724     10000
  weighted avg     0.9725    0.9724    0.9724     10000
  
  Confusion Matrix:
  [[978   3   3   3   0   1   0   0   9   3]
   [  0 985   0   0   0   0   0   0   2  13]
   [  5   0 969   9   6   3   6   0   2   0]
   [  1   0   4 949  10  26   4   2   3   1]
   [  0   0   3   8 980   3   1   5   0   0]
   [  0   0   3  45   6 942   1   3   0   0]
   [  2   0   3   5   1   1 987   0   0   1]
   [  1   0   2   1  12   4   1 978   0   1]
   [ 10   0   0   0   0   0   0   1 985   4]
   [  2  21   1   0   0   0   0   0   5 971]]
  Time usage: 0:00:03
  ```

> [!Summary] 结论与分析
>
> - **1  迁移学习带来显著跃升：** 从自定义轻量 CNN 切换到 ResNet-18 预训练模型后，准确率从 **81.19% 跃升至 96.08%**（提升 14.89%），损失值大幅下降（0.5429 $\rightarrow$ 0.2173）。预训练权重提取的通用特征极大地弥补了 CIFAR-10 数据规模较小带来的泛化瓶颈。
>
> -  **2  更深网络收益稳定：** 相比 ResNet-18，ResNet-34 的准确率进一步提升至 **97.24%**，损失降至 0.1769，且全类别 F1-Score 均达到 0.93 以上，体现出更深网络在特征细分上的优势。
>
> - **3  混淆模式具有高度一致性：**
>
>   - 3.1  无论是自定义网络还是 ResNet，**猫（Cat）与狗（Dog）** 均为分类难度最高的类别（自定义 CNN 中猫与狗互错各超 150 例；ResNet-18 中仍有 46 例狗误判为猫、53 例猫误判为狗；ResNet-34 中该混淆降至 45 例与 26 例）。
>
>   - 3.2  车辆类类别（Automobile / Truck / Ship）特征显著，在各模型中均保持了最高区间的准确率与召回率。
