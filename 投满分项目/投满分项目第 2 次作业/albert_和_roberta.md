# 1  替换 bert 模型为 albert

- 1  将模型替换为 albert

  - 1.1  对于投满分项目来说，测试集准确率下降到 92.61%，并且 epoch 跑满 4 轮依旧没有提升 

  - 1.2  训练出来的模型大小减小了将近 1 个数量级

  ```zsh
  38936 -rw-r--r--. 1 disorient disorient   39869753 10月24日 17:28 albert.pt
  399596 -rw-r--r--. 1 disorient disorient  409182765 10月24日 12:09 roberta.pt
  399580 -rw-r--r--. 1 disorient disorient  409169856 10月24日 10:30 bert.pt
  149472 -rw-r--r--. 1 disorient disorient  153055243 10月23日 23:39 bert_int8.pt
  149068 -rw-r--r--. 1 disorient disorient  152643571 10月23日 23:27 bert_fp8.pt
  1694056 -rw-r--r--. 1 disorient disorient 1734712369 10月19日 11:59 toutiao_fasttext_jieba_1792382370.bin
  ```

  ```zsh
  Starting training on device: cuda...
  Epoch [1/4]
  Training Epoch 1:  14%|██████████▍                                                               | 199/1407 [00:24<02:27,  8.19it/s]Iter:    200, Train Loss:  0.25, Train Acc: 92.97%, Val Loss:  0.37, Val Acc: 88.05%, Time: 0:00:28 *
  Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:52<02:03,  8.16it/s]Iter:    400, Train Loss:  0.43, Train Acc: 85.16%, Val Loss:  0.34, Val Acc: 89.34%, Time: 0:00:56 *
  Training Epoch 1:  43%|███████████████████████████████▌                                          | 599/1407 [01:20<01:39,  8.10it/s]Iter:    600, Train Loss:  0.34, Train Acc: 87.50%, Val Loss:  0.30, Val Acc: 90.19%, Time: 0:01:24 *
  Training Epoch 1:  57%|██████████████████████████████████████████                                | 799/1407 [01:48<01:15,  8.05it/s]Iter:    800, Train Loss:  0.18, Train Acc: 93.75%, Val Loss:  0.29, Val Acc: 90.95%, Time: 0:01:52 *
  Training Epoch 1:  71%|████████████████████████████████████████████████████▌                     | 999/1407 [02:17<00:50,  8.07it/s]Iter:   1000, Train Loss:  0.25, Train Acc: 91.41%, Val Loss:  0.27, Val Acc: 91.22%, Time: 0:02:21 *
  Training Epoch 1:  85%|██████████████████████████████████████████████████████████████▏          | 1199/1407 [02:45<00:25,  8.04it/s]Iter:   1200, Train Loss:  0.26, Train Acc: 92.19%, Val Loss:  0.26, Val Acc: 91.61%, Time: 0:02:49 *
  Training Epoch 1:  99%|████████████████████████████████████████████████████████████████████████▌| 1399/1407 [03:14<00:00,  8.03it/s]Iter:   1400, Train Loss:  0.17, Train Acc: 95.31%, Val Loss:  0.27, Val Acc: 91.35%, Time: 0:03:18 
  Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████| 1407/1407 [03:18<00:00,  7.09it/s]
  
  ......
  
  Epoch [4/4]
  Training Epoch 4:  13%|█████████▎                                                                | 178/1407 [00:22<02:33,  7.99it/s]Iter:   4400, Train Loss:  0.17, Train Acc: 95.31%, Val Loss:  0.26, Val Acc: 91.91%, Time: 0:10:26 
  Training Epoch 4:  27%|███████████████████▉                                                      | 378/1407 [00:50<02:08,  7.98it/s]Iter:   4600, Train Loss:  0.10, Train Acc: 95.31%, Val Loss:  0.30, Val Acc: 91.54%, Time: 0:10:55 
  Training Epoch 4:  41%|██████████████████████████████▍                                           | 578/1407 [01:19<01:43,  7.99it/s]Iter:   4800, Train Loss:  0.13, Train Acc: 94.53%, Val Loss:  0.27, Val Acc: 92.09%, Time: 0:11:24 
  Training Epoch 4:  55%|████████████████████████████████████████▉                                 | 778/1407 [01:48<01:18,  8.00it/s]Iter:   5000, Train Loss:  0.17, Train Acc: 96.09%, Val Loss:  0.27, Val Acc: 92.05%, Time: 0:11:52 
  Training Epoch 4:  70%|███████████████████████████████████████████████████▍                      | 978/1407 [02:16<00:53,  7.99it/s]Iter:   5200, Train Loss:  0.07, Train Acc: 97.66%, Val Loss:  0.28, Val Acc: 91.94%, Time: 0:12:21 
  Training Epoch 4:  84%|█████████████████████████████████████████████████████████████            | 1178/1407 [02:45<00:28,  8.00it/s]Iter:   5400, Train Loss:  0.10, Train Acc: 97.66%, Val Loss:  0.29, Val Acc: 91.70%, Time: 0:12:50 
  Training Epoch 4:  98%|███████████████████████████████████████████████████████████████████████▍ | 1378/1407 [03:13<00:03,  8.00it/s]Iter:   5600, Train Loss:  0.13, Train Acc: 96.88%, Val Loss:  0.29, Val Acc: 91.52%, Time: 0:13:18 
  Training Epoch 4:  98%|███████████████████████████████████████████████████████████████████████▌ | 1379/1407 [03:17<00:34,  1.23s/it]No optimization for a long time, auto-stopping...
  Training Epoch 4:  98%|███████████████████████████████████████████████████████████████████████▌ | 1379/1407 [03:17<00:04,  6.97it/s]
  Test Loss:  0.23, Test Acc: 92.61%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9533    0.8780    0.9141      1000
         realty     0.9325    0.9530    0.9426      1000
         stocks     0.8756    0.8590    0.8672      1000
      education     0.9731    0.9400    0.9563      1000
        science     0.8743    0.8970    0.8855      1000
        society     0.9116    0.9280    0.9197      1000
       politics     0.9187    0.9040    0.9113      1000
         sports     0.9695    0.9850    0.9772      1000
           game     0.9326    0.9540    0.9432      1000
  entertainment     0.9233    0.9630    0.9427      1000
  
       accuracy                         0.9261     10000
      macro avg     0.9264    0.9261    0.9260     10000
   weighted avg     0.9264    0.9261    0.9260     10000
  
  Confusion Matrix...
  [[878  13  66   1  10   9  12   3   3   5]
   [  7 953  10   1   2  12   1   3   2   9]
   [ 30  27 859   0  43   3  23   2   8   5]
   [  0   7   0 940   8  18   7   2   4  14]
   [  1   4  16   0 897  17  12   2  36  15]
   [  1   7   1  10   9 928  22   2   3  17]
   [  4   6  24  11  23  17 904   2   2   7]
   [  0   1   1   0   2   3   2 985   3   3]
   [  0   2   4   1  26   6   0   2 954   5]
   [  0   2   0   2   6   5   1  13   8 963]]
  Time usage: 0:00:04
  ```

# 2  替换 bert 模型为 roberta

- 1  将模型替换为 roberta

  - 1.1  模型对测试集准确率提升到 94.64%

  ```zsh
  Running evaluation on device: cuda...
  Test Loss:  0.17, Test Acc: 94.64%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9577    0.9280    0.9426      1000
         realty     0.9526    0.9650    0.9588      1000
         stocks     0.9014    0.9230    0.9121      1000
      education     0.9604    0.9700    0.9652      1000
        science     0.9402    0.8960    0.9176      1000
        society     0.9382    0.9410    0.9396      1000
       politics     0.9136    0.9310    0.9222      1000
         sports     0.9939    0.9820    0.9879      1000
           game     0.9499    0.9670    0.9584      1000
  entertainment     0.9581    0.9610    0.9596      1000
  
       accuracy                         0.9464     10000
      macro avg     0.9466    0.9464    0.9464     10000
   weighted avg     0.9466    0.9464    0.9464     10000
  
  Confusion Matrix...
  [[928  15  40   0   4   2  10   1   0   0]
   [  5 965   8   2   2   4   6   0   2   6]
   [ 25  13 923   0  12   1  23   0   2   1]
   [  1   1   2 970   2   9   9   0   0   6]
   [  2   1  26   4 896  14  17   0  30  10]
   [  1  11   1  13   6 941  15   0   5   7]
   [  5   4  16   9   9  22 931   0   0   4]
   [  0   2   3   1   1   2   2 982   1   6]
   [  0   0   3   2  18   4   3   1 967   2]
   [  2   1   2   9   3   4   3   4  11 961]]
  Time usage: 0:00:03
  ```
