替换 bert 模型为 macbert

- 1  将模型替换为 macbert

  - 1.1  对于投满分项目来说，macbert 模型测试集准确率为 94.74%

  - 1.2  训练出来的模型大小与 roberta 相当

  ```zsh
  399596 -rw-r--r--. 1 disorient disorient  409182765 10月26日 22:54 macbert.pt
   38936 -rw-r--r--. 1 disorient disorient   39869753 10月24日 17:28 albert.pt
  399596 -rw-r--r--. 1 disorient disorient  409182765 10月24日 12:09 roberta.pt
  399580 -rw-r--r--. 1 disorient disorient  409169856 10月24日 10:30 bert.pt
  149472 -rw-r--r--. 1 disorient disorient  153055243 10月23日 23:39 bert_int8.pt
  149068 -rw-r--r--. 1 disorient disorient  152643571 10月23日 23:27 bert_fp8.pt
  ```

  ```zsh
  Test Loss:  0.18, Test Acc: 94.74%
  Precision, Recall and F1-Score...
                 precision    recall  f1-score   support
  
        finance     0.9454    0.9350    0.9402      1000
         realty     0.9617    0.9550    0.9584      1000
         stocks     0.9268    0.8990    0.9127      1000
      education     0.9726    0.9580    0.9652      1000
        science     0.9372    0.9110    0.9239      1000
        society     0.9501    0.9320    0.9409      1000
       politics     0.8931    0.9520    0.9216      1000
         sports     0.9899    0.9790    0.9844      1000
           game     0.9680    0.9690    0.9685      1000
  entertainment     0.9336    0.9840    0.9581      1000
  
       accuracy                         0.9474     10000
      macro avg     0.9478    0.9474    0.9474     10000
   weighted avg     0.9478    0.9474    0.9474     10000
  
  Confusion Matrix...
  [[935  10  34   3   4   3  10   1   0   0]
   [  8 955   8   1   2  12   7   2   0   5]
   [ 31  16 899   0  22   0  29   0   2   1]
   [  2   0   1 958   3  12  14   0   0  10]
   [  1   3  11   3 911   7  21   2  26  15]
   [  2   5   1  11   3 932  27   0   2  17]
   [  7   1  11   6   8   9 952   0   0   6]
   [  2   1   3   0   0   1   4 979   0  10]
   [  0   0   2   1  17   2   1   2 969   6]
   [  1   2   0   2   2   3   1   3   2 984]]
  Time usage: 0:00:03
  ```
