# 1  开关参数梯度计算的区别

- 1  开启 param.requires_grad = True 开启参数反向传播后，模型的训练速度

  ```python
  # 此处用于调节是否将 BERT 纳入微调训练（置为 True 进行端到端全量微调；置为 True 则冻结骨干只训练分类头）
  for param in self.bert.parameters():
       param.requires_grad = True 
  ```

  ```zsh
  Starting training on device: cuda...
  Epoch [1/3]
  Training Epoch 1:  14%|██████████▍                                                               | 199/1407 [00:27<02:47,  7.21it/s]Iter:    200, Train Loss:  0.22, Train Acc: 92.97%, Val Loss:  0.27, Val Acc: 91.65%, Time: 0:00:31 *
  Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:58<02:19,  7.21it/s]Iter:    400, Train Loss:  0.25, Train Acc: 93.75%, Val Loss:  0.25, Val Acc: 92.43%, Time: 0:01:02 *
  Training Epoch 1:  43%|███████████████████████████████▌                                          | 599/1407 [01:30<01:54,  7.03it/s]Iter:    600, Train Loss:  0.28, Train Acc: 88.28%, Val Loss:  0.24, Val Acc: 92.20%, Time: 0:01:34 *
  Training Epoch 1:  57%|██████████████████████████████████████████                                | 799/1407 [02:01<01:26,  7.05it/s]Iter:    800, Train Loss:  0.16, Train Acc: 95.31%, Val Loss:  0.22, Val Acc: 92.95%, Time: 0:02:06 *
  Training Epoch 1:  71%|████████████████████████████████████████████████████▌                     | 999/1407 [02:33<00:58,  7.03it/s]Iter:   1000, Train Loss:  0.15, Train Acc: 92.97%, Val Loss:  0.22, Val Acc: 93.06%, Time: 0:02:37 *
  Training Epoch 1:  85%|██████████████████████████████████████████████████████████████▏          | 1199/1407 [03:05<00:29,  7.04it/s]Iter:   1200, Train Loss:  0.17, Train Acc: 94.53%, Val Loss:  0.22, Val Acc: 92.83%, Time: 0:03:09 *
  Training Epoch 1:  99%|████████████████████████████████████████████████████████████████████████▌| 1399/1407 [03:37<00:01,  7.14it/s]Iter:   1400, Train Loss:  0.12, Train Acc: 94.53%, Val Loss:  0.20, Val Acc: 93.99%, Time: 0:03:41 *
  Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████| 1407/1407 [03:41<00:00,  6.35it/s]
  ```

- 2  开启 param.requires_grad = False 禁止参数反向传播后，模型的训练速度

  ```python
  # 此处用于调节是否将 BERT 纳入微调训练（置为 False 进行端到端全量微调；置为 False 则冻结骨干只训练分类头）
  for param in self.bert.parameters():
       param.requires_grad = False 
  ```

  ```zsh
  Starting training on device: cuda...
  Epoch [1/3]
  Training Epoch 1:  14%|██████████▎                                                               | 197/1407 [00:07<00:47, 25.54it/s]Iter:    200, Train Loss:  2.12, Train Acc: 37.50%, Val Loss:  2.07, Val Acc: 51.50%, Time: 0:00:11 *
  Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:18<00:39, 25.44it/s]Iter:    400, Train Loss:  1.86, Train Acc: 50.00%, Val Loss:  1.85, Val Acc: 58.01%, Time: 0:00:22 *
  Training Epoch 1:  42%|███████████████████████████████▍                                          | 597/1407 [00:29<00:31, 25.42it/s]Iter:    600, Train Loss:  1.78, Train Acc: 50.78%, Val Loss:  1.68, Val Acc: 61.96%, Time: 0:00:33 *
  Training Epoch 1:  57%|██████████████████████████████████████████                                | 799/1407 [00:41<00:24, 25.02it/s]Iter:    800, Train Loss:  1.57, Train Acc: 66.41%, Val Loss:  1.54, Val Acc: 64.19%, Time: 0:00:45 *
  Training Epoch 1:  71%|████████████████████████████████████████████████████▍                     | 997/1407 [00:52<00:16, 24.88it/s]Iter:   1000, Train Loss:  1.52, Train Acc: 60.16%, Val Loss:  1.43, Val Acc: 66.96%, Time: 0:00:56 *
  Training Epoch 1:  85%|██████████████████████████████████████████████████████████████▏          | 1199/1407 [01:04<00:08, 24.85it/s]Iter:   1200, Train Loss:  1.38, Train Acc: 64.84%, Val Loss:  1.34, Val Acc: 68.43%, Time: 0:01:08 *
  Training Epoch 1:  99%|████████████████████████████████████████████████████████████████████████▍| 1397/1407 [01:15<00:00, 24.34it/s]Iter:   1400, Train Loss:  1.40, Train Acc: 62.50%, Val Loss:  1.26, Val Acc: 69.27%, Time: 0:01:19 *
  Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████| 1407/1407 [01:19<00:00, 17.76it/s]
  ```

- 3  打印所有的参数名称 name

  ```python
  for name, param in self.bert.named_parameters():
    param.requires_grad = True
    print(name)
  ```

  ```zsh
  embeddings.word_embeddings.weight
  embeddings.position_embeddings.weight
  embeddings.token_type_embeddings.weight
  embeddings.LayerNorm.weight
  embeddings.LayerNorm.bias
  encoder.layer.0.attention.self.query.weight
  encoder.layer.0.attention.self.query.bias
  encoder.layer.0.attention.self.key.weight
  encoder.layer.0.attention.self.key.bias
  encoder.layer.0.attention.self.value.weight
  encoder.layer.0.attention.self.value.bias
  encoder.layer.0.attention.output.dense.weight
  encoder.layer.0.attention.output.dense.bias
  encoder.layer.0.attention.output.LayerNorm.weight
  encoder.layer.0.attention.output.LayerNorm.bias
  encoder.layer.0.intermediate.dense.weight
  encoder.layer.0.intermediate.dense.bias
  encoder.layer.0.output.dense.weight
  encoder.layer.0.output.dense.bias
  encoder.layer.0.output.LayerNorm.weight
  encoder.layer.0.output.LayerNorm.bias
  ......
  encoder.layer.11.attention.self.query.weight
  encoder.layer.11.attention.self.query.bias
  encoder.layer.11.attention.self.key.weight
  encoder.layer.11.attention.self.key.bias
  encoder.layer.11.attention.self.value.weight
  encoder.layer.11.attention.self.value.bias
  encoder.layer.11.attention.output.dense.weight
  encoder.layer.11.attention.output.dense.bias
  encoder.layer.11.attention.output.LayerNorm.weight
  encoder.layer.11.attention.output.LayerNorm.bias
  encoder.layer.11.intermediate.dense.weight
  encoder.layer.11.intermediate.dense.bias
  encoder.layer.11.output.dense.weight
  encoder.layer.11.output.dense.bias
  encoder.layer.11.output.LayerNorm.weight
  encoder.layer.11.output.LayerNorm.bias
  pooler.dense.weight
  pooler.dense.bias
  Starting training on device: cuda...
  ```

# 2  锁死指定层的参数

- 1  锁死 embeddings 参数观察训练耗时后，变化很小

  ```python
  # 锁死 embeddings 层的参数不让参与反向传播
  for name, param in self.bert.embeddings.named_parameters():
      print(name)
      param.requires_grad = False
  ```

  ```sh
  word_embeddings.weight
  position_embeddings.weight
  token_type_embeddings.weight
  LayerNorm.weight
  LayerNorm.bias
  Starting training on device: cuda...
  Epoch [1/3]
  Training Epoch 1:  14%|██████████▍                                                               | 199/1407 [00:27<02:46,  7.25it/s]Iter:    200, Train Loss:  0.23, Train Acc: 93.75%, Val Loss:  0.27, Val Acc: 91.47%, Time: 0:00:31 *
  Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:58<02:20,  7.19it/s]Iter:    400, Train Loss:  0.26, Train Acc: 93.75%, Val Loss:  0.25, Val Acc: 92.10%, Time: 0:01:02 *
  Training Epoch 1:  43%|███████████████████████████████▌                                          | 599/1407 [01:29<01:50,  7.29it/s]Iter:    600, Train Loss:  0.27, Train Acc: 92.97%, Val Loss:  0.25, Val Acc: 92.02%, Time: 0:01:33 *
  Training Epoch 1:  43%|███████████████████████████████▊                                          | 604/1407 [01:33<05:43,  2.34it/s]^
  ```

- 2  锁死所有的 .weight 参数后

  ```python
  # 锁死全连接层中的 .weight 部分参数不让参与反向传播
  for name, param in self.bert.named_parameters():
      if(name.endswith("weight")):
          print(name)
          param.requires_grad = False
  ```

  ```zsh
  embeddings.word_embeddings.weight
  embeddings.position_embeddings.weight
  embeddings.token_type_embeddings.weight
  embeddings.LayerNorm.weight
  encoder.layer.0.attention.self.query.weight
  encoder.layer.0.attention.self.key.weight
  encoder.layer.0.attention.self.value.weight
  encoder.layer.0.attention.output.dense.weight
  encoder.layer.0.attention.output.LayerNorm.weight
  encoder.layer.0.intermediate.dense.weight
  encoder.layer.0.output.dense.weight
  encoder.layer.0.output.LayerNorm.weight
  ......
  encoder.layer.11.attention.self.query.weight
  encoder.layer.11.attention.self.key.weight
  encoder.layer.11.attention.self.value.weight
  encoder.layer.11.attention.output.dense.weight
  encoder.layer.11.attention.output.LayerNorm.weight
  encoder.layer.11.intermediate.dense.weight
  encoder.layer.11.output.dense.weight
  encoder.layer.11.output.LayerNorm.weight
  pooler.dense.weight
  Starting training on device: cuda...
  Epoch [1/3]
  Training Epoch 1:  14%|██████████▍                                                               | 198/1407 [00:18<01:53, 10.69it/s]Iter:    200, Train Loss:  1.69, Train Acc: 64.84%, Val Loss:  1.68, Val Acc: 60.26%, Time: 0:00:22 *
  Training Epoch 1:  28%|████████████████████▉                                                     | 399/1407 [00:40<01:34, 10.61it/s]^
  ```

- 3  封锁 bert 第 1, 3, 5 层 Encoder Block 的参数不让参与反向传播

  ```python
  index_layer = ["1", "3", "5"]
  for name, param in self.bert.encoder.named_parameters():
      parts = name.split(".")
      if len(parts) > 1 and parts[0] == "layer":
          new_index = parts[1]
          if new_index in index_layer:
              print(name)
              param.requires_grad = False
  ```

  ```zsh
  layer.1.attention.self.query.weight
  layer.1.attention.self.query.bias
  layer.1.attention.self.key.weight
  layer.1.attention.self.key.bias
  layer.1.attention.self.value.weight
  layer.1.attention.self.value.bias
  layer.1.attention.output.dense.weight
  layer.1.attention.output.dense.bias
  layer.1.attention.output.LayerNorm.weight
  layer.1.attention.output.LayerNorm.bias
  layer.1.intermediate.dense.weight
  layer.1.intermediate.dense.bias
  layer.1.output.dense.weight
  layer.1.output.dense.bias
  layer.1.output.LayerNorm.weight
  layer.1.output.LayerNorm.bias
  layer.3.attention.self.query.weight
  layer.3.attention.self.query.bias
  layer.3.attention.self.key.weight
  layer.3.attention.self.key.bias
  layer.3.attention.self.value.weight
  layer.3.attention.self.value.bias
  layer.3.attention.output.dense.weight
  layer.3.attention.output.dense.bias
  layer.3.attention.output.LayerNorm.weight
  layer.3.attention.output.LayerNorm.bias
  layer.3.intermediate.dense.weight
  layer.3.intermediate.dense.bias
  layer.3.output.dense.weight
  layer.3.output.dense.bias
  layer.3.output.LayerNorm.weight
  layer.3.output.LayerNorm.bias
  layer.5.attention.self.query.weight
  layer.5.attention.self.query.bias
  layer.5.attention.self.key.weight
  layer.5.attention.self.key.bias
  layer.5.attention.self.value.weight
  layer.5.attention.self.value.bias
  layer.5.attention.output.dense.weight
  layer.5.attention.output.dense.bias
  layer.5.attention.output.LayerNorm.weight
  layer.5.attention.output.LayerNorm.bias
  layer.5.intermediate.dense.weight
  layer.5.intermediate.dense.bias
  layer.5.output.dense.weight
  layer.5.output.dense.bias
  layer.5.output.LayerNorm.weight
  layer.5.output.LayerNorm.bias
  Starting training on device: cuda...
  Epoch [1/3]
  Training Epoch 1:   1%|▋                                                                          | 14/1407 [00:01<02:56,  7.91it/s]^
  ```
