# 1  准备数据与下载模型

- 1  通过下载的 train_medical.json 数据构建 alpaca_zh_medical.json 数据并保存在 LLaMA-Factory/data 目录下

  - 1.1  对齐 instruction 字段

    - 1.1.1  instruction

    - 1.1.2  question

    - 1.1.3  ask

    - 1.1.4  prompt

  - 2.1  对齐 input_text 字段

    - 2.1.1  input

  - 3.1  对齐 output_text 字段

    - 3.1.1  output

    - 3.1.2  answer

    - 3.1.3  response

- 2  在 LLaMA-Factory/data 目录中的 dataset_info.json 中注册登记 alpaca_zh_medical 数据

- 3  下载 Qwen3.5-4B 模型

# 2  lora 和 qlora 的后训练

- 1  进行 lora 微调

  - 1.1  修改 LLaMA-Factory/examples/train_lora 目录中的 qwen3_lora_sft.yaml 文件

  - 1.2  在 LLaMA-Factory 目录下执行命令进行 lora 微调

    ```zsh
    llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
    ```

  - 1.3  保存微调后的权重文件夹 qwen3.5-4b/lora/sft 在 LLaMA-Factory/saves 文件夹下

- 2  进行 qlora 微调

  - 1.1  修改 LLaMA-Factory/examples/train_qlora 目录中的 llama3_lora_sft_awq.yaml 文件

  - 1.2  在 LLaMA-Factory 目录下执行命令进行 qlora 微调

    ```zsh
    llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
    ```

  - 1.3  保存微调后的权重文件夹 qwen3.5-4b/qlora/sft 在 LLaMA-Factory/saves 文件夹下

# 3  运行微调后的模型

- 1  运行 lora 微调后的模型

  - 1.1  在 LLaMA-Factory/inference 目录中的 qwen3_lora_sft.yaml 注册 qwen3.5-4b 模型文件和经过 lora 微调后的 stf 权重文件

  - 1.2  运行 lora 微调后的模型

  ```zsh
  llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
  ```

  ```zsh
  User: 小孩晚上总磨牙怎么办？
  磨牙（Bruxism）是许多小孩常见的现象，大多数情况下是良性的，随着孩子年龄的增长会自行消失。但是，如果磨牙严重影响了孩子的睡眠或健康，可以尝试以下方法：
  
  1. 检查牙齿：牙齿咬合不协调或牙齿排列不齐可能是导致磨牙的原因之一。带孩子去看牙医，进行专业的检查，以确定是否需要牙齿矫正或其他治疗。
  
  2. 缓解压力：有些孩子会因为压力或焦虑而磨牙。家长应该多与孩子交流，了解他们的感受，帮助他们缓解压力。
  
  3. 饮食调整：有些食物，如巧克力、咖啡、茶等，可能会增加磨牙的频率。建议减少这些食物的摄入。
  
  4. 磨牙垫：对于磨牙严重的孩子，牙医可能会建议佩戴磨牙垫，以保护牙齿和减少磨牙带来的不适。
  
  5. 改善睡眠环境：确保孩子的睡眠环境舒适、安静，避免外界干扰。
  
  总之，磨牙通常是良性的，家长应该保持耐心，观察孩子的情况，必要时寻求专业医生的帮助。
  
  User: 需要去看医生吗？
  
  如果孩子的磨牙现象轻微，没有影响到他们的睡眠或健康，可能不需要去看医生。但是如果磨牙严重，导致孩子睡眠不安、牙齿疼痛、头痛、面部肌肉疼痛或磨损，就应该带孩子去看医生了。牙医或儿科医生可以帮助诊断磨牙的原因，并提供适当的治疗建议。
  ```

- 2  运行 qlora 微调后的模型

  - 2.1  在 LLaMA-Factory/examples/inference 目录中的 qwen3_lora_sft.yaml 注册 qwen3.5-4b 模型文件和经过 qlora 微调后的 stf 权重文件

  - 2.2  运行 qlora 微调后的模型

  ```zsh
  llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
  ```

  ```zsh
  User: 小孩晚上总磨牙怎么办？
  您好，引起磨牙的原因有：
  User: 需要去看医生吗？
  建议您带孩子到口腔门诊，检查一下牙齿，看是否有龋齿，牙病及牙结石等问题，如果有牙病及牙结石，需要及时治疗，同时检查一下微量元素，看是否有微量元素缺乏，可以服用点葡萄糖酸钙片。
  ```

# 4  使用 vllm 加速推理

- 1  调整 LLaMA-Factory/examples/inference 目录中 qwen3_lora_sft.yaml 的 infer_backend 参数为 vllm

  ```python
  infer_backend: vllm # choices: [huggingface, vllm, sglang, ktransformers]
  ```

- 2  运行 vllm 加速推理

  ```zsh
  (EngineCore_DP0 pid=23697) INFO 11-09 02:19:05 [vllm.py:957] Cudagraph is disabled under eager mode
  Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.
  User: 孩子晚上总磨牙是什么问题？
  你好，可能是缺钙或者微量元素缺乏引起的
  User: 我需要带他去医院吗？
  最好去医院检查一下
  User: 需要给他吃什么药？
  最好先确定是什么原因引起的
  ```

> [!Note] 注意
>
> - 这里 vllm 的最低版本为 0.17.0，因为这是对 qwen3.5 模型收录的最低版本

# 5  通过接入 fastAPI 调用功能

- 1 合并 lora 微调权重文件 stf

  - 1  首先修改 LLaMA-Factory/examples/merge_lora 目录下的 qwen3_lora_stf.yaml 文件

  - 2  然后通过命令合并 lora 产生的 sft 与 qwen3.5-4b 模型参数，保存在 LLaMA-Factory/saves/qwen3.5-4b_sft_merged 文件夹下

  ```zsh
  llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml 
  ```

- 2  构造服务接口

  - 1.1  主要流程

    - 1.1.1  配置与全局对象

    - 1.1.2  Pydantic 校验与数据契约

    - 1.1.3  使用 lifespan 管理模型初始化与释放

    - 1.1.4  鲁棒地拆分思考过程与正式回答并移除可能残留的终止符

    - 1.1.5  过滤掉 history 中可能损坏或空的轮次

    - 1.1.6  构建输入 Prompt：使用 conversation 关键字传参，并做异常回退兼容

    - 1.1.7  设置超参策略

    - 1.1.8  拆解思考链与实际回答

    - 1.1.9  拼接多轮对话历史

  - 1.2  启动后端服务

  ```zsh
  INFO:     Started server process [34544]
  INFO:     Waiting for application startup.
  [transformers] Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='axial'
  [transformers] Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='axial'
  [transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
  Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████| 426/426 [00:07<00:00, 54.46it/s]
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8103 (Press CTRL+C to quit)
  ```

- 3  构造测试用例

  - 2.1  测试用例代码

    ```python
    # 模拟多轮对话提问列表
    prompts = [
        "宝宝白天爱磨牙会是哪些情况呢？",
        "那需要去医院检查吗？"
    ]
    ```

  - 2.2  启用测试用例结果

    ```zsh
    ==================== 第 1 轮对话 ====================
    用户问题: 宝宝白天爱磨牙会是哪些情况呢？
    模型回答: 宝宝白天磨牙可能是由以下几个原因引起的：
    
    1. 牙齿发育：宝宝在长牙期间，牙齿可能会比较敏感，他们可能会通过磨牙来缓解不适感。
    
    2. 消化系统问题：如果宝宝有消化不良、肠胃不适等问题，也可能会通过磨牙来缓解症状。
    
    3. 精神压力：宝宝可能会因为一些环境因素、家庭因素等产生精神压力，磨牙可能是他们释放压力的一种方式。
    
    4. 睡眠问题：宝宝可能存在睡眠问题，如睡眠不足、睡眠质量差等，这些都会导致磨牙。
    
    5. 其他疾病：有些宝宝可能存在其他疾病，如癫痫、佝偻病等，这些疾病也可能导致磨牙。
    
    建议家长观察宝宝的具体情况，如是否有其他症状、是否有其他疾病等，如果磨牙情况持续存在，建议及时咨询医生，以便进行诊断和治疗。
    单轮耗时: 4.96 s
    
    ==================== 第 2 轮对话 ====================
    用户问题: 那需要去医院检查吗？
    模型回答: 如果宝宝的磨牙情况比较轻微，并且没有其他症状，可以暂时观察一下，注意调整宝宝的饮食和作息，看是否能改善。但如果磨牙情况比较频繁或严重，或者伴随有其他症状，如食欲减退、睡眠问题、情绪异常等，建议及时带宝宝去医院检查，以排除潜在的健康问题。医生会根据宝宝的具体情况，进行诊断和治疗建议。
    单轮耗时: 1.87 s
    ```
