# 1  fasttext 快速基线模型的实现过程

- 1  准备基础数据

  - 1.1  class.txt( 分类信息 )

  - 1.2  train.txt( 训练集 )

  - 1.3  dev.txt( 验证集 )

  - 1.4  test.txt( 测试集 )

- 2   预处理 fasttext 专用训练数据

  - 2.1  通过 preprocess.py 文件加载 1 中的数据

  - 2.2  选定 class.txt 的分类结果与 preprocess 处理分词方法 

    - 2.2.1  jieba 分词

    - 2.2.2  char 分词

    - ...

  - 2.3  生成 fasttest 专用训练数据

    - 2.3.1  train_fast.txt ( fasttest 专用训练集)
    - 2.3.2  dev_fast.txt ( fasttest 专用验证集)
    - 2.3.3  test_fast.txt( fasttest 专用测试集)

- 3  训练 fasttext 模型并评估保存

  - 3.1  通过 fast_text_baseline.py 加载 2 中 fasttest 专用训练数据

  - 3.2  进行 fasttext 有监督训练，返回训练模型，需要以下参数

    - 3.2.1  input 的训练数据

    - 3.2.2  autotuneValidationFile 自动验证集数据

    - 3.2.3  autotuneDuration 自动调参时间（单位 s ）

    - 3.2.4  wordNgrams  词 n-gram 的最大长度

      - 3.2.4.1  `wordNgrams=1` 表示只使用单个词（unigram）

      - 3.2.4.2  `wordNgrams=2` 表示同时考虑相邻两个词的组合（bigram）

    - 3.2.5  verbose 日志输出级别。

    - 3.2.6  thread 训练时使用的线程数

  - 3.3  评估训练出的模型并返回 `n_samples`, `precision` 以及 `recall`

    - 3.3.1  `n_samples`（样本数量）

    - 3.3.2  `precision`（精确率 / 查准率）

    - 3.3.3  `recall`（召回率 / 查全率）

  - 3.4  保存数据模型二进制文件

- 4  训练的 fasttext 模型评估结果

  ```zsh
  正在评估测试集...
  ----------------------------------------
  测试样本总数: 10000
  Top-1 准确率 (Accuracy): 91.98%
  Top-1 召回率 (Recall@1): 91.98%
  ----------------------------------------
  ```

# 2  fasttext 快速基线模型的测试用例

- 1  通过 Flask 后端框架模拟数据传入流程

  - 1.1  通过 json 传入以下内容

    ```json
      {
          "uid": "AI-3-20260411", 
          "text": "公共英语(PETS)写作中常见的逻辑词汇总"
      }
    ```

  - 1.2  启动 Flask 后端服务

    ```zsh
    (score-cast) [disorient@idesktop]~/Projects/learning-AI-Projects/nlp-learning/信息自动投递项目% gunicorn -w 1 -b 0.0.0.0:5000 app:app
    [2026-10-19 17:22:49 +0800] [37754] [INFO] Starting gunicorn 26.2.0
    [2026-10-19 17:22:49 +0800] [37754] [INFO] Listening at: http://0.0.0.0:5000 (37754)
    [2026-10-19 17:22:49 +0800] [37754] [INFO] Using worker: sync
    [2026-10-19 17:22:49 +0800] [37755] [INFO] Booting worker with pid: 37755
    [2026-10-19 17:22:49 +0800] [37754] [INFO] Control socket listening at /run/user/1000/gunicorn.ctl
    停用词加载完成，共 734 个词。
    正在加载 FastText 模型...
    FastText 模型实例化完毕。
    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.247 seconds.
    Prefix dict has been built successfully.
    ```

  - 1.3  传入该 json 第 1 次得到

    ```zsh
    状态码: 200
    输入文本: 公共英语(PETS)写作中常见的逻辑词汇总
    返回内容 (JSON): {'code': 200, 'data': {'label': 'education', 'probability': 0.9978, 'uid': 'AI-3-20260411'}, 'msg': 'success'}
    单条样本网络+推理总耗时: 249.83 ms
    ```

  - 1.4  传入该 json 第 2 次得到

    ```zsh
    状态码: 200
    输入文本: 公共英语(PETS)写作中常见的逻辑词汇总
    返回内容 (JSON): {'code': 200, 'data': {'label': 'education', 'probability': 0.9978, 'uid': 'AI-3-20260411'}, 'msg': 'success'}
    单条样本网络+推理总耗时: 1.61 ms
    ```
