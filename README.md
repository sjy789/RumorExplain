# 可解释谣言检测模型

本项目用于《人工智能导论》大作业中的谣言检测与判断依据生成。系统输入一段推文文本，输出二分类结果和中文判断依据：

- `0`：非谣言
- `1`：谣言

分类结果由本地融合模型确定，大语言模型仅根据分类结果和模型证据生成解释，不会修改分类标签。

## 小组成员与分工

| 姓名 | GitHub 用户名 | 主要职责 | 具体贡献 | 贡献比例 |
|---|---|---|---|---:|
| 施佳瑜 | [sjy789] | 组长、模型训练 | 分类模型实现、参数调优、模型融合 | 33.3% |
| 邹佳怡 | [chowic] | 模型解释 | 大模型 API 接入、解释提示词 | 33.3% |
| 陶泽镐 | [lalalalala13579] | 模型训练、撰写报告 | 协助完成模型参数调优、项目功能集成与复现、报告整理 | 33.3% |


## 项目结构

```text
configs/
├── model_config.json       # 分类模型参数、融合权重和阈值
└── llm_config.json         # 大模型调用参数
docs/
└── experiment_summary.md   # 代表性实验结果汇总
src/
├── preprocessing.py       # 数据检查和文本预处理
├── models.py              # 模型定义、训练和融合
├── train.py               # 训练入口
├── evaluate.py            # 验证集评估入口
├── explanation.py         # 模型证据提取和本地降级解释
├── llm_explainer.py       # 大模型 API 客户端
└── predict.py             # 分类与解释入口
```

## 模型方案

最终分类器对三个组件的分数进行训练集标准化后加权融合：

```text
0.60 × Word/Char TF-IDF 集成模型
0.20 × BERTweet 均值池化特征 + LinearSVC
0.20 × MPNet 句向量 + LogisticRegression
```

分类阈值固定为 `0.03`，随机种子、模型名称及其他参数记录在 `configs/model_config.json`。

在课程提供的数据划分上，最终模型的验证集结果为：

```text
Accuracy:  0.8953（359/401）
Precision: 0.9030
Recall:    0.8514
F1:        0.8765
```

实验阶段共尝试 274 种模型、参数、阈值和融合组合，其中具有代表性的结果见 `docs/experiment_summary.md`。

## 运行环境

已验证环境：

```text
Python 3.13.7
Windows 11
PyTorch 2.8.0（CPU 构建）
```

安装依赖：

```powershell
pip install -r requirements.txt
```

首次运行时会从 Hugging Face 下载 BERTweet 和 MPNet。

## 准备数据

将课程提供的数据放置为：

```text
data/split/train.csv
data/split/val.csv
```

两个文件均应包含以下字段：

```text
id,text,label,event
```

## 训练模型

在项目根目录运行：

```powershell
python src/train.py
```

训练只读取 `train.csv`，最终模型保存为：

```text
models/best_model.joblib
```

## 模型评估

```powershell
python src/evaluate.py
```

程序在 `val.csv` 上比较代表性模型，并生成：

```text
results/metrics.json
results/classification_report.txt
results/confusion_matrix.png
```

## 配置学校大模型 API

判断依据默认由上海交大大模型接口中的 `deepseek-chat` 生成。使用前需要：

1. 申请学校大模型 API Key。
2. 连接上海交大校园网；校外使用学校 VPN。
3. 在运行程序的同一个终端中设置环境变量。

PowerShell 设置方法：

```powershell
$env:SJTU_API_KEY="你的API密钥"
$env:SJTU_BASE_URL="https://models.sjtu.edu.cn/api/v1"
$env:SJTU_LLM_MODEL="deepseek-chat"
```

环境变量只对当前终端有效。关闭终端后，下次运行前重新设置即可。

`.env.example` 是不含真实密钥的配置模板，用于说明需要设置哪些环境变量。当前程序不会自动读取 `.env` 文件，因此仍需使用上面的命令设置环境变量。真实 API Key 不得写入代码、README 或上传到 GitHub。

测试学校接口是否连通：

```powershell
python src/llm_explainer.py
```

正常情况下会返回类似：

```text
连接成功
```

## 分类与大模型解释

建议使用 `--show-source`，这样可以确认解释实际来自哪个模型：

```powershell
python src/predict.py --show-source "Breaking news example text"
```

成功调用学校接口时，输出形式为：

```text
检测结果：1（谣言）
判断依据：……
解释来源：deepseek-chat
```

不使用 `--show-source` 时，分类和解释仍会正常输出，只是不显示解释来源：

```powershell
python src/predict.py "Breaking news example text"
```

## API 失败与降级解释

如果 API Key 未设置、校园网络不可用、请求超时或学校接口返回错误，程序不会影响分类结果，也不会中断运行，而是自动使用本地证据生成解释。

使用以下命令可以观察自动降级后的解释来源：

```powershell
python src/predict.py --show-source "Breaking news example text"
```

降级时会显示：

```text
解释来源：local_fallback
```

也可以主动跳过大模型，强制使用本地解释：

```powershell
python src/predict.py --no-llm --show-source "Breaking news example text"
```

相同文本的大模型解释会缓存在 `.cache/` 中，减少重复请求。需要忽略缓存并重新调用 DeepSeek 时运行：

```powershell
python src/predict.py --refresh-explanation --show-source "Breaking news example text"
```

客户端按照配置限制为每分钟最多 10 次请求。

## Python 调用

```python
from src.predict import RumourDetectClass

detector = RumourDetectClass("models/best_model.joblib")

label = detector.classify("Breaking news example text")
explanation = detector.explain("Breaking news example text")
result = detector.predict_with_explanation("Breaking news example text")

print(result["label"])
print(result["explanation"])
print(result["explanation_source"])
```

`explanation_source` 的常见取值：

- `deepseek-chat`：解释由学校 DeepSeek 接口生成。
- `local_fallback`：学校接口不可用或主动使用本地降级解释。
