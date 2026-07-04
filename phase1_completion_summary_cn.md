# 第一阶段完成情况与未完成内容整理

## 1. 总体结论

当前项目已经完成第一阶段的“最小可运行 baseline 闭环”：

```text
数据输入 -> 标签读取 -> 模型训练 -> 阈值校准 -> 测试评估 -> 预测样例 -> 错误分析 -> 复现说明
```

但项目尚未完全达到第一阶段计划书中的完整模型设计目标。主要缺口是：

```text
真实 GOP 声学模型
自监督语音表征模型
融合模型
target_phone 级阈值校准
目标指标达标
review 样本处理
内部数据验证
```

因此当前状态可定义为：

```text
第一阶段 baseline 交付版：已完成
第一阶段完整模型设计验收版：未完成
```

## 2. 已完成内容

### 2.1 主数据输入

项目当前主输入文件为：

```text
phones_aligned.csv
phones.csv
```

其中 `phones_aligned.csv` 是第一轮模型训练和评估的主输入：

```text
总音素数：94,445
alignment_quality=pass：90,323
alignment_quality=review：4,122
```

第一轮训练已按计划书建议，只使用：

```text
alignment_quality == "pass"
```

`phones.csv` 是合并源表：

```text
总行数：114,429
SpeechOcean762：94,445
L2-ARCTIC-v5.0-Mandarin：19,984
```

### 2.2 标签字段

项目中已经具备音素级标签字段：

```text
gold_binary
attention_binary
gold_three_class
source_score
error_type
phone_group
```

当前第一轮模型使用：

```text
gold_binary
```

作为二分类训练目标。

标签含义：

```text
gold_binary = 1：音素可接受 / 正确
gold_binary = 0：音素错误 / 不可接受
```

对应说明文件：

```text
reports/label_spec.md
```

### 2.3 训练 / 验证 / 测试划分

项目已有：

```text
train
dev
test
```

当前 pass 样本分布：

```text
train：36,064
dev：8,788
test：45,471
```

对应数据清单：

```text
reports/data_manifest.csv
```

### 2.4 多数类基线

已完成多数类 baseline，用于证明仅看 Accuracy 会误导模型判断。

产物：

```text
reports/majority_baseline_metrics.csv
reports/majority_baseline_predictions.csv
```

当前多数类基线 test 结果：

```text
Accuracy = 0.959952
Balanced Accuracy = 0.500000
Macro-F1 = 0.489784
AUC = 0.500000
```

### 2.5 传统特征模型

已完成两个传统特征模型：

```text
Logistic Regression
Random Forest
```

模型文件：

```text
models/feature_logreg.joblib
models/feature_random_forest.joblib
```

当前使用的输入特征：

```text
target_phone
phone_group
duration_ms
phone_index
```

产物：

```text
reports/feature_baseline_metrics.csv
reports/feature_baseline_predictions.csv
```

### 2.6 proxy GOP / 伪 GOP 流程

项目已完成 proxy GOP 风格流程：

```text
proxy_gop_score
group_threshold
prediction
confidence
```

相关产物：

```text
reports/proxy_gop_metrics.csv
reports/proxy_gop_predictions.csv
reports/proxy_gop_group_thresholds.csv
```

需要注意：

```text
当前 proxy_gop_score 是模型概率分数，不是真正基于声学似然或候选音素后验概率的 GOP。
```

### 2.7 阈值校准

已完成：

```text
全局阈值
phone_group 阈值
```

阈值选择在 dev 集完成，test 集只用于最终评估。

尚未完成：

```text
target_phone 级阈值
样本不足时回退到 phone_group 阈值
```

### 2.8 正式评估

已输出计划书要求的核心评估指标：

```text
Accuracy
Balanced Accuracy
Precision
Recall
Macro-F1
AUC
Confusion Matrix
```

相关产物：

```text
reports/formal_eval_metrics.csv
reports/formal_eval_confusion_matrix.csv
reports/formal_eval_summary.md
```

当前最佳 test 结果：

```text
feature_random_forest + global_threshold
Balanced Accuracy = 0.604845
Macro-F1 = 0.384813
AUC = 0.645927
```

### 2.9 100 条预测样例

已生成 100 条测试音频的音素级预测样例。

产物：

```text
reports/prediction_samples_100_utterances.csv
reports/prediction_sample_utterances.csv
```

预测样例包含字段：

```text
utterance_id
speaker_id
target_phone
start_ms
end_ms
duration_ms
gold_binary
prediction
confidence
proxy_gop_score
group_threshold
phone_group
audio_path
model
calibration
```

### 2.10 错误分析

已完成错误分析。

产物：

```text
reports/error_analysis_summary.md
reports/error_cases.csv
reports/error_analysis_by_phone_group.csv
reports/error_analysis_by_target_phone.csv
```

当前主要薄弱类别：

```text
vowel
fricative
stop
nasal
liquid
```

当前代表性薄弱音素：

```text
AH
N
L
S
IH
T
DH
R
IY
Z
```

### 2.11 模型对比

已生成模型对比表。

产物：

```text
reports/model_comparison.md
reports/model_comparison.csv
```

当前 test 排名按 Balanced Accuracy 排序：

```text
1. feature_random_forest + global_threshold
   Balanced Accuracy = 0.604845
   Macro-F1 = 0.384813
   AUC = 0.645927

2. feature_logreg + global_threshold
   Balanced Accuracy = 0.582945
   Macro-F1 = 0.440925
   AUC = 0.623729

3. feature_random_forest + phone_group_threshold
   Balanced Accuracy = 0.557820
   Macro-F1 = 0.527742
   AUC = 0.645927

4. feature_logreg + phone_group_threshold
   Balanced Accuracy = 0.538468
   Macro-F1 = 0.513364
   AUC = 0.623729

5. majority_class
   Balanced Accuracy = 0.500000
   Macro-F1 = 0.489784
   AUC = 0.500000
```

### 2.12 阶段验收材料

已生成：

```text
reports/data_manifest.csv
reports/label_spec.md
reports/phase1_acceptance_checklist.csv
reports/phase1_model_design_status.md
```

这些文件用于对应第一阶段计划书中的交付物和验收要求。

### 2.13 复现材料

已具备：

```text
README.md
requirements.txt
scripts/
```

一键复跑命令：

```powershell
python scripts\run_phase1_pipeline.py --input phones_aligned.csv --alignment-quality pass
```

## 3. 未完成内容

### 3.1 真实 GOP 声学模型未完成

计划书要求：

```text
GOP 或等价声学似然模型
```

当前项目只有：

```text
proxy_gop_score
```

该分数来自传统模型概率，不是真正基于声学模型、候选音素似然或音素后验概率计算的 GOP。

后续需要补：

```text
声学模型
目标音素似然
候选音素似然
真实 GOP 计算
按音素 / 音素组阈值校准
```

### 3.2 自监督语音表征模型未完成

计划书建议尝试：

```text
wav2vec2
HuBERT
WavLM
XLS-R
```

当前尚未实现：

```text
音频片段切分
embedding 抽取
冻结预训练模型
embedding 分类器
```

### 3.3 融合模型未完成

计划书建议融合：

```text
GOP 分数
目标音素 ID
音素组
音素时长
语速特征
自监督 embedding
```

当前模型只使用了基础表格特征，尚未完成：

```text
GOP + SSL embedding + 表格特征融合模型
```

### 3.4 target_phone 阈值校准未完成

当前已完成：

```text
全局阈值
phone_group 阈值
```

尚未完成：

```text
target_phone 级阈值
样本不足时回退到 phone_group 阈值
```

### 3.5 指标尚未达到计划书目标线

计划书目标线大致为：

```text
Balanced Accuracy >= 0.70
Macro-F1 >= 0.55
AUC >= 0.70
```

当前最佳：

```text
Balanced Accuracy = 0.604845
Macro-F1 = 0.384813
AUC = 0.645927
```

尚未达到目标线。

### 3.6 review 样本尚未进入正式训练

当前按计划建议只使用：

```text
alignment_quality == "pass"
```

未完成：

```text
review 样本复核
review 样本重新对齐
review 样本进入鲁棒性测试或二轮训练
```

### 3.7 内部项目数据验证未完成

计划书提到可用项目内部小样本做迁移验证。

当前尚未看到：

```text
内部验证集
内部数据标签规范
内部数据评估报告
内部数据和公开数据指标对比
```

## 4. 当前项目状态判断

当前项目已经达到：

```text
第一阶段 baseline 交付版
```

因为已经完成：

```text
数据表
标签规范
模型训练
阈值校准
测试评估
预测样例
错误分析
复现说明
验收检查表
```

但尚未达到：

```text
第一阶段完整模型设计验收版
```

原因是仍缺：

```text
真实 GOP
自监督语音模型
融合模型
target_phone 阈值
指标达标
review / 内部数据验证
```

## 5. 建议下一步

建议按以下顺序继续：

1. 增加真实声学特征或真实 GOP 分数。
2. 实现 `target_phone` 级阈值校准，并设置样本不足回退到 `phone_group`。
3. 对 `vowel`、`fricative`、`stop`、`nasal`、`liquid` 做专项错误分析和规则/模型优化。
4. 接入自监督语音表征，先冻结模型抽 embedding，再训练轻量分类器。
5. 构建融合模型，对比：

```text
多数类基线
传统特征模型
真实 GOP 模型
自监督表征模型
融合模型
```

6. 复核 `review` 样本，决定是否进入第二轮训练。
7. 接入项目内部小样本验证集，检查迁移效果。

