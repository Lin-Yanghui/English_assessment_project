# 音素发音正确性 Baseline 项目说明

本项目用于构建一个可复现的音素级发音正确性判断 baseline。当前主线基于项目中已有的 CSV 数据文件完成第一轮建模、评估、预测样例和错误分析。

## 1. 当前范围

- 主模型输入：`phones_aligned.csv`
- 源数据 / 对照表：`phones.csv`
- 第一轮训练过滤条件：`alignment_quality == "pass"`
- 任务类型：音素级二分类
- 标签定义：`gold_binary=1` 表示音素可接受 / 正确，`gold_binary=0` 表示音素错误 / 不可接受
- 当前模型类型：传统表格特征 baseline 和 proxy GOP 风格分数
- 尚未包含：真实声学 GOP、wav2vec2 / HuBERT / WavLM / XLS-R 自监督语音表征、融合模型

## 2. 目录结构

```text
phones.csv               完整音素级源表 / 对照表
phones_aligned.csv       当前主模型输入表
models/                  训练好的 sklearn 模型文件
reports/                 指标、预测、对比表、错误分析和验收材料
scripts/                 可复现实验脚本
requirements.txt         Python 依赖
README.md                项目说明
```

## 3. 环境依赖

当前实验使用 Python 3.12。

安装依赖：

```powershell
pip install -r requirements.txt
```

如果本地环境中已经有 `pandas`、`scikit-learn`、`joblib`，则可以不重新安装。

## 4. 一键复现实验

在项目根目录运行：

```powershell
python scripts\run_phase1_pipeline.py --input phones_aligned.csv --alignment-quality pass
```

该命令会按顺序完成：

```text
多数类基线
传统特征模型训练
proxy GOP 阈值校准
正式评估
100 条预测样例导出
错误分析
模型对比表生成
阶段验收材料生成
```

## 5. 分步骤复现命令

如需单独运行每一步，可使用以下命令。

### 5.1 多数类基线

```powershell
python scripts\run_majority_baseline.py --input phones_aligned.csv --alignment-quality pass --metrics-output reports\majority_baseline_metrics.csv --predictions-output reports\majority_baseline_predictions.csv
```

### 5.2 特征模型训练

```powershell
python scripts\run_feature_baseline.py --input phones_aligned.csv --alignment-quality pass --metrics-output reports\feature_baseline_metrics.csv --predictions-output reports\feature_baseline_predictions.csv --model-dir models
```

### 5.3 proxy GOP 阈值校准

```powershell
python scripts\calibrate_proxy_gop_thresholds.py --input reports\feature_baseline_predictions.csv --thresholds-output reports\proxy_gop_group_thresholds.csv --metrics-output reports\proxy_gop_metrics.csv --predictions-output reports\proxy_gop_predictions.csv --objective macro_f1
```

### 5.4 正式评估

```powershell
python scripts\evaluate_model_outputs.py --predictions reports\proxy_gop_predictions.csv --metrics-output reports\formal_eval_metrics.csv --confusion-output reports\formal_eval_confusion_matrix.csv --summary-output reports\formal_eval_summary.md
```

### 5.5 导出 100 条测试音频预测样例

```powershell
python scripts\export_prediction_samples.py --input reports\proxy_gop_predictions.csv --output reports\prediction_samples_100_utterances.csv --utterance-list-output reports\prediction_sample_utterances.csv --model feature_random_forest --split test --num-utterances 100
```

### 5.6 错误分析

```powershell
python scripts\analyze_prediction_errors.py --input reports\proxy_gop_predictions.csv --model feature_random_forest --split test --cases-output reports\error_cases.csv --group-output reports\error_analysis_by_phone_group.csv --phone-output reports\error_analysis_by_target_phone.csv --summary-output reports\error_analysis_summary.md --max-cases-per-type 50
```

### 5.7 模型对比

```powershell
python scripts\build_model_comparison.py --majority-metrics reports\majority_baseline_metrics.csv --feature-metrics reports\feature_baseline_metrics.csv --proxy-gop-metrics reports\proxy_gop_metrics.csv --output reports\model_comparison.csv --summary-output reports\model_comparison.md
```

## 6. 输入 CSV 文件说明

### 6.1 主输入表

```text
phones_aligned.csv
```

该表是当前第一轮模型训练和评估的主输入。

规模：

```text
总音素行数：94,445
alignment_quality=pass：90,323
alignment_quality=review：4,122
```

第一轮训练只使用：

```text
alignment_quality == "pass"
```

pass 样本分布：

```text
总行数：90,323
train：36,064
dev：8,788
test：45,471
gold_binary=1：86,180
gold_binary=0：4,143
```

### 6.2 合并源表

```text
phones.csv
```

该表是更完整的合并源表，可作为后续模型训练输入或数据追溯依据。

规模：

```text
总行数：114,429
SpeechOcean762：94,445
L2-ARCTIC-v5.0-Mandarin：19,984
```

### 6.3 关键字段

```text
utterance_id
speaker_id
target_phone
phone_index
start_ms
end_ms
duration_ms
source_score
gold_binary
attention_binary
gold_three_class
perceived_phone
error_type
phone_group
split
audio_path
annotation_path
alignment_method
alignment_score
alignment_quality
dataset_source
```

下游预测报告中还会生成兼容字段，例如：

```text
observed_phone
error_type_hint
annotation_text
```

## 7. 对齐状态

A 部分已经完成 SpeechOcean762 的自动音素对齐。

```text
SpeechOcean 对齐音素总数：94,445
pass：90,323
review：4,122
```

合并表 `phones.csv` 共 114,429 行，可在后续 review 样本复核和重新对齐后作为更完整的训练输入。

对齐质量曾用 L2-ARCTIC 人工边界进行验证：

```text
边界误差中位数：约 20ms
85.3% 的边界误差在 50ms 内
93.4% 的边界误差在 100ms 内
```

因此当前第一轮模型默认只使用 `alignment_quality == "pass"`，暂不把 `review` 样本作为主训练数据。

## 8. 当前模型结果

查看完整结果：

```text
reports/model_comparison.md
reports/model_comparison.csv
```

当前 test 集按 Balanced Accuracy 排名：

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

说明：

```text
Accuracy 不能单独作为主要判断标准。
由于数据类别不平衡，应优先查看 Balanced Accuracy、Macro-F1、AUC 和错误召回。
```

## 9. 关键交付物

```text
phones_aligned.csv
models/feature_logreg.joblib
models/feature_random_forest.joblib
reports/model_comparison.md
reports/formal_eval_summary.md
reports/formal_eval_metrics.csv
reports/formal_eval_confusion_matrix.csv
reports/prediction_samples_100_utterances.csv
reports/error_analysis_summary.md
reports/error_cases.csv
reports/data_manifest.csv
reports/label_spec.md
reports/split_manifest.csv
reports/speaker_split_check.csv
reports/phase1_acceptance_checklist.csv
reports/phase1_model_design_status.md
```

## 10. 当前错误分析

查看：

```text
reports/error_analysis_summary.md
reports/error_analysis_by_phone_group.csv
reports/error_analysis_by_target_phone.csv
reports/error_cases.csv
```

当前 Random Forest + phone_group 阈值模型的薄弱类别：

```text
vowel
fricative
stop
nasal
liquid
```

需要重点优化的目标音素包括：

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

## 11. 第一阶段验收材料

查看：

```text
reports/data_manifest.csv
reports/label_spec.md
reports/split_manifest.csv
reports/speaker_split_check.csv
reports/phase1_acceptance_checklist.csv
reports/phase1_model_design_status.md
phase1_completion_summary_cn.md
```

这些文件用于将当前项目实现与第一阶段计划书要求进行对应。

当前判断：

```text
第一阶段 baseline 交付版：已完成
第一阶段完整模型设计验收版：未完成
```

## 12. 后续建议

建议按以下顺序继续：

1. 增加真实音频声学特征或真实 GOP 分数。
2. 实现 `target_phone` 级阈值校准，并在样本不足时回退到 `phone_group`。
3. 针对 `vowel`、`fricative`、`stop`、`nasal`、`liquid` 做专项优化。
4. 接入 wav2vec2 / HuBERT / WavLM / XLS-R 等自监督语音表征。
5. 构建融合模型：GOP + 时长 + 音素 ID + 音素组 + 自监督 embedding。
6. 复核 `review` 样本，决定是否进入第二轮训练。
7. 接入项目内部小样本验证集，检查模型迁移效果。
