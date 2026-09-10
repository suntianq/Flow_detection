# 项目架构与维护约定

## 模块边界

### `etm_flow.trace`

负责读取原始 ETM/TRBE 解码文本、ELF 和 maps 信息，输出带符号及上下文信息的 JSONL。该层可以依赖 `capstone` 和 `pyelftools`，但不应依赖 TensorFlow。

### `etm_flow.data`

负责把符号化事件转换为连续控制流片段，并编码成可用于训练的 memmap 数据集。该层可以依赖 NumPy，但不应导入具体模型实现。

### `etm_flow.modeling`

负责模型定义、训练、剪枝、预测和 TFLite 导出。模型只能从 `etm_flow.data` 读取稳定的数据接口，不能反向修改 Trace 解析结果。

### `tools/ghidra`

负责静态 CFG 提取和版本间二进制差异分析。这里的脚本在 Ghidra/PyGhidra 环境中直接执行，因此启动脚本、postScript 和 `ghidra_common.py` 必须保持相邻。它们不属于可安装的 `etm_flow` Python 包。

## 产物管理

以下内容由流水线生成，不进入 Git：

- `flow_preprocessed/`：结构化流序列；
- `flow_dataset/`、`flow_model/`：编码数据和训练产物；
- `*.keras`、`*.tflite`、`*.dat`：模型与大体积数组；
- `prediction_report.jsonl`：推理报告；
- `output/`：图片、PPT 和其他临时交付物。

如果未来需要发布固定模型，建议通过 GitHub Release 或对象存储发布，并在 README 中记录版本、校验值和对应的代码提交，不要直接把大模型文件塞进 Git 历史。

## 变更原则

1. Trace 格式变化先在 `trace` 层兼容，再更新 `data` 层。
2. 数据字段变化必须同步更新 `dataset.py`、模型输入和配置说明。
3. 新模型放在 `modeling/models/`，并在该目录的 `__init__.py` 注册。
4. 命令行行为变化要更新 README，并为纯函数补充快速测试。
5. 配置、代码和生成产物分开管理；配置可以提交，运行数据不提交。
