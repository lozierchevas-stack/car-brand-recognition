# 汽车品牌识别系统

基于 **Visual RAG 度量学习** 的汽车品牌识别，含边缘端 MQTT/OTA 闭环。

## 核心交付

| 文件 | 说明 |
| --- | --- |
| `data_pipeline.py` | 合规数据集下载 / .mat 标注解析 / GT 边界框精准裁剪 |
| `train_export.py` | ArcFace + MobileNetV3 训练（P-K Sampling / AMP）与 ONNX 自动导出 |
| `edge_iot_inference.py` | 边缘端主控：多线程抽帧 / Faiss 双重阈值检索 / MQTT 联动 / 难例采集 |
| `build_gallery.py` | 构建 Faiss 索引类中心 gallery |
| `test_retrieval.py` | 检索效果评测 |
| `export_trt.py` | TensorRT 加速部署 |
| `simulate_ota_push.py` | OTA 增量推送模拟 |
| `default.yaml` | 全局配置（数据 / 模型 / 训练 / 检索 / 边缘端 / MQTT / OTA） |

## 模型与推理

- **Backbone**：MobileNetV3（移除分类层）→ 512 维 Embedding
- **损失**：ArcFace（加性角度间隔），可选 CosFace / Softmax 对照
- **采样**：P-K Sampler（每 Batch 抽 P 个品牌、每品牌 K 张图）
- **检索**：Faiss 索引，双重阈值 + 类中心召回
- **部署**：ONNX 导出（含 L2 归一化）→ TensorRT / 边缘端推理

## 配置

敏感信息通过环境变量注入：

- `CAR_RECOG_ONENET_KEY`
- `CAR_RECOG_MQTT_PASSWORD`
