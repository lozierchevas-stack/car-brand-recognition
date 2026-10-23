"""汽车品牌识别系统 —— Visual RAG 度量学习 + 边缘端 MQTT/OTA 闭环.

核心交付 (项目根目录):
- data_pipeline.py       合规数据集下载 / .mat 标注解析 / GT 边界框精准裁剪
- train_export.py        ArcFace + MobileNetV3 训练 (P-K Sampling / AMP) 与 ONNX 自动导出
- edge_iot_inference.py  边缘端主控: 多线程抽帧 / Faiss 双重阈值检索 / MQTT 联动 / 难例采集
"""
__version__ = "1.0.0"
