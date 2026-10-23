"""配置加载: YAML -> 支持点号访问的 Config 对象, 并支持环境变量注入敏感信息.

敏感信息注入约定 (避免明文写入仓库):
- CAR_RECOG_ONENET_KEY       -> mqtt.onenet.access_key
- CAR_RECOG_MQTT_PASSWORD    -> mqtt.password
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config(dict):
    """dict 子类, 支持 cfg.train.lr 点号访问与 cfg.get("train.lr", default) 层级取值.

    嵌套 dict 在首次访问时惰性包装为 Config 并回写, 保证 cfg.edge.source = x 这类
    写操作作用于原始配置树 (而非临时副本)。
    """

    def __getattr__(self, key: str):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"配置项不存在: {key}") from None
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value):
        self[key] = value

    def get(self, path, default=None):
        """支持点号路径的层级取值, 例如 cfg.get("retrieval.min_sim", 0.55)."""
        node = self
        for part in str(path).split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _set_dotted(cfg: dict, path: str, value) -> None:
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _inject_env_secrets(cfg: Config) -> None:
    access_key = os.environ.get("CAR_RECOG_ONENET_KEY")
    if access_key:
        _set_dotted(cfg, "mqtt.onenet.access_key", access_key)
    password = os.environ.get("CAR_RECOG_MQTT_PASSWORD")
    if password:
        _set_dotted(cfg, "mqtt.password", password)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """加载 YAML 配置; overrides 为 {"train.epochs": 60} 形式的点号路径覆盖."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f) or {})
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(cfg, key, value)
    _inject_env_secrets(cfg)
    return cfg
