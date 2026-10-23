"""MQTT 云平台通信: OneNET 优先, 兼容标准 MQTT Broker (如本地 mosquitto).

OneNET (中国移动物联网开放平台) 接入要点:
- client_id = 设备名称, username = 产品 ID, password = 鉴权 token;
- token 算法 (官方文档): sign = base64(hmac_sha1(access_key, et + '\\n' + method + '\\n' + res)),
  其中 res = products/{pid}/devices/{device}, method = sha1, et 为过期时间戳;
- 数据点上报: $sys/{pid}/{device}/dp/post/json, 载荷 {"id":..., "dp": {"k":[{"v":...}]}};
- 命令下发: 订阅 $sys/{pid}/{device}/cmd/request/+, 回复 $sys/{pid}/{device}/cmd/response/{cmdid}。

断线重连由 paho loop_start 自愈, 离线期间的识别结果进入带上限的 outbox,
重连后自动补发, 保证业务数据不丢失 (超限时丢弃最旧, 防止内存膨胀)。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.parse
import uuid
from collections import deque

logger = logging.getLogger("car-recog.mqtt")


def generate_onenet_token(product_id: str, device_name: str, access_key: str,
                          ttl: int = 86400) -> str:
    """生成 OneNET MQTT 鉴权 token (sha1 签名, 默认有效期 24 小时)."""
    res = f"products/{product_id}/devices/{device_name}"
    et = str(int(time.time()) + ttl)
    sign_src = f"{et}\nsha1\n{res}"
    sign = base64.b64encode(
        hmac.new(access_key.encode("utf-8"), sign_src.encode("utf-8"),
                 hashlib.sha1).digest()).decode()
    token = (f"version=2018-10-31"
             f"&res={urllib.parse.quote(res, safe='')}"
             f"&et={et}"
             f"&method=sha1"
             f"&sign={urllib.parse.quote(sign, safe='')}")
    return token


def build_onenet_dp_payload(pairs: dict) -> dict:
    """OneNET 数据点上报载荷: {"id":..., "dp": {"key":[{"v": value}]}}."""
    return {"id": uuid.uuid4().hex, "dp": {str(k): [{"v": v}] for k, v in pairs.items()}}


class MqttBridge:
    """后台 MQTT 桥: 上报结果/难例, 订阅云端命令与 OTA 增量包."""

    def __init__(self, cfg, on_command=None, on_ota=None):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise SystemExit("未安装 paho-mqtt: pip install paho-mqtt")
        self._mqtt_mod = mqtt
        self.on_command = on_command  # (cmd_dict, reply_fn) -> None
        self.on_ota = on_ota          # (payload_dict) -> None
        o = cfg.mqtt.onenet
        self.onenet_mode = bool(o.product_id and o.device_name)
        self._topics = self._build_topics(cfg)

        if self.onenet_mode:
            client_id = o.device_name
            username = o.product_id
            access_key = os.environ.get("CAR_RECOG_ONENET_KEY") or o.access_key
            if not access_key:
                raise ValueError("OneNET 模式需要 access_key: 配置 mqtt.onenet.access_key "
                                 "或环境变量 CAR_RECOG_ONENET_KEY")
            self._token_fn = lambda: generate_onenet_token(
                o.product_id, o.device_name, access_key, int(o.token_ttl))
            password = self._token_fn()
            logger.info("OneNET 模式: product=%s device=%s", o.product_id, o.device_name)
        else:
            client_id = cfg.mqtt.client_id or f"car-recog-{uuid.uuid4().hex[:8]}"
            username = cfg.mqtt.username or None
            password = cfg.mqtt.password or None
            logger.info("标准 Broker 模式: %s:%s client_id=%s",
                        cfg.mqtt.host, cfg.mqtt.port, client_id)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=client_id, protocol=mqtt.MQTTv311)
        if username:
            self.client.username_pw_set(username, password)
        if cfg.mqtt.tls:
            self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        self._connected = threading.Event()
        self._outbox: deque = deque(maxlen=int(cfg.mqtt.outbox_size or 512))
        self._lock = threading.Lock()
        self._host, self._port = cfg.mqtt.host, cfg.mqtt.port
        self._keepalive = int(cfg.mqtt.keepalive or 60)

    def _build_topics(self, cfg) -> dict:
        t = cfg.mqtt.topics
        if self.onenet_mode:
            pid = cfg.mqtt.onenet.product_id
            dev = cfg.mqtt.onenet.device_name
            return {
                "result": t.result or f"$sys/{pid}/{dev}/dp/post/json",
                "command": t.command or f"$sys/{pid}/{dev}/cmd/request/+",
                "response_prefix": f"$sys/{pid}/{dev}/cmd/response",
                "hardcase": t.hardcase or None,
                "ota": t.ota or None,
            }
        return {
            "result": t.result or "car-recog/result",
            "command": t.command or "car-recog/cmd",
            "response_prefix": t.command or "car-recog/cmd/response",
            "hardcase": t.hardcase or "car-recog/hardcase",
            "ota": t.ota or None,
        }

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self.client.connect_async(self._host, int(self._port), keepalive=self._keepalive)
        self.client.loop_start()
        logger.info("MQTT 客户端已启动, 连接 %s:%s ...", self._host, self._port)

    def stop(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        self._connected.clear()
        logger.info("MQTT 客户端已停止")

    # ---------- paho 回调 ----------

    def _on_connect(self, client, userdata, *args):
        reason = args[1] if len(args) >= 2 else 0
        if getattr(reason, "is_failure", False):
            logger.error("MQTT 连接失败: %s", reason)
            return
        self._connected.set()
        logger.info("MQTT 已连接")
        subs = [(self._topics["command"], 1)]
        if self._topics["ota"]:
            subs.append((self._topics["ota"], 1))
        for topic, qos in subs:
            client.subscribe(topic, qos)
            logger.info("已订阅主题: %s", topic)
        # 补发离线期间的积压消息
        with self._lock:
            while self._outbox:
                topic, data, qos = self._outbox.popleft()
                client.publish(topic, data, qos=qos)
                logger.info("补发离线消息 -> %s", topic)

    def _on_disconnect(self, client, userdata, *args):
        self._connected.clear()
        logger.warning("MQTT 连接断开, 自动重连中...")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("收到非 JSON 消息 (%s): %s", msg.topic, msg.payload[:100])
            return
        topic = msg.topic
        if topic.startswith(self._topics["command"].rstrip("+")):
            cmdid = topic.rstrip("/").split("/")[-1]

            def reply_fn(code=200, message="ok"):
                self.reply_command(cmdid, code, message)

            if self.on_command:
                threading.Thread(target=self.on_command, args=(data, reply_fn),
                                 daemon=True, name="mqtt-cmd").start()
            return
        if self._topics["ota"] and topic == self._topics["ota"]:
            if self.on_ota:
                threading.Thread(target=self.on_ota, args=(data,),
                                 daemon=True, name="mqtt-ota").start()
            return

    # ---------- 发布接口 ----------

    def publish_json(self, topic: str, payload: dict, qos: int = 1) -> bool:
        """发布 JSON 消息; 离线时进入 outbox, 重连后自动补发."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._lock:
            if self._connected.is_set():
                try:
                    self.client.publish(topic, data, qos=qos)
                    return True
                except Exception as e:
                    logger.warning("MQTT 发布异常 (%s), 转入 outbox", e)
            self._outbox.append((topic, data, qos))
            return False

    def publish_result(self, pairs: dict) -> bool:
        """上报识别结果: OneNET 模式走数据点格式, 普通模式直发 JSON."""
        topic = self._topics["result"]
        payload = build_onenet_dp_payload(pairs) if self.onenet_mode else pairs
        return self.publish_json(topic, payload)

    def publish_hardcase(self, meta: dict) -> bool:
        topic = self._topics["hardcase"] or self._topics["result"]
        if self.onenet_mode:
            payload = build_onenet_dp_payload(
                {"hardcase": json.dumps(meta, ensure_ascii=False)})
        else:
            payload = meta
        return self.publish_json(topic, payload)

    def reply_command(self, cmdid: str, code: int = 200, msg: str = "ok") -> bool:
        return self.publish_json(f"{self._topics['response_prefix']}/{cmdid}",
                                 {"code": code, "msg": msg})
