"""边缘端硬件执行机构抽象: 串口 (STM32/ESP8266) 或仿真模式.

与底层控制板的协议约定 (JSON 行, 固件侧需按同样格式实现):
  下发: {"cmd":"gate","value":"open","duration_ms":3000}\\n
  应答: {"code":0,"msg":"ok"}\\n

仿真模式 (hardware.mode=simulate) 不依赖任何硬件即可完整走通业务闭环,
是联调前的默认选择。
"""
from __future__ import annotations

import json
import logging
import threading

logger = logging.getLogger("car-recog.hardware")


class HardwareController:
    """硬件控制器基类."""

    def execute(self, action: str, **params) -> bool:
        raise NotImplementedError

    def gate_open(self, duration_ms: int = 3000) -> bool:
        """道闸抬杆."""
        return self.execute("gate", value="open", duration_ms=duration_ms)

    def gate_close(self) -> bool:
        """道闸落杆."""
        return self.execute("gate", value="close")

    def close(self) -> None:
        pass


class SimulatedController(HardwareController):
    """仿真控制器: 仅记录日志, 用于无硬件联调."""

    def execute(self, action: str, **params) -> bool:
        logger.info("[仿真硬件] 执行指令 %s %s", action, params or "")
        return True


class SerialController(HardwareController):
    """串口控制器: 通过 UART 与 STM32/ESP8266 控制板通信 (pyserial)."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        try:
            import serial
        except ImportError:
            raise SystemExit("串口模式需要 pyserial: pip install pyserial")
        self._serial_mod = serial
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self._lock = threading.Lock()
        logger.info("串口已打开: %s @ %d", port, baudrate)

    def execute(self, action: str, **params) -> bool:
        payload = {"cmd": action, **params}
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write(line.encode("utf-8"))
                resp = self._ser.readline().decode("utf-8", "ignore").strip()
                if not resp:
                    logger.warning("串口指令无应答: %s", payload)
                    return False
                ack = json.loads(resp)
                ok = int(ack.get("code", 1)) == 0
                logger.info("串口指令 %s -> %s", payload, "OK" if ok else resp)
                return ok
            except (self._serial_mod.SerialException, json.JSONDecodeError) as e:
                logger.error("串口通信异常: %s", e)
                return False

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass


def build_hardware_controller(cfg) -> HardwareController:
    mode = (cfg.hardware.mode or "simulate").lower()
    if mode == "serial":
        s = cfg.hardware.serial
        return SerialController(port=s.port, baudrate=int(s.baudrate),
                                timeout=float(s.timeout))
    return SimulatedController()
