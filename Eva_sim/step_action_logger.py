
# 负责把每一步的控制动作写成:
# time,intersection,action
# 并且按权重文件的上一级目录分文件夹

import os
import csv
from typing import Optional


class StepActionLogger:
    def __init__(self, base_dir: str = "eval_logs", weight_path: Optional[str] = None):
        """
        base_dir   : 所有日志的根目录，比如 eval_logs
        weight_path: 这次评估加载的权重全路径，用来决定写到哪个子目录
                     例: ckpt/regular/foo/2025-10-25_xx/42_xx.pt
                     -> eval_logs/2025-10-25_xx/step_actions.csv
        """
        self.base_dir = base_dir
        self.weight_path = weight_path
        self.csv_path = self._make_csv_path()
        self._ensure_header()

    def _make_csv_path(self) -> str:
        # 没传权重，就直接写到根目录
        if not self.weight_path:
            os.makedirs(self.base_dir, exist_ok=True)
            return os.path.join(self.base_dir, "step_actions.csv")

        # 规范路径 → 取权重所在目录的最后一段
        norm = os.path.normpath(self.weight_path)
        parent = os.path.dirname(norm)
        last = os.path.basename(parent)
        target_dir = os.path.join(self.base_dir, last)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, "step_actions.csv")

    def _ensure_header(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["time", "intersection", "action"])

    def log_step(self, t: float, intersection: str, action: int) -> None:
        """
        t           : 仿真时间（秒）——来自 traci.simulation.getTime()
        intersection: 这一步做动作的路口 ID（比如 B1 / C3 / E4）
        action      : 你 agent 选出来的 0~7
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([t, str(intersection), int(action)])
