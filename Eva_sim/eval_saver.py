
import os
import csv
import datetime
import json
from typing import Optional, Dict, Any


class EvalSaver:
    """
    评估专用的 CSV 保存器，只写真正有用的列：
    timestamp, episode, seed, return,
    waiting, queue, speed,
    ATT, AWT, AS, TP,
    route_info

    并且会根据“这次用的权重文件”的上一级目录分文件夹。
    例如：
        ckpt/regular/First_2.5_test/2025-10-25_00-48-02/42_xxx.pt
    会写到
        eval_logs/2025-10-25_00-48-02/eval_results.csv
    """

    def __init__(
        self,
        base_dir: str = "eval_logs",
        csv_dir: Optional[str] = None,
    ) -> None:
        # 兼容 csv_dir 老写法
        self.base_dir = csv_dir if csv_dir is not None else base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ---------- 路径相关 ----------

    def _dir_for_weight(self, weight_path: Optional[str]) -> str:
        """根据权重路径决定要把 csv 写到哪个子目录"""
        if not weight_path:
            return self.base_dir

        norm = os.path.normpath(weight_path)
        parent = os.path.dirname(norm)
        last = os.path.basename(parent)
        target = os.path.join(self.base_dir, last)
        os.makedirs(target, exist_ok=True)
        return target

    def _ensure_header(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "timestamp",
                        "episode",
                        "seed",
                        "return",
                        "waiting",
                        "queue",
                        "speed",
                        "ATT",
                        "AWT",
                        "AS",
                        "TP",
                        "route_info",
                    ]
                )

    # ---------- 主方法 ----------

    def save_episode(
        self,
        episode: int,
        seed: int,
        episode_return: float,
        waiting: float,
        queue: float,
        speed: float,
        metrics: Dict[str, Any],
        weight_path: Optional[str] = None,
        route_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        metrics 要求至少有：ATT / AWT / AS / TP
        其他的会自动忽略
        """
        target_dir = self._dir_for_weight(weight_path)
        csv_path = os.path.join(target_dir, "eval_results.csv")
        self._ensure_header(csv_path)

        att = float(metrics.get("ATT", 0.0))
        awt = float(metrics.get("AWT", 0.0))
        avg_speed = float(metrics.get("AS", 0.0))
        tp = int(metrics.get("TP", 0))

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_json = json.dumps(route_info, ensure_ascii=False) if route_info else ""

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    ts,
                    episode,
                    seed,
                    float(episode_return),
                    float(waiting),
                    float(queue),
                    float(speed),
                    att,
                    awt,
                    avg_speed,
                    tp,
                    route_json,
                ]
            )
