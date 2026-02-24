# -*- coding: utf-8 -*-
"""

强化版奖励构造：
r(alpha) = (1 - alpha) * r_warmup + alpha * r_band

其中
- r_warmup:
    queue_ratio = queue_in / cap
    queue_norm  = clip(queue_ratio, 0, 1)          # 原有 0~1 归一队列
    over        = max(0, queue_ratio - 1)          # 超载比例（>0 表示爆负荷）
    pressure    = (queue_out - queue_in) / cap
    pressure_norm = clip(pressure, -1, 1)

    · 主项：   - queue_norm
    · 超载罚： - λ_over * over
    · 压力项： + w_pressure * pressure_norm
    · 切换罚： - penal_switch * switched
    · 公平罚： - λ_fair * fair   （方向/入口不均衡）
    · 等待罚： - λ_wait * excess_wait/T_wait_max  （最大等待时间超阈值）

- r_band:
    使用 queue_norm 做带宽损失 bandloss(queue_norm)
    r_band = -alpha_band*bandloss(queue_norm) - penal_switch*switched + 0.2*pressure_norm

说明：
- 只依赖 TrafficSignal 常规属性：lanes_in, lanes_out, phase_has_changed, is_yellow
- sumo 参数是 traci 接口对象：sumo.lane.getLastStepHaltingNumber / getWaitingTime
"""

from typing import Dict, Tuple, Optional
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x


class CustomRewardHelper:
    def __init__(
        self,
        alpha: float = 0.05,
        qcap_per_lane: float = 12.0,
        w_pressure: float = 0.4,
        penal_switch: float = 0.05,
        band_ratio: Tuple[float, float] = (0.10, 0.15),
        alpha_band: float = 1.0,
        ewma_rho: float = 0.0,
    ):
        """
        参数：
        - alpha       : r_warmup / r_band 混合权重，训练中可由 RewardScheduler 动态调节
        - qcap_per_lane: 每条入向车道的“容量”估计（车辆数）
        - w_pressure  : 压力项权重
        - penal_switch: 相位切换惩罚系数
        - band_ratio  : (qL, qU)，队列带宽目标区间
        - alpha_band  : r_band 内部损失缩放
        - ewma_rho    : epoch 平均队列的 EWMA 平滑系数（0 表示不用平滑）
        """
        self.alpha = float(alpha)
        self.qcap_per_lane = float(qcap_per_lane)
        self.w_pressure = float(w_pressure)
        self.penal_switch = float(penal_switch)
        self.qL, self.qU = band_ratio
        self.alpha_band = float(alpha_band)
        self.ewma_rho = float(ewma_rho)

        # === 用于 scheduler 的 epoch 统计 ===
        self._epoch_q_sum = 0.0   # 累积 queue_norm
        self._epoch_q_cnt = 0     # 调用次数
        self._ewma_qbar: Optional[float] = None

        # === 公平 / 等待 / tail 相关超参数 ===
        self.lambda_over = 0.7    # 超载惩罚 λ_over
        # 加强方向不均衡与长等待的影响，更关注 tail
        self.lambda_fair = 0.5    # 方向不均衡惩罚 λ_fair（原 0.3）
        self.q_thresh = 0.1       # 启动公平惩罚的 q_mean 阈值（原 0.2）
        self.lambda_wait = 0.4    # 等待时间惩罚 λ_wait（原 0.2）
        self.T_wait_max = 40.0    # 最大可容忍等待时间（秒）
        # 每个方向级 queue_norm 的“tail”惩罚：q_dir_norm 最大值超过该阈值时加重惩罚
        self.q_tail_thresh = 0.7  # 方向级 queue_norm 超过 0.7 视为非常拥堵
        self.lambda_tail = 0.5    # tail 惩罚强度

        # 等待时间惩罚的饱和上限（单位：T_wait_max 的倍数）
        # 例如 3.0 表示：最多再为“超出部分”多罚 3 * T_wait_max
        self.wait_over_clip = 3.0

        # 单步 reward 的截断范围，避免极端 outlier 拉爆 value 函数
        self.r_min = -5.0
        self.r_max =  1.0


    # ===== 接口：给 scheduler 用 =====
    def set_alpha(self, alpha: float):
        # 为了避免 r_band 完全压制 r_warmup，这里限制 alpha 的上限为 0.5
        self.alpha = float(_clamp(alpha, 0.0, 0.5))

    def get_alpha(self) -> float:
        return self.alpha


    # ===== epoch 统计 =====
    def begin_epoch(self):
        """在一个 epoch/若干 episode 开始前调用，用于累积 qbar。"""
        self._epoch_q_sum = 0.0
        self._epoch_q_cnt = 0

    def end_epoch(self) -> float:
        """
        在 epoch 结束时调用，返回本 epoch 的平均 queue_norm（可带 EWMA）。
        Train 侧会把这个 qbar 交给 RewardScheduler.update()。
        """
        if self._epoch_q_cnt <= 0:
            qbar = 0.0
        else:
            qbar = self._epoch_q_sum / float(self._epoch_q_cnt)

        if self.ewma_rho > 0.0:
            if self._ewma_qbar is None:
                self._ewma_qbar = qbar
            else:
                rho = self.ewma_rho
                self._ewma_qbar = rho * self._ewma_qbar + (1.0 - rho) * qbar
            return self._ewma_qbar
        else:
            return qbar

    # ===== 内部工具 =====
    def _bandloss(self, q: float) -> float:
        """
        q: queue_norm ∈ [0, 1]
        bandloss = [max(0, q - qU)]^2 + [max(0, qL - q)]^2
        """
        qL, qU = self.qL, self.qU
        hi = max(0.0, q - qU)
        lo = max(0.0, qL - q)
        return hi * hi + lo * lo

    # --------- 统一奖励入口（sumo-rl 会调用）---------
    def __call__(self, tls) -> float:
        """
        tls: TrafficSignal 对象（来自 sumo-rl）
        - tls.lanes_in / lanes_out: list[str]
        - tls.phase_has_changed: bool
        - tls.is_yellow: bool
        sumo: traci 接口
        - sumo.lane.getLastStepHaltingNumber(lane)
        - sumo.lane.getWaitingTime(lane)
        """
        # 更稳健地获取 traci 接口
        # 1) 优先从 TrafficSignal 自身拿
        sumo = getattr(tls, "sumo", None)

        # 2) 如果没有，再尝试从 env 上拿
        if sumo is None:
            env = getattr(tls, "env", None)
            if env is not None:
                sumo = getattr(env, "sumo", None)

        # 3) 仍然拿不到就直接返回 0（极早期/异常兜底）
        if sumo is None:
            # 可以临时打开一行 debug，确认不会频繁触发：
            # print("[Reward] tls 没有 sumo 句柄，返回 0.0", getattr(tls, "id", "?"))
            return 0.0

        lanes_in = getattr(tls, "lanes", [])
        lanes_out = getattr(tls, "out_lanes", [])

        if not lanes_in:
            return 0.0

        cap = self.qcap_per_lane * max(1, len(lanes_in))

        # ==== 1) 队列（入向 / 出向） ====
        q_in = 0.0
        q_out = 0.0

        # 入向
        for l in lanes_in:
            try:
                q_in += float(sumo.lane.getLastStepHaltingNumber(l))
            except Exception:
                # 退化方案：按速度近零判停
                veh_ids = sumo.lane.getLastStepVehicleIDs(l)
                q_in += float(sum(1 for vid in veh_ids if sumo.vehicle.getSpeed(vid) <= 0.1))

        # 出向
        for l in lanes_out:
            try:
                q_out += float(sumo.lane.getLastStepHaltingNumber(l))
            except Exception:
                veh_ids = sumo.lane.getLastStepVehicleIDs(l)
                q_out += float(sum(1 for vid in veh_ids if sumo.vehicle.getSpeed(vid) <= 0.1))

        queue_ratio = q_in / cap
        queue_norm = _clamp(queue_ratio, 0.0, 1.0)
        over = max(0.0, queue_ratio - 1.0)

        # 压力：出 - 入
        pressure = (q_out - q_in) / cap
        pressure_norm = _clamp(pressure, -1.0, 1.0)

        # 保存到 epoch 统计
        self._epoch_q_sum += queue_norm
        self._epoch_q_cnt += 1

        # 相位切换
        switched = 1.0 if getattr(tls, "phase_has_changed", False) and not getattr(tls, "is_yellow", False) else 0.0

        # ==== 2) 入口方向不均衡（这里按“入向 edge 分组”，无需 lane_dir，泛用）====
        # 按 edgeID 聚合入向队列
        edge_queues: Dict[str, float] = {}
        edge_lanes_cnt: Dict[str, int] = {}

        for l in lanes_in:
            # 简单从 laneID 拆 edgeID：SUMO 默认 laneID 类似 "edge_0_0"
            edge_id = l.rsplit("_", 1)[0] if "_" in l else l
            edge_queues[edge_id] = edge_queues.get(edge_id, 0.0)
            edge_lanes_cnt[edge_id] = edge_lanes_cnt.get(edge_id, 0) + 1

        # 估一个“方向归一队列”：q_dir_norm = q_dir_raw / (qcap_per_lane * lane_cnt)
        q_dir_norm_list = []
        for edge_id, lane_cnt in edge_lanes_cnt.items():
            # 粗略：同一 edge 内的队列按比例估计
            # 这里简单地“均分”：q_dir_raw ≈ lane_cnt / len(lanes_in) * q_in
            # （不是精确按 lane，但足够用于不均衡检测）
            q_dir_raw = q_in * (lane_cnt / float(len(lanes_in)))
            cap_dir = self.qcap_per_lane * lane_cnt
            if cap_dir <= 0:
                continue
            q_dir_norm_list.append(q_dir_raw / cap_dir)

        r_fair = 0.0
        if len(q_dir_norm_list) >= 2:
            q_mean = sum(q_dir_norm_list) / float(len(q_dir_norm_list))
            if q_mean > self.q_thresh:
                q_max = max(q_dir_norm_list)
                q_min = min(q_dir_norm_list)
                fair = (q_max - q_min) / (q_mean + 1e-6)
                r_fair = - self.lambda_fair * fair

        # tail 惩罚：关注最拥堵方向的 queue_norm
        r_tail = 0.0
        if len(q_dir_norm_list) >= 2:
            q_max = max(q_dir_norm_list)
            excess = max(0.0, q_max - self.q_tail_thresh)
            if excess > 0.0:
                r_tail = - self.lambda_tail * (excess ** 2)


        # ==== 3) 最大等待时间惩罚（单位：秒）====
        max_wait_in = 0.0
        for l in lanes_in:
            try:
                w = float(sumo.lane.getWaitingTime(l))
            except Exception:
                w = 0.0
            if w > max_wait_in:
                max_wait_in = w

        excess_wait = max(0.0, max_wait_in - self.T_wait_max)

        if self.T_wait_max > 0.0 and self.lambda_wait > 0.0:
            # 先按 T_wait_max 归一，得到“超过几倍阈值”
            excess_factor = excess_wait / self.T_wait_max
            # 再限制到 [0, wait_over_clip]，避免极端 episode 惩罚爆炸
            excess_factor = min(excess_factor, self.wait_over_clip)
            r_wait = - self.lambda_wait * excess_factor
        else:
            r_wait = 0.0


        # ==== 4) 组合 r_warmup ====
        r_warmup = (
            - queue_norm
            + self.w_pressure * pressure_norm
            - self.penal_switch * switched
        )
        r_warmup += - self.lambda_over * over
        r_warmup += r_fair
        r_warmup += r_wait
        r_warmup += r_tail


        # ==== 5) 带宽 reward r_band ====
        band = self._bandloss(queue_norm)
        r_band = (
            - self.alpha_band * band
            - self.penal_switch * switched
            + 0.2 * pressure_norm
        )

        # ==== 6) 混合（v3：根据拥堵程度 gating 带宽项） ====
        alpha = self.alpha

        # 情况 1：队列明显高于上界 → 只看 r_warmup，关闭带宽项
        if queue_norm >= self.qU + 0.1:
            alpha = 0.0

        # 情况 2：队列略高于上界 → 弱化带宽项影响
        elif queue_norm >= self.qU:
            alpha = alpha * 0.5

        # 情况 3：队列明显低于下界 → 允许一些带宽偏好，但限制不超过 0.5
        elif queue_norm <= self.qL - 0.1:
            alpha = min(alpha, 0.5)

        # 其余情况：保持 scheduler 给的 alpha 不变
        r = (1.0 - alpha) * r_warmup + alpha * r_band

        # 统一对 reward 做截断，抑制极端 outlier 对训练稳定性的破坏
        if self.r_min is not None or self.r_max is not None:
            lo = self.r_min if self.r_min is not None else -float("inf")
            hi = self.r_max if self.r_max is not None else  float("inf")
            if r < lo:
                r = lo
            elif r > hi:
                r = hi

        return float(r)




class RewardScheduler:
    """
    RewardScheduler: 依据 epoch 级别的平均队列 qbar，对 helper.alpha 做小步调整。

    逻辑：
    - 若 qbar 持续落在 [qL+eps, qU-eps] 区间内：
        连续 good >= patience，则 alpha += step
    - 若 qbar 持续落在 [qL-2eps, qU+2eps] 之外：
        连续 bad >= backoff，则 alpha -= step
    """
    def __init__(
        self,
        helper: CustomRewardHelper,
        qL: float = 0.10,
        qU: float = 0.15,
        eps: float = 0.01,
        step: float = 0.10,
        patience: int = 50,
        backoff: int = 100,
    ):
        self.h = helper
        self.qL = qL
        self.qU = qU
        self.eps = eps
        self.step = step
        self.patience = patience
        self.backoff = backoff
        self.good = 0
        self.bad = 0

    def update(self, qbar_epoch: float):
        L, U = self.qL + self.eps, self.qU - self.eps
        if L <= qbar_epoch <= U:
            self.good += 1
            self.bad = 0
            if self.good >= self.patience and self.h.alpha < 1.0:
                self.h.set_alpha(self.h.alpha + self.step)
                self.good = 0
        else:
            self.bad += 1
            self.good = 0
            if (qbar_epoch < self.qL - 2 * self.eps or qbar_epoch > self.qU + 2 * self.eps) \
               and self.bad >= self.backoff and self.h.alpha > 0.0:
                self.h.set_alpha(self.h.alpha - self.step)
                self.bad = 0
