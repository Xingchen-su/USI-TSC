# -*- coding: utf-8 -*-
"""
ReEnv — 语义对齐 + 黄灯保序 + 稳健映射 + 轻度过渡平滑
观测向量结构（固定 32 维）：
[ sem_one_hot(8), density(8), queue(8), head_since_green(8) ] ∈ [0,1]
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from gymnasium import spaces

# ========= 常量 =========
VEH_LEN   = 7.5       # 车辆等效长度(m)
MIN_GAP   = 2.5       # 最小车距(m)
CELL_LEN  = VEH_LEN + MIN_GAP  # 单车占用长度
EPS       = 1e-6
BETA_OCC  = 0.3       # 密度融合占有率权重
V_THR     = 1.0       # m/s，均速低于此阈值更像排队
GAM_LOW   = 1.2       # 低速放大
GAM_HIGH  = 0.8       # 非低速略降权
TAU_S     = 5.0       # s，EMA 时间常数
SEM_SMOOTH_RHO = 0.5  # 绿灯切换当步的前8维过渡平滑系数 ρ∈[0,1)

# 展平顺序（与 lane_dir 对齐）
DIRS   = ["north", "east", "south", "west"]
TURNS  = ["l", "s"]
PAIRS  = [(d, t) for d in DIRS for t in TURNS]  # NESW × (l,s)


class CustomObservationFunction:
    """
    sumo-rl 约定：
      __init__(ts, **kwargs)
      __call__(tls_id=None) -> np.ndarray
      observation_space() -> spaces.Box
    """

    def __init__(
        self,
        ts,
        lane_dir_map: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
        tls_action_spec: Optional[Dict[str, Any]] = None,
        enable_sem_smooth: bool = False,
    ):
        self.ts = ts
        self.id: str = getattr(ts, "id", None) or getattr(ts, "ts_id", "unknown")
        self.enable_sem_smooth = bool(enable_sem_smooth)

        # lane_dir: 优先外部 map，其次 ts.lane_dir
        from_map = (lane_dir_map or {}).get(self.id) if lane_dir_map else None
        self.lane_dir: Dict[str, Dict[str, List[str]]] = (
            from_map or getattr(ts, "lane_dir", {}) or {}
        )

        # 规范：tls_action_spec（名称与 run.py 保持一致）
        self.action_spec = tls_action_spec or {}

        # 语义头数固定 8
        self.A = 8
        self._pad_mask_head = [1] * self.A
        self._head_to_phase: List[int] = list(range(self.A))  # 兜底
        self._true_heads: List[int] = [i for i in range(self.A)]

        # 从规范中解析 per_agent 项
        item = None
        try:
            item = (self.action_spec or {}).get("per_agent", {}).get(self.id, None)
        except Exception:
            item = None

        if item is not None:
            pm = list(item.get("pad_mask_head", []))
            if len(pm) == self.A:
                self._pad_mask_head = [1 if int(v) != 0 else 0 for v in pm]
            self._true_heads = [i for i, v in enumerate(self._pad_mask_head) if v]

            h2p = list(item.get("head_to_phase", []))
            if len(h2p) == self.A:
                self._head_to_phase = [int(x) for x in h2p]
            elif len(h2p) == len(self._true_heads) and len(h2p) > 0:
                # 压缩表：需要扩展到全长
                full = [0] * self.A
                for k, head in enumerate(self._true_heads):
                    full[head] = int(h2p[k])
                self._head_to_phase = full

        # 预建反查：绝对相位 -> 语义 head
        self._phase_to_head: Dict[int, int] = {}
        for h in range(self.A):
            if self._pad_mask_head[h]:
                self._phase_to_head[int(self._head_to_phase[h])] = h

        # === 扫描当前 program，构建本地绿列表（k -> 绝对相位）与 k->head 快表 ===
        self._green_phases: List[int] = self._build_green_abs_list()

        self._k_to_head: Dict[int, int] = {}
        for k, abs_ph in enumerate(self._green_phases):
            h = self._phase_to_head.get(int(abs_ph), None)
            if h is not None and self._pad_mask_head[h]:
                self._k_to_head[int(k)] = int(h)


        # 缓存与 EMA
        self._lane_len: Dict[str, float] = {}
        self._ema_density: Optional[np.ndarray] = None
        self._ema_queue: Optional[np.ndarray] = None
        self._last_sem_onehot: Optional[np.ndarray] = None
        self._last_sem_head: Optional[int] = None
        # 记录每个语义 head 距离上次绿灯的步数，用于构造公平相关特征
        self._head_since_green: np.ndarray = np.zeros(self.A, dtype=np.float32)
        # 超过该步数后按 1.0 截断
        self._max_head_since_green: float = 100.0

        # 预分配观测（32维）：[8 sem][8 dens][8 queue][8 head_since]
        self._obs = np.zeros(32, dtype=np.float32)


    # 对外空间：固定 32 维
    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(32,), dtype=np.float32)


    # ===== 工具 =====
    def _get_sumo(self):
        sumo = getattr(self.ts, "sumo", None)
        if sumo is None:
            sumo = getattr(self.ts, "_sumo", None)
        if sumo is None and hasattr(self.ts, "env"):
            sumo = getattr(self.ts.env, "sumo", None)
        return sumo

    def _precache_lane_lengths(self):
        sumo = self._get_sumo()
        if sumo is None:
            return
        for d, t in PAIRS:
            for ln in self.lane_dir.get(d, {}).get(t, []):
                if ln not in self._lane_len:
                    try:
                        self._lane_len[ln] = float(sumo.lane.getLength(ln))
                    except Exception:
                        self._lane_len[ln] = 1.0  # 兜底

    def _lane_group_aggregate(self, lanes: List[str]) -> Tuple[float, float]:
        """
        对单个 (方向, 转向) 组返回 (density, queue) ∈ [0,1]
        旧版风格：长度加权 + (veh/cap, occ) 混合 + 速度门控 halting 比例
        """
        sumo = self._get_sumo()
        if sumo is None or not lanes:
            return 0.0, 0.0

        # 计算总长度
        tot_len = 0.0
        for ln in lanes:
            L = float(self._lane_len.get(ln, 0.0))
            if L <= 0.0:
                try:
                    L = float(sumo.lane.getLength(ln))
                except Exception:
                    L = 1.0
                self._lane_len[ln] = L
            tot_len += L
        if tot_len <= 0.0:
            return 0.0, 0.0

        dens = 0.0
        queu = 0.0
        for ln in lanes:
            L   = float(self._lane_len.get(ln, 1.0))
            w   = L / tot_len
            cap = max(L / CELL_LEN, 1.0)

            try:
                veh   = float(sumo.lane.getLastStepVehicleNumber(ln))
                occ_p = float(sumo.lane.getLastStepOccupancy(ln))  # %
                occ   = occ_p / 100.0
                halt  = float(sumo.lane.getLastStepHaltingNumber(ln))
                vavg  = float(sumo.lane.getLastStepMeanSpeed(ln))
            except Exception:
                veh = occ = halt = 0.0
                vavg = V_THR

            dens_lane = (1.0 - BETA_OCC) * (veh / cap) + BETA_OCC * occ
            gamma     = GAM_LOW if vavg <= V_THR else GAM_HIGH
            queu_lane = (halt / cap) * gamma

            dens += w * dens_lane
            queu += w * queu_lane

        return float(np.clip(dens, 0.0, 1.0)), float(np.clip(queu, 0.0, 1.0))

    def _compute_groups_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        dens_raw = np.zeros(8, dtype=np.float32)
        qued_raw = np.zeros(8, dtype=np.float32)
        for i, (d, t) in enumerate(PAIRS):
            lanes = self.lane_dir.get(d, {}).get(t, [])
            dens_raw[i], qued_raw[i] = self._lane_group_aggregate(lanes)
        return dens_raw, qued_raw

    def _ema_filter(self, x_raw: np.ndarray, ema_cache: Optional[np.ndarray]) -> np.ndarray:
        dt = float(getattr(self.ts, "delta_time", 1.0))
        alpha = float(np.exp(-max(dt, 1e-6) / TAU_S))
        if ema_cache is None:
            return x_raw.copy()
        return alpha * ema_cache + (1.0 - alpha) * x_raw

    def _current_phase_and_has_green(self) -> Tuple[Optional[int], int]:
        """
        返回 (绝对相位索引, has_green)。
        逻辑：优先使用 ts.green_phase -> green_phases[k] 得到绝对相位；
             若不可用，再回退到 traci.getPhase(self.id)。
        """
        sumo = self._get_sumo()
        if sumo is None:
            return None, 0

        # has_green: 只看当前灯态字符串是否含 G/g
        has_green = 0
        try:
            state = str(sumo.trafficlight.getRedYellowGreenState(self.id))
            if ("G" in state) or ("g" in state):
                has_green = 1
        except Exception:
            pass

        # 优先：ts.green_phase（本地绿索引 k）→ 绝对相位
        abs_phase = None
        try:
            k = getattr(self.ts, "green_phase", None)
            if k is not None:
                k = int(k)
                if self._green_phases is not None and 0 <= k < len(self._green_phases):
                    abs_phase = int(self._green_phases[k])
        except Exception:
            abs_phase = None

        # 兜底：直接询问 SUMO 的当前绝对相位
        if abs_phase is None:
            try:
                abs_phase = int(sumo.trafficlight.getPhase(self.id))
            except Exception:
                abs_phase = None

        return abs_phase, has_green

    def _build_green_abs_list(self) -> List[int]:
        """
        读取当前 program 的完整相位定义，按顺序提取“绿相位”的绝对索引列表。
        规则：state 中包含 'G' 或 'g'，且不包含 'y' 或 'Y'（排除黄灯）。
        """
        sumo = self._get_sumo()
        if sumo is None:
            return []

        # 当前 programID
        try:
            prog_id = str(sumo.trafficlight.getProgram(self.id))
        except Exception:
            prog_id = None

        # 完整定义（不同 SUMO 版本函数名可能不同，做双重兜底）
        try:
            defs = list(sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id))
        except Exception:
            try:
                defs = list(sumo.trafficlight.getAllProgramLogics(self.id))
            except Exception:
                defs = []

        # 选中当前 program，找不到就用第一个
        logic = None
        for lg in defs:
            try:
                if prog_id is None or str(getattr(lg, "programID", None)) == prog_id:
                    logic = lg
                    break
            except Exception:
                continue
        if logic is None and len(defs) > 0:
            logic = defs[0]

        phases = []
        try:
            phases = list(getattr(logic, "phases", []) or [])
        except Exception:
            phases = []

        green_abs: List[int] = []
        for idx, ph in enumerate(phases):
            try:
                st = str(getattr(ph, "state", "") or "")
            except Exception:
                st = ""
            has_green  = ("G" in st) or ("g" in st)
            has_yellow = ("y" in st) or ("Y" in st)
            if has_green and not has_yellow:
                green_abs.append(int(idx))

        return green_abs

    def _phase_to_sem_head(self, abs_phase: Optional[int]) -> Optional[int]:
        """
        绝对相位 -> 语义 head（0..7），找不到返回 None
        """
        if abs_phase is None:
            return None
        if abs_phase in self._phase_to_head:
            return int(self._phase_to_head[abs_phase])
        # 容错：再扫一遍全长表
        try:
            for h in range(self.A):
                if self._pad_mask_head[h] and int(self._head_to_phase[h]) == int(abs_phase):
                    return h
        except Exception:
            pass
        return None

    # ===== 主入口 =====
    def __call__(self, tls_id: Optional[str] = None) -> np.ndarray:
        # 预热 lane 长度缓存
        self._precache_lane_lengths()

        # 相位与是否有绿
        abs_phase, has_green = self._current_phase_and_has_green()

        # 前8维：语义 one-hot（黄灯/全红保序；绿灯切换当步过渡）
        sem_onehot_new = np.zeros(8, dtype=np.float32)

        if has_green == 1:
            # 优先：ts.green_phase (本地绿索引 k) → head
            h = None
            try:
                k = int(getattr(self.ts, "green_phase", None))
                h = self._k_to_head.get(k, None)
            except Exception:
                h = None

            # 回退：绝对相位 -> head（依赖 _phase_to_head）
            if h is None:
                h = self._phase_to_sem_head(abs_phase)

            # 稳健回退：沿用上一帧或首个可用 head
            if h is None:
                if self._last_sem_head is not None and self._pad_mask_head[self._last_sem_head]:
                    h = self._last_sem_head
                else:
                    usable = self._true_heads
                    h = usable[0] if len(usable) > 0 else 0

            sem_onehot_new[h] = 1.0

            # 绿灯切换当步：柔性过渡（防“硬跳”）
            if self.enable_sem_smooth and self._last_sem_onehot is not None:
                if int(np.argmax(self._last_sem_onehot)) != int(h):
                    sem_onehot_new = (
                        SEM_SMOOTH_RHO * self._last_sem_onehot
                        + (1.0 - SEM_SMOOTH_RHO) * sem_onehot_new
                    )

            self._last_sem_head = int(np.argmax(sem_onehot_new))

        else:
            # 无绿：黄灯/全红，保序保持
            if self._last_sem_onehot is not None:
                sem_onehot_new = self._last_sem_onehot.copy()
            else:
                usable = self._true_heads
                h0 = usable[0] if len(usable) > 0 else 0
                sem_onehot_new[h0] = 1.0
                self._last_sem_head = h0

        # 记住当前 one-hot
        self._last_sem_onehot = sem_onehot_new.copy()

        # 更新每个 head 距离上次被服务的步数（用于公平相关观测）
        # 所有 head 先 +1
        self._head_since_green += 1.0
        # 当前执行的 head 视为“被服务”，距离清零
        cur_head = int(np.argmax(sem_onehot_new))
        if 0 <= cur_head < self.A:
            self._head_since_green[cur_head] = 0.0
        # 归一到 [0,1]，超过 _max_head_since_green 后按 1.0 截断
        head_since_norm = np.clip(self._head_since_green / self._max_head_since_green, 0.0, 1.0)

        # 后16维：密度/排队（旧版计算）
        dens_raw, qued_raw = self._compute_groups_raw()
        dens_ema = self._ema_filter(dens_raw, self._ema_density)
        queu_ema = self._ema_filter(qued_raw, self._ema_queue)
        self._ema_density = dens_ema
        self._ema_queue   = queu_ema


        # 组装观测： [8 sem][8 dens][8 queue][8 head_since]
        obs = self._obs
        obs[:8] = np.clip(sem_onehot_new, 0.0, 1.0)
        obs[8:16]  = np.clip(dens_ema, 0.0, 1.0)
        obs[16:24] = np.clip(queu_ema, 0.0, 1.0)
        obs[24:32] = head_since_norm
        return obs.copy()

