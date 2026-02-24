# -*- coding: utf-8 -*-
"""
多交叉口评估脚本（适用于你的 MacLight 项目）
------------------------------------------------
特性：
1. 指标：
   - ATT（Average Travel Time）
   - AWT（Average Waiting Time）
   - AS  （Average Speed）
   - TP  （Throughput）
   - queue：时间平均排队车数（全网所有 lane 上 halting 车辆数的时间平均）

2. 权重加载采用：每个路口一个 agent
   - ckpt 里如果是：{"agent": {tls_id: MacLight对象, ...}} 或 {"agent": {tls_id: state_dict, ...}}
   - 则评估时 self.agents[tls_id] = agent_for_that_tls
   - 评估循环里：每个 agent_name 用对应 self.agents[agent_name] 来 act_eval

3. 保留了单一 agent 的兼容逻辑（如果以后你有“共享参数”的实验，也能复用此脚本）
"""

use_gui = False
import os
if use_gui == False:
    os.environ['LIBSUMO_AS_TRACI'] = '1'

import os, sys
import time
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch, sumo_rl

import traci  # 使用 libsumo 作为 traci

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from random_block import BlockStreet
from agent import MacLight
from net import PolicyNet, ValueNet
from ReEnv import CustomObservationFunction
from lane_dir.lane_dir import build_all_tls_lane_dirs
from Eva_sim.eval_saver import EvalSaver
from Eva_sim.sim_metrics import SimulationMetrics
from Eva_sim.step_action_logger import StepActionLogger
from utils.spec_guard import ensure_action_spec


# ====================== 日志工具 ======================

class StepActionLogger:
    """
    记录每一步每个路口的动作，格式与 GESA 的 step_actions.csv 相同：
    time,intersection,action
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, "step_actions.csv")
        self._f = open(self.file_path, "w", encoding="utf-8-sig")
        self._f.write("time,intersection,action\n")
        self._f.flush()

    def log_step(self, sim_time: float, agent_name: str, action: int):
        line = f"{sim_time:.1f},{agent_name},{int(action)}\n"
        self._f.write(line)
        self._f.flush()

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()


class EvalSaver:
    """
    保存每个 episode 的评估结果
    字段：
        timestamp: 写入时间
        episode  : 第几个 episode
        seed     : 随机种子
        return   : episode 累计 reward（路口平均后加总）
        waiting  : AWT（平均等待时间，秒/车）
        queue    : 时间平均排队车数
        speed    : AS（平均速度，m/s）
        ATT      : 平均旅行时间（秒/车）
        AWT      : 同 waiting
        AS       : 同 speed
        TP       : 完成行程的车辆数
        route_info: 场景信息 JSON（可自定义 net/route 名称等）
        weight_path: 当前评估使用的权重文件路径
    """

    def __init__(self, base_dir: str, ckpt_path: str):
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        parent_name = os.path.basename(ckpt_dir)  # 用父目录名做子目录
        self.log_dir = os.path.join(base_dir, parent_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.file_path = os.path.join(self.log_dir, "eval_results.csv")
        self.columns = [
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
            "weight_path",
        ]

        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8-sig") as f:
                f.write(",".join(self.columns) + "\n")

    def save_episode(
        self,
        episode: int,
        seed: int,
        episode_return: float,
        waiting: float,
        queue: float,
        speed: float,
        ATT: float,
        AWT: float,
        AS: float,
        TP: int,
        route_info: dict,
        weight_path: str,
    ):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            str(episode),
            str(seed),
            f"{episode_return:.6f}",
            f"{waiting:.6f}",
            f"{queue:.6f}",
            f"{speed:.6f}",
            f"{ATT:.6f}",
            f"{AWT:.6f}",
            f"{AS:.6f}",
            str(TP),
            json.dumps(route_info, ensure_ascii=False),
            weight_path,
        ]
        with open(self.file_path, "a", encoding="utf-8-sig") as f:
            f.write(",".join(row) + "\n")


class EpisodeMetrics:
    """
    GESA 风格的单个 episode 指标累加器：
    - ATT / AWT / AS / TP
    - queue：时间平均排队车辆数
    """

    def __init__(self, wait_th: float = 0.1):
        self.wait_th = wait_th  # 速度小于该阈值视为“在等待”
        self.reset()

    def reset(self):
        # queue 相关
        self.cum_queue = 0.0
        self.step_cnt = 0

        # 旅行 / 等待 / 速度
        self.depart_time = {}        # {veh_id: depart_time}
        self.veh_wait = {}           # {veh_id: waiting_time}
        self.total_travel_time = 0.0
        self.total_wait_time = 0.0
        self.completed = 0

        self.speed_sum = 0.0
        self.speed_cnt = 0

        # 时间
        self.prev_time = None

    def step(self, traci_conn):
        """
        在每个仿真步调用：
        先根据当前仿真时间算 step_len，再更新 queue / ATT / AWT / AS / TP 的原始累加量。
        """
        sim_time = traci_conn.simulation.getTime()

        # 初始化 prev_time
        if self.prev_time is None:
            self.prev_time = sim_time

        step_len = max(0.0, sim_time - self.prev_time)
        self.prev_time = sim_time

        # 1) queue：全网停车车辆数
        queue_cnt_step = 0
        for lane_id in traci_conn.lane.getIDList():
            queue_cnt_step += traci_conn.lane.getLastStepHaltingNumber(lane_id)
        self.cum_queue += queue_cnt_step
        self.step_cnt += 1

        # 2) 刚出发的车辆
        for vid in traci_conn.simulation.getDepartedIDList():
            self.depart_time[vid] = sim_time
            self.veh_wait.setdefault(vid, 0.0)

        # 3) 在网内车辆：速度和等待时间
        for vid in traci_conn.vehicle.getIDList():
            v = traci_conn.vehicle.getSpeed(vid)
            self.speed_sum += v
            self.speed_cnt += 1
            if v < self.wait_th:
                self.veh_wait[vid] = self.veh_wait.get(vid, 0.0) + step_len

        # 4) 到达车辆：累计旅行时间和等待时间
        for vid in traci_conn.simulation.getArrivedIDList():
            dt = sim_time - self.depart_time.pop(vid, sim_time)
            self.total_travel_time += max(0.0, dt)
            self.total_wait_time += self.veh_wait.pop(vid, 0.0)
            self.completed += 1

    def get_metrics(self):
        # episode 结束时调用
        ATT = self.total_travel_time / self.completed if self.completed > 0 else 0.0
        AWT = self.total_wait_time / self.completed if self.completed > 0 else 0.0
        AS = self.speed_sum / self.speed_cnt if self.speed_cnt > 0 else 0.0
        TP = int(self.completed)
        queue_time_avg = self.cum_queue / self.step_cnt if self.step_cnt > 0 else 0.0

        return {
            "ATT": ATT,
            "AWT": AWT,
            "AS": AS,
            "TP": TP,
            "queue": queue_time_avg,
        }


# ====================== 评估主类 ======================

class EvaluatorSim:
    """
    多智能体评估器

    - env: sumo_rl 并行环境（pettingzoo 风格）
    - agent: 单一 MacLight（作为模板 / 兜底）
    - 我们会尽量从 ckpt 里恢复出 {路口ID: MacLight对象 或 state_dict} 的 self.agents
    """

    def __init__(
        self,
        env,
        agent,
        eval_episodes: int,
        ckpt_path: str,
        spec=None,
        net_name: str = None,
        route_name: str = None,
        log_base_dir: str = "eval_logs",
        seed: int = 42,
    ):
        self.env = env
        self.agent = agent          # 单一 agent（模板/兜底）
        self.agents = None          # 多路口 agent 字典（方案 A）
        self.eval_episodes = eval_episodes
        self.ckpt_path = ckpt_path
        self.spec = spec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed

        # 基于 GESA 的 episode 指标计算器
        self.emetrics = EpisodeMetrics(wait_th=0.1)

        # 保存 eval 结果
        self.saver = EvalSaver(base_dir=log_base_dir, ckpt_path=ckpt_path)

        # 场景信息（可以按需要修改）
        self.route_info = {
            "net": net_name if net_name is not None else "unknown_net",
            "route": route_name if route_name is not None else "unknown_route",
        }

        # step 动作日志
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        self.log_dir = os.path.join(log_base_dir, parent_name)
        self.step_logger = StepActionLogger(self.log_dir)

    # ---------- 权重加载（方案 A，修正 state_dict 分支） ----------

    def load_weights(self, model_path: str):
        """
        支持三种情况：
        1. checkpoint["agent"] 是单一 MacLight 对象  --> 用到 self.agent 上；
        2. checkpoint["agent"] 是 {路口ID: MacLight对象} --> 方案 A：self.agents = 这个 dict；
        3. checkpoint["agent"] 是 {路口ID: state_dict}  --> 方案 A：为每个路口复制一份模板 agent 并加载到 actor/critic。
        """
        print(f"Loading weights from: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)

        if "agent" not in ckpt:
            raise RuntimeError("checkpoint 中未找到 'agent' 字段，请检查保存逻辑。")
        agent_blob = ckpt["agent"]

        # 情况 1：整体就是一个 agent 对象（例如共享参数的版本）
        if not isinstance(agent_blob, dict):
            self.agent = agent_blob
            print("检测到 ckpt['agent'] 是一个完整的 agent 对象，已直接替换 self.agent。")
            return

        # 情况 2：dict，可能是 {tls_id: agent} 或 {tls_id: state_dict}
        if len(agent_blob) == 0:
            raise RuntimeError("checkpoint['agent'] 是空 dict，没有任何路口的 agent。")

        first_key, first_val = next(iter(agent_blob.items()))

        # 2A: value 看起来是“完整的 agent 对象”（有 act_eval）
        if hasattr(first_val, "act_eval"):
            all_are_agents = all(hasattr(v, "act_eval") for v in agent_blob.values())
            if all_are_agents:
                self.agents = agent_blob
                print(f"检测到 ckpt['agent'] 为每路口一个完整 agent，对应 {len(self.agents)} 个路口，已按交叉口恢复权重。")
                return

        # 2B: value 是 state_dict（更符合你现在 win32.pt 的情况）
        if isinstance(first_val, dict):
            import copy
            self.agents = {}
            for tls_id, sd in agent_blob.items():
                ag = copy.deepcopy(self.agent)  # 用模板 agent 复制一份
                loaded = False

                # 优先判断是否为 {"actor": ..., "critic": ...} 结构
                if hasattr(ag, "actor") and isinstance(sd, dict) and "actor" in sd:
                    try:
                        ag.actor.load_state_dict(sd["actor"])
                        loaded = True
                    except Exception as e:
                        print(f"[warn] 加载 {tls_id} 的 actor 失败：{e}")

                if hasattr(ag, "critic") and isinstance(sd, dict) and "critic" in sd:
                    try:
                        ag.critic.load_state_dict(sd["critic"])
                        loaded = True
                    except Exception as e:
                        print(f"[warn] 加载 {tls_id} 的 critic 失败：{e}")

                # 如果 sd 不是 {"actor":..., "critic":...}，尝试把整个 sd 当成 actor 的 state_dict
                if not loaded and hasattr(ag, "actor"):
                    try:
                        ag.actor.load_state_dict(sd)
                        loaded = True
                        print(f"[info] 将 {tls_id} 的 state_dict 视为 actor 权重加载（critic 使用模板初始化）。")
                    except Exception:
                        pass

                if not loaded:
                    # 到这里说明我们完全没法识别 sd 的结构，给你一些提示信息
                    raise RuntimeError(
                        f"无法将权重加载到 MacLight，路口 {tls_id} 的 state_dict 结构不识别，"
                        f"示例 keys: {list(sd.keys())[:5]}"
                    )

                self.agents[tls_id] = ag

            print(f"检测到 ckpt['agent'] 为 {{tls_id: state_dict}} 结构，已为 {len(self.agents)} 个路口创建 agent 并加载对应权重。")
            return

        # 兜底：其它情况暂时按单一 agent 处理
        self.agent = first_val
        print("ckpt['agent'] 是 dict，但未识别出多路口结构，已用第一个 value 替换 self.agent。")

    # ---------- 辅助：获取 TraCI 连接 ----------

    @staticmethod
    def _get_traci_conn():
        """
        这里直接返回 traci（libsumo_as_traci 已在文件顶部设置）。
        """
        return traci

    # ---------- 主评估循环 ----------

    def run_simulation(self):
        """
        按 GESA 风格评估：
        - 每个 episode 运行完整仿真（直到 env 自己终止）
        - 用 EpisodeMetrics 统计 ATT/AWT/AS/TP + queue
        - 每步记录动作到 step_actions.csv
        - 每个 episode 写一行 eval_results.csv
        """
        for ep in range(self.eval_episodes):
            print(f"===== 开始第 {ep + 1}/{self.eval_episodes} 次评估 =====")

            # 设定随机种子（保证可复现）
            seed = self.seed + ep
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # reset 环境
            # state, _ = self.env.reset(seed=seed, options={})
            state, _ = self.env.reset(seed=seed)
            self.emetrics.reset()

            # 获取 traci 连接，初始化时间
            traci_conn = self._get_traci_conn()
            _ = traci_conn.simulation.getTime()  # 触发一次，便于 EpisodeMetrics 内部初始化 prev_time

            episode_return = 0.0
            done_flag = False

            while not done_flag:
                traci_conn = self._get_traci_conn()
                # 1) 用当前仿真时间更新指标（与 GESA 相同：先统计，再决策）
                self.emetrics.step(traci_conn)

                # 2) 按路口选择动作
                actions = {}
                sim_time = traci_conn.simulation.getTime()

                for agent_name in state.keys():
                    if self.agents is not None:
                        # 有多路口 agent：按路口取对应的一个；如果缺，就退回共享 self.agent
                        agent_obj = self.agents.get(agent_name, self.agent)
                    else:
                        # 没有多路口结构：所有路口共用 self.agent
                        agent_obj = self.agent

                    act, _, _ = agent_obj.act_eval(
                        state[agent_name],
                        agent_name,
                        spec=self.spec
                    )
                    actions[agent_name] = act

                    # 记录动作
                    if self.step_logger is not None:
                        self.step_logger.log_step(sim_time, agent_name, act)

                # 3) 推进环境一步
                next_state, reward, terminated, truncated, info = self.env.step(actions)

                # reward: {agent_name: scalar}
                if isinstance(reward, dict):
                    r_vals = list(reward.values())
                    if len(r_vals) > 0:
                        episode_return += float(np.mean(r_vals))
                else:
                    episode_return += float(np.mean(reward))

                state = next_state

                # 4) 终止条件（pettingzoo 风格）
                if isinstance(terminated, dict):
                    done_flag = all(terminated.values()) or all(truncated.values())
                else:
                    done_flag = bool(terminated or truncated)

            # ----- episode 结束，汇总指标 -----
            m = self.emetrics.get_metrics()
            ATT = m["ATT"]
            AWT = m["AWT"]
            AS = m["AS"]
            TP = m["TP"]
            queue_time_avg = m["queue"]

            waiting_csv = AWT
            queue_csv = queue_time_avg
            speed_csv = AS

            print(
                f"[Eval] ep={ep} seed={seed} "
                f"Return={episode_return:.2f}  "
                f"ATT={ATT:.2f}s  AWT={AWT:.2f}s  AS={AS:.2f}m/s  TP={TP}  "
                f"QueueAvg={queue_time_avg:.2f}"
            )

            self.saver.save_episode(
                episode=ep,
                seed=seed,
                episode_return=episode_return,
                waiting=waiting_csv,
                queue=queue_csv,
                speed=speed_csv,
                ATT=ATT,
                AWT=AWT,
                AS=AS,
                TP=TP,
                route_info=self.route_info,
                weight_path=self.ckpt_path,
            )

        # 关闭 step logger
        self.step_logger.close()


# ====================== 命令行入口 / main ======================

def main():
    """
    主函数：初始化环境、代理，加载权重并运行评估。
    """

    spec_path = "lane_dir/outputs_specs"
    net_file = r"env\map_ff\ff.net.xml"
    route_file = r"env\map_ff\ff_normal.rou.xml"

    block_task = False
    block_num = 8

    # net_file = r"env\cologne8\cologne8.net.xml"
    # route_file = r"env\cologne8\cologne8.rou.xml"

    ckpt_file = r"ckpt\block\Clean\2025-11-24_18-03-12\43_2025-11-24_18-03-12_win32.pt"

    if 'cologne8' in net_file:
        begin_time = 25200
    elif 'ingo' in net_file:
        begin_time = 57600
    else:
        begin_time = 0

    net_name = os.path.basename(os.path.normpath(net_file))
    route_name = os.path.basename(os.path.normpath(route_file))

    # 构建 lane_dir_map 和 spec
    lane_dir_map = build_all_tls_lane_dirs(net_file)
    spec, _ = ensure_action_spec(
        net_xml=net_file,
        desired=spec_path,
        auto_generate=True
    )

    # 初始化环境
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        num_seconds=3600,
        begin_time=begin_time,
        use_gui=use_gui,
        observation_class=lambda ts: CustomObservationFunction(
            ts,
            lane_dir_map=lane_dir_map,
            tls_action_spec=spec
        )
    )
    agent_names = env.possible_agents
    state_dim = max(env.observation_space(agent_name).shape[0] for agent_name in agent_names)
    hidden_dim = 2 * state_dim
    action_dim = 8  

    agent = MacLight(
        policy_net=PolicyNet(state_dim, hidden_dim, action_dim),
        critic_net=ValueNet(state_dim, hidden_dim)
    )
    agent.bind_spec(spec)

    if block_task:
        env = BlockStreet(env, block_num, 3600)

    evaluator_sim = EvaluatorSim(
        env,
        agent,
        eval_episodes=1,
        ckpt_path=ckpt_file,  
        spec=spec,
        net_name=net_name,
        route_name=route_name,
    )

    evaluator_sim.load_weights(ckpt_file)
    evaluator_sim.run_simulation()


if __name__ == "__main__":
    main()
