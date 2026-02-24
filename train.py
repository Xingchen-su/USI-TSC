
import os, sys, time, hashlib, xml.etree.ElementTree as ET
from typing import Dict, Any, Iterable, Optional, Tuple, List

import csv 
import numpy as np
import torch, traci, copy
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import action_spec_utils as asu
from utils import spec_guard
from net import VAE  



# ---------------- 小工具 ----------------
def _rel(path: str, base: Optional[str] = None) -> str:
    if base is None:
        base = os.getcwd()
    try:
        return os.path.relpath(os.path.abspath(path), base).replace("\\", "/")
    except Exception:
        return path

def _make_train_step_csv(file_path: str) -> str:
    """
    在 file_path/step_actions/ 下创建/获取 train_step_actions.csv，
    若文件不存在则写入表头。

    file_path 由 run 传入，形如:
      data/{task}/{model}/{time_stamp}
    """
    step_dir = os.path.join(file_path, "step_actions")
    os.makedirs(step_dir, exist_ok=True)

    csv_path = os.path.join(step_dir, "train_step_actions.csv")
    if not os.path.exists(csv_path):
        # 第一次创建时写表头
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([
                "episode",      # 当前 episode 序号
                "seed",         # run 使用的随机种子
                "step",         # episode 内步数（0,1,2,...）
                "time",         # traci 仿真时间（秒）
                "intersection", # 路口 ID (tls_id)
                "sem_head",     # 实际执行的语义动作 head (0..7)
                "phase_idx",    # 在 net / SUMO 中的真实相位索引
                "env_action",   # 传给 env.step 的 ordinal (0..n-1)
                "override_by_starve",  # 0/1：本步动作是否由饥饿逻辑强制覆盖
            ])
    return csv_path

def _phase_to_head(spec: Dict[str, Any], tls_id: str, a_env: int) -> int:
    """
    反查：真实相位 a_env -> 语义 a_sem
    兼容全长度与压缩式 head_to_phase 表。
    """
    item = spec["per_agent"][tls_id]
    A = int(spec["global_action_dim"])
    table = item["head_to_phase"]
    mask = item["pad_mask_head"]  # 长度 A 的 0/1 列表

    # 全长度表，table 长度 == A
    if len(table) == A:
        for sem in range(A):
            if mask[sem] and int(table[sem]) == int(a_env):
                return sem
        # 容错：找不到就返回第一个合法位
        for sem in range(A):
            if mask[sem]:
                return sem
        return 0

    # 压缩表：len(table) == 可用 head 数；与 mask 的 True 位置一一对应
    true_positions = [i for i, v in enumerate(mask) if v]
    for k, env_phase in enumerate(table):
        if int(env_phase) == int(a_env):
            return true_positions[k]
    # 容错
    return true_positions[0] if true_positions else 0

def _head_to_phase(spec, tls_id, a_sem: int) -> int:
    item  = spec["per_agent"][tls_id]
    A     = int(spec["global_action_dim"])
    table = item["head_to_phase"]         # 可能是“全长表”或“压缩表”
    mask  = item["pad_mask_head"]         # 长度 A 的 0/1 合法位

    if len(table) == A:                   # 全长：直接查
        return int(table[a_sem])
    # 压缩：先把语义位在合法位中的排名找出来
    true_pos = [i for i,v in enumerate(mask) if v]
    k = true_pos.index(a_sem)
    return int(table[k])

def _phase_order(spec, tls_id: str):
    """legal_mask_phase==1 的下标按升序构成 env ordinal→phase_idx 的映射顺序"""
    legal = spec["per_agent"][tls_id]["legal_mask_phase"]
    return [i for i, v in enumerate(legal) if v == 1]

def _env_to_phase(spec, tls_id: str, env_act: int) -> int:
    """ordinal(0..n-1) -> phase_idx"""
    order = _phase_order(spec, tls_id)
    return int(order[int(env_act)])

def _phase_to_env(spec, tls_id: str, phase_idx: int) -> int:
    """phase_idx -> ordinal(0..n-1)"""
    order = _phase_order(spec, tls_id)
    return int(order.index(int(phase_idx)))

# ---------------- 规范加载/校验 ----------------
def _load_or_build_spec(spec_hint: str, net_file: str) -> Tuple[Dict[str, Any], int]:
    """
    获取与 net_file 匹配的规范。
    返回：(spec, A)
    """
    # 允许传入单个文件
    if os.path.isfile(spec_hint) and spec_hint.lower().endswith(".json"):
        spec = asu.load_spec(spec_hint)
        if not spec_guard._spec_matches_net(spec, net_file):
            raise SystemExit(
                f"[FATAL] 指定规范与 net 不匹配：spec={_rel(spec_hint)}  net={_rel(net_file)}"
            )
        A = asu.action_dim(spec)
        print(f"[spec] 使用指定规范：{_rel(spec_hint)}  (A={A})")
        return spec, A

    # 否则当目录处理，并允许自动重建
    pick_root = spec_hint if os.path.isdir(spec_hint) else os.path.join("lane_dir", "outputs_specs")
    spec, chosen_path = spec_guard.ensure_action_spec(
        net_xml=net_file,
        desired=pick_root,
        reorder_py=os.path.join("lane_dir", "reorder_mask_builder.py"),
        out_root=pick_root,
        auto_generate=True,
    )
    A = asu.action_dim(spec)
    print(f"[spec] 已匹配当前 net：{_rel(chosen_path)}  (A={A})")
    return spec, A

def _append_transition(
    agent_ids,
    epi_training: bool,
    trans: Dict[str, Dict[str, torch.Tensor]],
    state, done, action_env, action_sem, next_state, reward,
    old_logp_sem=None  
):


    keys = ["states", "actions", "actions_sem", "old_logp_sem", "next_states", "rewards", "dones"]
    payloads = [state, action_env, action_sem, old_logp_sem, next_state, reward, done]

    for key, payload in zip(keys, payloads):
        for aid in agent_ids:
            t = payload[aid]
            if not torch.is_tensor(t):
                t = torch.tensor(t)
            if not epi_training:
                trans[key][aid] = t.unsqueeze(0)
            else:
                trans[key][aid] = torch.cat([trans[key][aid], t.unsqueeze(0)], dim=0)
    return trans

# ---------------- VAE ----------------
def _vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def _train_vae(model: VAE, optimizer: optim.Optimizer, data: torch.Tensor, epochs: int = 5) -> float:
    model.train()
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    total = 0.0
    for _ in range(epochs):
        for (x,) in dataloader:
            x = x.to(next(model.parameters()).device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = _vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
    return total

# ---------------- TLS 位置 → 网格 ----------------
def parse_tls_positions(net_file, agent_name, state_dim, T) -> List[Dict[str, float]]:
    tree = ET.parse(net_file)
    root = tree.getroot()
    tls_nodes = []
    for j in root.findall("junction"):
        if "traffic_light" in j.attrib.get("type", ""):
            tls_nodes.append({"id": j.attrib["id"], "x": float(j.attrib["x"]), "y": float(j.attrib["y"])})
    tls_nodes = [node for node in tls_nodes if node['id'] in agent_name]

    xs = sorted(set(n["x"] for n in tls_nodes))
    ys = sorted(set(n["y"] for n in tls_nodes))
    pos_map = {n["id"]: (n["y"], n["x"]) for n in tls_nodes}
    pos_y = {y: i for i, y in enumerate(ys)}
    pos_x = {x: i for i, x in enumerate(xs)}

    B, C, H, W = T, state_dim, len(ys), len(xs)
    tensor = torch.zeros((B, C, H, W), dtype=torch.float32)
    return tensor, pos_map, pos_y, pos_x


def reshape_whole_state_auto(state_dict: Dict[str, list], tls_nodes: str, state_dim: int, agent_name) -> torch.Tensor:
    """
    把 {tls_id: 1D 状态向量} 按 tls 的几何分布拼成网格张量 [1, C, H, W]
    """

    xs = sorted(set(n["x"] for n in tls_nodes))
    ys = sorted(set(n["y"] for n in tls_nodes))
    pos_map = {n["id"]: (n["y"], n["x"]) for n in tls_nodes}
    pos_y = {y: i for i, y in enumerate(ys)}
    pos_x = {x: i for i, x in enumerate(xs)}

    B, C, H, W = 1, state_dim, len(ys), len(xs)
    tensor = torch.zeros((B, C, H, W), dtype=torch.float32)
    for aid, vec in state_dict.items():
        if aid in pos_map:
            y, x = pos_map[aid]
            tensor[0, :, pos_y[y], pos_x[x]] = torch.tensor(vec, dtype=torch.float32)
    return tensor

# ---------------- 调试工具 ----------------
def audit_head_phase_mapping(spec, tls_id):
    """逐路口打印“动作映射对照表”"""
    item  = spec["per_agent"][tls_id]
    A     = int(spec["global_action_dim"])
    mask  = item["pad_mask_head"]            # [A] 0/1
    legal = item["legal_mask_phase"]         # [n_ph] 0/1
    table = item["head_to_phase"]            # len = A 或 可用head数
    true_heads = [i for i,v in enumerate(mask) if v]
    order = [i for i,v in enumerate(legal) if v==1]

    print(f"\n[TLS {tls_id}] A={A}, usable_heads={len(true_heads)}, legal_phases={len(order)}")
    print("env_action  phase_idx  sem_head  note")
    print("---------------------------------------")
    for k, ph in enumerate(order):
        if len(table)==A:
            candidates = [h for h in true_heads if int(table[h])==ph]
            sem = candidates[0] if candidates else -1
        else:
            sem = true_heads[table.index(ph)] if ph in table else -1
        note = "" if sem!=-1 else "(!) unmapped"
        print(f"{k:>9}  {ph:>9}  {sem:>8}  {note}")

# ---------------- 训练主过程 ----------------
def train_ours_agent(
    env: Any,
    agents: Any,                     
    agent_name: Iterable[str],
    representation: bool,
    writer: Any,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    file_path: str,
    evaluator: Any,

    net_file: str,
    vae: object ,
    spec_path: str = "lane_dir/outputs_specs",  # 可传目录或确切 json
    min_green_steps: int = 2,
    reward_helper=None,            
    reward_scheduler=None,         
    epoch_episodes: int = 1,       
):
    
    vae_opt = optim.Adam(vae.parameters(), lr=1e-3) if vae else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 选择/校验 规范，并下发给各 Agent
    spec, A = _load_or_build_spec(spec_path, net_file)
    for aid in agent_name:
        if hasattr(agents[aid], "bind_spec"):
            agents[aid].bind_spec(spec)
    step_csv_path = _make_train_step_csv(file_path)

    # 2) 统计容器
    actor_loss_list, critic_loss_list, vae_loss_list = [], [], []
    return_list, waiting_list, queue_list, speed_list, time_list, seed_list = [], [], [], [], [], []
    ATT_list, AWT_list, AS_list, TP_list = [], [], [], []

    # 3) 训练循环
    start_time = time.time()

    for ep in range(total_episodes):
        # ====== epoch 开始 ======
        if reward_helper is not None and (ep % epoch_episodes == 0):
            reward_helper.begin_epoch()
        

        epi_training = False
        transition = {
            "states": {a: 0 for a in agent_name},
            "actions": {a: 0 for a in agent_name},        # 真实相位 a_env（只做日志）
            "actions_sem": {a: 0 for a in agent_name},    # 语义动作 a_sem（PPO 用这个）
            "old_logp_sem": {a: 0 for a in agent_name},   # 采样当刻的 log π_old(a_sem|s)
            "next_states": {a: 0 for a in agent_name},
            "rewards": {a: 0 for a in agent_name},
            "dones": {a: 0 for a in agent_name},
            "global_emb": [],
        }

        
        step_idx = 0        # 本 episode 内步计数，用于 CSV 的 step 列
        max_green_steps = 4 
        cap_warmup_ep = 0 
        sys_stop_sum = 0.0  
        sys_wait_sum = 0.0  
        sys_speed_sum = 0.0 
        T = 0               

        # ---- ATT / AWT / AS / TP 统计容器（每个 episode 重置）----
        depart_time = {}
        veh_wait = {}
        WAIT_TH = 0.1
        total_travel_time = 0.0
        total_wait_time = 0.0
        completed = 0
        speed_sum = 0.0
        speed_cnt = 0

        # 减少 ATT 等指标的计算开销
        metric_interval = 1  
        enable_metric = (ep % metric_interval == 0)

        # 获取 TraCI 连接
        try:
            traci_conn = getattr(env, "sim", None) or traci
            prev_time = traci_conn.simulation.getTime()
        except Exception:
            traci_conn = None
            prev_time = 0.0

        # 最长饥饿时间计数（按 head 维度
        # 最长饥饿时间计数（按 head 维度
        # 1 step = 5s → 120s / 5 = 24 step
        STARVE_MAX_STEPS = 24
        STARVE_REWARD_PENALTY = 2.0

        starve_steps = {aid: [0 for _ in range(A)] for aid in agent_name}

        override_by_starve = {aid: False for aid in agent_name}
        
        episode_return = 0.0
        hold_counter = {aid: 0 for aid in agent_name}
        last_env_action = {aid: None for aid in agent_name}
        _ret = env.reset(seed=seed)
        state, _ = _ret if isinstance(_ret, tuple) else (_ret, {})
        done, truncated = False, False

        while not (done | truncated):
            action_env = {}
            action_sem = {}
            old_logp_sem = {}  # 保存采样当刻的 old_logp(语义)

            # 每步先累加所有 head 的饥饿计数
            for aid in agent_name:
                for h in range(A):
                    starve_steps[aid][h] += 1

            for aid in agent_name:
                # 获取策略输出的动作
                state_np = state[aid]
                a_env, a_sem, probs = agents[aid].act_train(state_np, aid)

                # === 统一候选：以“语义 -> 相位 -> ordinal”为准 ===
                phase_from_sem = _head_to_phase(spec, aid, int(a_sem))          # head -> phase_idx
                cand_env = _phase_to_env(spec, aid, phase_from_sem)              # phase_idx -> ordinal
                cand_sem = int(a_sem)

                # === 最短绿保持 ===
                if (last_env_action[aid] is not None
                    and cand_env != last_env_action[aid]
                    and hold_counter[aid] < min_green_steps):
                    # 保持上一相位（ordinal）
                    action_env[aid] = int(last_env_action[aid])
                    # 把上一相位(ordinal)反查为语义位
                    prev_phase = _env_to_phase(spec, aid, action_env[aid])
                    action_sem[aid] = int(_phase_to_head(spec, aid, prev_phase))
                    hold_counter[aid] += 1

                else:
                    # === 最长绿约束（硬上限，温和挑法）===
                    use_cap = (max_green_steps is not None) and (ep >= cap_warmup_ep)
                    if (use_cap
                        and last_env_action[aid] == cand_env        # 策略也想继续保持
                        and hold_counter[aid] >= max_green_steps):  # 已达上限
                        # 当前相位对应的语义位
                        curr_phase = _env_to_phase(spec, aid, last_env_action[aid])
                        curr_sem = _phase_to_head(spec, aid, curr_phase)
                        # 候选语义：可用且 != 当前
                        mask = spec["per_agent"][aid]["pad_mask_head"]
                        candidates = [s for s, m in enumerate(mask) if m and s != curr_sem]
                        if candidates:
                            # 选当步分布中概率最高的候选
                            best_sem = max(candidates, key=lambda s: float(probs[s].item()))
                            best_phase = _head_to_phase(spec, aid, best_sem)
                            best_env = _phase_to_env(spec, aid, best_phase)
                            action_sem[aid] = int(best_sem)
                            action_env[aid] = int(best_env)
                        else:
                            # 极端兜底：退回策略候选
                            action_sem[aid] = cand_sem
                            action_env[aid] = cand_env
                        # 切换了，重置计数
                        hold_counter[aid] = 1
                        last_env_action[aid] = action_env[aid]
                    else:
                        # 常规执行策略候选
                        action_env[aid] = int(cand_env)
                        action_sem[aid] = int(cand_sem)
                        if last_env_action[aid] == action_env[aid]:
                            hold_counter[aid] += 1
                        else:
                            hold_counter[aid] = 1
                        last_env_action[aid] = action_env[aid]
                        
                # === 最长饥饿时间约束（基于 head + 当前队列） ===
                # 只在“有明显队列且饥饿时间超阈值”的 head 上触发 starve
                mask = spec["per_agent"][aid]["pad_mask_head"]
                starving_heads = [
                    h for h, m in enumerate(mask)
                    if m and starve_steps[aid][h] >= STARVE_MAX_STEPS
                ]

                if starving_heads:
                    QUEUE_OFFSET = 16  # 观测中 queue 段的起始索引：[8 sem][8 dens][8 queue]
                    QUEUE_THRESH = 0.1  # 归一后队列超过该值才认为“有明显 backlog”
                    # 只保留当前队列足够大的饥饿 head
                    starving_heads = [
                        h for h in starving_heads
                        if state_np[QUEUE_OFFSET + h] >= QUEUE_THRESH
                    ]

                if starving_heads:
                    # 选饥饿时间最长的 head
                    h_starve = max(starving_heads, key=lambda h: starve_steps[aid][h])
                    phase_starve = _head_to_phase(spec, aid, h_starve)
                    env_starve = _phase_to_env(spec, aid, phase_starve)
                    action_sem[aid] = int(h_starve)
                    action_env[aid] = int(env_starve)
                    # 被选中的 head 饥饿计数清零
                    starve_steps[aid][h_starve] = 0

                    # 标记本步是“饥饿强制覆盖”
                    override_by_starve[aid] = True
                else:
                    # 正常执行策略/最短绿/最长绿决定的动作
                    override_by_starve[aid] = False


                # 选定最终执行的 head，清零它的饥饿计数
                starve_steps[aid][action_sem[aid]] = 0

                # === 重要：按“最终执行的语义位”取 log 概率 ===
                old_logp_sem[aid] = torch.log(probs[action_sem[aid]].clamp_min(1e-12))
            
            # 执行动作
            next_state, reward, done_dict, trunc_dict, info = env.step(action_env)

            ref = next(iter(agent_name))
            sys_stop_sum  += float(info[ref].get("system_total_stopped", 0.0))
            sys_wait_sum  += float(info[ref].get("system_total_waiting_time", 0.0))
            sys_speed_sum += float(info[ref].get("system_mean_speed", 0.0))
            T += 1

            # 对本步使用饥饿补偿的路口，在 reward 中施加惩罚
            # 注意：reward 是一个 {aid: float} 字典
            try:
                for aid in agent_name:
                    if override_by_starve.get(aid, False):
                        # 若 env 返回的 reward 中没有该路口，get(aid, 0.0) 兜底
                        r_old = reward.get(aid, 0.0)
                        reward[aid] = r_old - STARVE_REWARD_PENALTY
            except Exception:
                # 即使这里出问题，也不要影响训练主流程
                pass


            if enable_metric and traci_conn is not None:
                # ----- ATT / AWT / AS / TP 统计 -----
                try:
                    sim_time = traci_conn.simulation.getTime()
                    step_len = max(0.0, sim_time - prev_time)
                    prev_time = sim_time

                    # 1) 刚刚出发的车辆
                    for vid in traci_conn.simulation.getDepartedIDList():
                        depart_time[vid] = sim_time
                        veh_wait.setdefault(vid, 0.0)

                    # 2) 在网内车辆：速度与等待时间
                    for vid in traci_conn.vehicle.getIDList():
                        v = traci_conn.vehicle.getSpeed(vid)
                        speed_sum += v
                        speed_cnt += 1
                        if v < WAIT_TH:
                            veh_wait[vid] = veh_wait.get(vid, 0.0) + step_len

                    # 3) 到达车辆：结算旅行时间与等待时间
                    for vid in traci_conn.simulation.getArrivedIDList():
                        dt = sim_time - depart_time.pop(vid, sim_time)
                        total_travel_time += max(0.0, dt)
                        total_wait_time += veh_wait.pop(vid, 0.0)
                        completed += 1
                except Exception:
                    # TraCI 统计失败时不影响训练流程
                    pass

            # 将本步各路口“实际执行”的相位写入训练 CSV
            if traci_conn is not None:
                try:
                    log_time = traci_conn.simulation.getTime()
                except Exception:
                    log_time = None

                try:
                    with open(step_csv_path, "a", newline="", encoding="utf-8-sig") as f_csv:
                        writer_csv = csv.writer(f_csv)
                        for aid in agent_name:
                            # 语义 head（0..7）
                            sem = int(action_sem[aid])
                            # 通过规范得到在 net / SUMO 里的真实相位索引
                            phase_idx = int(_head_to_phase(spec, aid, sem))
                            # 环境动作（在合法绿相位中的 ordinal）
                            env_act = int(action_env[aid])

                            starved = 1 if override_by_starve.get(aid, False) else 0

                            writer_csv.writerow([
                                int(ep),                   # episode
                                int(seed),                 # seed
                                int(step_idx),             # 当前 episode 内的步号
                                log_time if log_time is not None else "",
                                str(aid),                  # intersection
                                sem,                       # sem_head
                                phase_idx,                 # phase_idx
                                env_act,                   # env_action
                                starved,                   # 本步动作是否由饥饿逻辑强制覆盖
                            ])
                except Exception as e:
                    # 日志失败不影响训练流程
                    print(f"[warn] step action log failed at ep={ep}, step={step_idx}: {e}")

            # 记录转换数据
            transition = _append_transition(
                agent_name, epi_training, transition,
                state, done_dict, action_env, action_sem, next_state, reward,
                old_logp_sem=old_logp_sem
            )

            step_idx += 1 
            
            # 更新状态
            epi_training = True
            state = next_state
            episode_return += float(np.mean(list(reward.values())))
            done = all(done_dict.values())
            truncated = all(trunc_dict.values())

        # ---- 日志指标 ----
        queue_avg   = sys_stop_sum  / max(1, T)
        waiting_avg = sys_wait_sum  / max(1, T)
        speed_avg   = sys_speed_sum / max(1, T)

        # 记录
        return_list.append(episode_return)
        waiting_list.append(waiting_avg) 
        queue_list.append(queue_avg) 
        speed_list.append(speed_avg) 
        time_list.append(time.strftime("%m-%d %H:%M:%S", time.localtime()))
        seed_list.append(seed)
                    
        # ---- ATT / AWT / AS / TP 计算并记录 ----
        if completed > 0:
            ATT = total_travel_time / completed
            AWT = total_wait_time / completed
        else:
            ATT = 0.0
            AWT = 0.0
        AS = (speed_sum / speed_cnt) if speed_cnt > 0 else 0.0
        TP = int(completed)

        ATT_list.append(ATT)
        AWT_list.append(AWT)
        AS_list.append(AS)
        TP_list.append(TP)


        # ---- 可选 VAE 表示学习 ----
        if vae is not None:
            whole_state = torch.stack(list(transition["states"].values())).transpose(1, 0)
            end_state = torch.stack(list(transition["next_states"].values())).transpose(1, 0)[-1]
            whole_state = torch.cat([whole_state, end_state.unsqueeze(0)])  # [T, N, C]

            T, _, state_dim = whole_state.shape
            grid_tensor, pos_map, pos_y, pos_x = parse_tls_positions(net_file, agent_name, state_dim, T)

            for t in range(T):
                frame_dict = {agent_name[i]: whole_state[t, i, :].tolist() for i in range(len(agent_name))}
                for aid, vec in frame_dict.items():
                    if aid in pos_map:
                        y, x = pos_map[aid]
                        grid_tensor[t, :, pos_y[y], pos_x[x]] = torch.tensor(vec, dtype=torch.float32)

            vae_loss = _train_vae(vae, vae_opt, grid_tensor, epochs=5)

            # 生成全局嵌入
            dataset = TensorDataset(grid_tensor)
            dl = DataLoader(dataset, 128, shuffle=False)
            global_emb = []
            with torch.no_grad():
                for (x,) in dl:
                    global_emb.append(vae.representation(x.to(device)))
            transition["global_emb"] = torch.cat(global_emb) if len(global_emb) > 0 else []
        else:
            vae_loss = None
        vae_loss_list.append(vae_loss)

        # 将当前 episode 所有路口的损失做平均，作为该集的 loss 记录
        ep_actor_losses = []
        ep_critic_losses = []

        for aid in agent_name:
            actor_loss, critic_loss = agents[aid].update(transition, aid, spec=spec)
            ep_actor_losses.append(float(actor_loss))
            ep_critic_losses.append(float(critic_loss))

        if len(ep_actor_losses) > 0:
            actor_loss_list.append(sum(ep_actor_losses) / len(ep_actor_losses))
            critic_loss_list.append(sum(ep_critic_losses) / len(ep_critic_losses))
        else:
            # 理论上不会走到这里，防御性兜底
            actor_loss_list.append(0.0)
            critic_loss_list.append(0.0)

        # ---- 评估/存储数据 ----
        evaluator.evaluate_and_save(
            writer, return_list, waiting_list, queue_list, speed_list,
            time_list, seed_list, ckpt_path, file_path, ep, agents, seed,
            actor_loss_list, critic_loss_list, vae_loss_list=vae_loss_list, vae=vae,
            att_list=ATT_list, awt_list=AWT_list, as_list=AS_list, tp_list=TP_list,
        )


        # ====== epoch 结束（计算 qbar 并小步调整 α）======
        if reward_helper is not None and ((ep % epoch_episodes) == (epoch_episodes - 1)):
            qbar = reward_helper.end_epoch()          # 跨 TLS 平均 queue_norm
            if reward_scheduler is not None:
                reward_scheduler.update(qbar)         # 只在边界小步调整 alpha
            print(f"[α-adapt] ep={ep} alpha={getattr(reward_helper,'alpha', None):.2f} qbar={qbar:.3f}")

        # ================================================


    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {total_time/60:.2f} min")
    return return_list, total_time/60.0
