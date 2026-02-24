
"""

功能：
1. 解析每个信号灯的可控相位；
2. 过滤非语义相位，仅保留 NESW×{直、左} 的候选；
3. 去重得到本地有效语义动作数 a_loc；
4. 生成 ActionSpec，导出 JSON + CSV（+ 可选 Markdown）。

与 train/agent 的接口对齐：
- JSON.meta 写入 net_hash/net_path/run_dir；
- per_agent.<tls_id> 含：
  head_to_phase / legal_mask_phase / pad_mask_head / mixing / stats。
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import datetime
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import traci
    from sumolib import checkBinary
except Exception:
    traci = None

# ================= 语义模板（A~H，NESW × {直,左}） =================
SEM_DIRS  = ["north", "east", "south", "west"]
SEM_TURNS = ["s", "l"]  # 忽略右转

IDX_TO_BIN = {
    1: "west_s",  2: "north_l", 3: "south_s", 4: "east_l",
    5: "east_s",  6: "west_l",  7: "north_s", 8: "south_l",
}
AH_TEMPLATES = [
    ("A", {IDX_TO_BIN[1], IDX_TO_BIN[5]}),  # 西直+东直
    ("B", {IDX_TO_BIN[4], IDX_TO_BIN[5]}),  # 东左+东直
    ("C", {IDX_TO_BIN[1], IDX_TO_BIN[6]}),  # 西直+西左
    ("D", {IDX_TO_BIN[4], IDX_TO_BIN[6]}),  # 东左+西左
    ("E", {IDX_TO_BIN[3], IDX_TO_BIN[7]}),  # 南直+北直
    ("F", {IDX_TO_BIN[8], IDX_TO_BIN[3]}),  # 南左+南直
    ("G", {IDX_TO_BIN[7], IDX_TO_BIN[2]}),  # 北直+北左
    ("H", {IDX_TO_BIN[8], IDX_TO_BIN[2]}),  # 南左+北左
]
AH_HEAD_INDEX = {name: i for i, (name, _) in enumerate(AH_TEMPLATES)}

# ================= 工具 =================
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _rel(path: str, base: str = None) -> str:
    base = base or os.getcwd()
    try:
        return os.path.relpath(os.path.abspath(path), base).replace("\\", "/")
    except Exception:
        return path

def _is_action_phase_sumo_rl(state: str) -> bool:
    """sumo-rl 并行动作空间风格：排除含 'y' 的过渡相位，且至少含一个 g/G"""
    return ('y' not in state) and any(c in 'gG' for c in state)

# ================= 数据结构 =================
class TLSStats:
    def __init__(self, tls_id: str, traci_phases: int, green_raw: int, green_semantic: int, non_semantic_count: int, a_loc: int):
        self.tls_id = tls_id
        self.traci_phases = traci_phases
        self.green_raw = green_raw
        self.green_semantic = green_semantic
        self.non_semantic_count = non_semantic_count
        self.a_loc = a_loc

class AgentSpec:
    def __init__(self, tls_id, head_to_phase, legal_mask_phase, pad_mask_head, mixing, stats: TLSStats):
        self.tls_id = tls_id
        self.head_to_phase = head_to_phase
        self.legal_mask_phase = legal_mask_phase
        self.pad_mask_head = pad_mask_head
        self.mixing = mixing
        self.stats = stats

class ActionSpec:
    def __init__(self, global_action_dim, per_agent: Dict[str, AgentSpec], meta: Dict[str, Any]):
        self.global_action_dim = global_action_dim
        self.per_agent = per_agent
        self.meta = meta

# ================= 相位激活（基于实际连接，仅计 s/l） =================
def _phase_active_bins_improved(fromlanes_on_phase, sem_bins, links, phase_state, tls_id, outgoing_connections):
    """基于具体连接的实际转向（仅 s/l，忽略 r）统计激活桶"""
    active = set()

    for link_idx, lane_links in enumerate(links):
        if link_idx >= len(phase_state):
            break
        if phase_state[link_idx] not in "gG":
            continue

        for (in_lane, out_lane, _via_lane) in lane_links:
            if not in_lane or in_lane not in fromlanes_on_phase:
                continue

            # 解析 laneID -> (edge, lane_index)
            try:
                fe, fli = in_lane.rsplit('_', 1)
                te, tli = out_lane.rsplit('_', 1)
                fli = int(fli); tli = int(tli)
            except Exception:
                continue

            # 在 from_lane 的连接表里找与 out_lane 对应的连接，取其 dir
            actual_turn = None
            for (c_fe, c_fli, c_te, c_tli, c_dir) in outgoing_connections.get(in_lane, []):
                if c_fe == fe and c_fli == fli and c_te == te and c_tli == tli:
                    actual_turn = c_dir or 's'
                    break

            if actual_turn not in ('s', 'l'):
                continue

            turn_key = 's' if actual_turn == 's' else 'l'
            # 将 in_lane 投到匹配方向与转向的语义桶
            for bin_key, bin_lanes in sem_bins.items():
                if bin_key.endswith('_' + turn_key) and in_lane in bin_lanes:
                    active.add(bin_key)
                    break

    return active

# ================= A~H 分配（包含“直行优先”的单向相位规则） =================
def _score_template(active_bins, templ_bins):
    """评分：(score, inter, miss, extra)"""
    inter = len(active_bins & templ_bins)
    miss  = len(templ_bins - active_bins)
    extra = len(active_bins - templ_bins)
    score = 2*inter - 2*miss - extra
    return (score, inter, miss, extra)

def _assign_heads_to_phases(phase_active_bins, sem_bins, topk=1):
    """鲁棒的 A~H 分配（改进：交集为 0 的候选一律跳过）"""
    P = len(phase_active_bins)
    head_to_phase = [-1] * 8
    legal_mask_phase = [1 if act else 0 for act in phase_active_bins]
    used_phases: Set[int] = set()

    # 收集出现过的桶
    present_overall = set()
    for act in phase_active_bins:
        present_overall |= act
    print(f"[DEBUG] Present overall directions: {sorted(present_overall)}")

    # 1) 精确匹配
    for h, (name, templ) in enumerate(AH_TEMPLATES):
        best_p_idx = -1
        for p_idx, A in enumerate(phase_active_bins):
            if p_idx in used_phases or not A:
                continue
            if A == templ:
                best_p_idx = p_idx
                break
        if best_p_idx != -1:
            head_to_phase[h] = best_p_idx
            used_phases.add(best_p_idx)
            print(f"[DEBUG] Exact match: Head {h}({name}) -> Phase {best_p_idx}")

    # 2) 子集匹配（A ⊇ T），extra 越少越好
    for h, (name, templ) in enumerate(AH_TEMPLATES):
        if head_to_phase[h] != -1:
            continue
        best_extra = 1e9
        best_p_idx = -1
        for p_idx, A in enumerate(phase_active_bins):
            if p_idx in used_phases or not A:
                continue
            if templ.issubset(A):
                extra = len(A - templ)
                if extra < best_extra:
                    best_extra = extra
                    best_p_idx = p_idx
        if best_p_idx != -1:
            head_to_phase[h] = best_p_idx
            used_phases.add(best_p_idx)
            print(f"[DEBUG] Subset match: Head {h}({name}) -> Phase {best_p_idx} (extra: {best_extra})")

    # 3) 交集评分（仅当 inter>0 时才允许匹配）
    remaining_phases = [p for p in range(P) if p not in used_phases]
    remaining_heads = [h for h in range(8) if head_to_phase[h] == -1]
    remaining_phases.sort(key=lambda p: len(phase_active_bins[p]), reverse=True)

    for p_idx in remaining_phases:
        A = phase_active_bins[p_idx]
        if len(A) <= 1:
            continue
        best = None
        best_h = -1
        best_inter = 0
        for h in remaining_heads:
            name, T = AH_TEMPLATES[h]
            score, inter, miss, extra = _score_template(A, T)
            # 跳过交集为 0 的候选，防止把含左转的头分给“直行-only”相位
            if inter <= 0:
                continue
            # 同方向优先（如 东直+东左 > 东直+西直）
            same_leg = 1 if {x.split('_')[0] for x in (A & T)} and len({x.split('_')[0] for x in (A & T)}) == 1 else 0
            cand = (score, same_leg, inter, -miss, -extra)
            if best is None or cand > best:
                best = cand
                best_h = h
                best_inter = inter
        if best_h != -1:
            head_to_phase[best_h] = p_idx
            used_phases.add(p_idx)
            remaining_heads.remove(best_h)
            print(f"[DEBUG] Intersection match: Head {best_h}({AH_TEMPLATES[best_h][0]}) -> "
                  f"Phase {p_idx} (score={best[0]}, same_leg={best[1]}, inter={best_inter})")

    # 4) 单向相位：优先“同方向(直+左)”，其后才是“对向直行”；适用性仅做微调
    for p_idx in [p for p in remaining_phases if p not in used_phases]:
        A = phase_active_bins[p_idx]
        if len(A) != 1:
            continue
        single_dir = next(iter(A))              # 如 'south_s'
        base_leg   = single_dir.split('_')[0]   # 'south'
        cand = []  # (h, same_approach, straight_bonus, applicability, support, other_dir)
        for h in remaining_heads:
            name, T = AH_TEMPLATES[h]
            if single_dir not in T:
                continue
            other_dir = (T - {single_dir}).pop()
            same_approach  = 1 if other_dir.split('_')[0] == base_leg else 0
            straight_bonus = 1 if other_dir.endswith('_s') else 0
            applicability  = 1 if other_dir in present_overall else 0
            support        = len(sem_bins.get(other_dir, []))  # 很弱的微调
            cand.append((h, same_approach, straight_bonus, applicability, support, other_dir))
        if not cand:
            continue
        cand.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        best_h, sa, sb, app, sup, other_dir = cand[0]
        head_to_phase[best_h] = p_idx
        used_phases.add(p_idx)
        remaining_heads.remove(best_h)
        why = []
        if sa:  why.append("same-leg")
        if sb:  why.append("prefer-straight")
        if not app:
            why.append("other-missing")  # 显式标注部分匹配
        print(f"[DEBUG] Single-dir match: Head {best_h}({AH_TEMPLATES[best_h][0]}) -> Phase {p_idx} "
              f"(dir={single_dir}, chosen_other={other_dir}, reason={'+'.join(why)})")

    mixing = []
    return head_to_phase, legal_mask_phase, mixing


# ================= 逐 TLS 构建 =================
def build_agent_spec_for_tls(
    tls_id: str,
    lane_dir_map_tls: Dict[str, Dict[str, List[str]]],
    *,
    align_with_env: bool,
    topk: Optional[int],
    net_file: Optional[str] = None  # 在函数签名中加入 net_file
) -> AgentSpec:
    plist = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    assert plist and len(plist) >= 1, f"{tls_id}: 无 phase 程序"
    phases = plist[0].getPhases()
    n_ph = len(phases)
    links = traci.trafficlight.getControlledLinks(tls_id)

    # —— sumo-rl 风格候选动作：不含 'y' 且含 g/G 的相位
    P_idx: List[int] = []
    F_sets: List[Set[str]] = []
    phase_states: List[str] = []

    for p in range(n_ph):
        state = phases[p].state
        if not _is_action_phase_sumo_rl(state):
            continue
        P_idx.append(p)
        phase_states.append(state)
        froms: Set[str] = set()
        for link_idx, lane_links in enumerate(links):
            if link_idx >= len(state) or state[link_idx] not in "gG":
                continue
            for (in_lane, _out_lane, _via) in lane_links:
                if in_lane:
                    froms.add(in_lane)
        F_sets.append(froms)

    green_raw = len(P_idx)

    # NESW×{s,l} 桶（仅直/左；右转天然忽略）
    sem_bins: Dict[str, Set[str]] = {}
    sem_union: Set[str] = set()
    for d in SEM_DIRS:
        for t in SEM_TURNS:
            key = f"{d}_{t}"
            Bs = set(lane_dir_map_tls.get(d, {}).get(t, []))
            sem_bins[key] = Bs
            sem_union |= Bs

    # 解析网络连接，构建 fromLaneID -> [(fe,fli,te,tli,dir), ...]
    outgoing_connections = defaultdict(list)
    try:
        import lane_dir.lane_dir as lane_dir
        net_file = net_file or traci.simulation.getNetFile()  # 使用传入的 net_file，若无则回退
        G = lane_dir._parse_net(net_file)
        for fe, fli, te, tli, d in G["connections"]:
            outgoing_connections[f"{fe}_{int(fli)}"].append((fe, int(fli), te, int(tli), d or 's'))
    except Exception as e:
        print(f"[WARN] 无法获取精确连接信息，使用基础方法: {e}")
        outgoing_connections = {}

    # 相位 → 语义桶
    phase_active_bins: List[Set[str]] = []
    for k, (lp, S) in enumerate(zip(P_idx, F_sets)):
        if outgoing_connections and k < len(phase_states):
            act = _phase_active_bins_improved(S, sem_bins, links, phase_states[k], tls_id, outgoing_connections)
        else:
            # 回退：from-lane ∈ 桶 → 计入（共享车道会误计，尽量不走这里）
            S_sem = S & sem_union
            act = set()
            for bin_key, bin_set in sem_bins.items():
                if bin_set and (S_sem & bin_set):
                    act.add(bin_key)
        phase_active_bins.append(act)

    # 统计
    green_semantic = sum(1 for act in phase_active_bins if act)
    non_semantic_count = green_raw - green_semantic

    # 审计
    print(f"[AUDIT][{tls_id}] P_idx(local)={P_idx}")
    for k, lp in enumerate(P_idx):
        act = sorted(list(phase_active_bins[k]))
        samples = {}
        for b in act:
            arr = list(sem_bins[b])
            lanes_hit = sorted(list((set(arr)) & F_sets[k]))
            samples[b] = lanes_hit[:2]
        print(f"[AUDIT][{tls_id}] phase#{lp} (local={k}) active_bins={act} samples={samples}")

    # A~H 分配
    head_to_phase_local, legal_mask_phase_local, _ = _assign_heads_to_phases(
        phase_active_bins, sem_bins, topk=topk or 1
    )

    # 映回全局相位索引
    head_to_phase: List[int] = []
    for hp in head_to_phase_local:
        head_to_phase.append(P_idx[hp] if (isinstance(hp, int) and 0 <= hp < len(P_idx)) else -1)

    # legal_mask_phase（长度 = n_ph；仅对 P_idx 置 1）
    legal_mask_phase = [0] * n_ph
    for k, lp in enumerate(P_idx):
        if legal_mask_phase_local[k]:
            legal_mask_phase[lp] = 1

    # 8 维掩码（仅映射成功的位为 1）
    pad_mask_head = [1 if p >= 0 else 0 for p in head_to_phase]
    a_loc = sum(pad_mask_head)

    # 一致性断言
    assert len(head_to_phase) == 8 and len(pad_mask_head) == 8
    assert len(legal_mask_phase) == n_ph
    for h, p in enumerate(head_to_phase):
        if p == -1:
            continue
        assert p in P_idx, f"[{tls_id}] head {h} -> phase {p} 不在 action_space 候选 {P_idx}"
        local = P_idx.index(p)
        act = phase_active_bins[local]
        templ = AH_TEMPLATES[h][1]
        if len(act & templ) == 0:
            raise AssertionError(f"[{tls_id}] head {h} 模板与相位 {p} 零交集: templ={templ}, act={act}")

    stats = TLSStats(
        tls_id=tls_id,
        traci_phases=n_ph,
        green_raw=green_raw,
        green_semantic=green_semantic,
        non_semantic_count=non_semantic_count,
        a_loc=a_loc
    )

    return AgentSpec(
        tls_id=tls_id,
        head_to_phase=head_to_phase,
        legal_mask_phase=legal_mask_phase,
        pad_mask_head=pad_mask_head,
        mixing=[],
        stats=stats
    )

# ================= 汇总导出 =================
def export_action_spec(spec: ActionSpec, out_root: str, *, net_file: Optional[str] = None, route_file: Optional[str] = None) -> str:
    tdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, tdir)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "global_action_dim": 8,
        "meta": {
            "run_dir": _rel(out_dir)
        },
        "per_agent": {}
    }

    if net_file:
        payload["meta"]["net_path"] = _rel(net_file)
        payload["meta"]["net_hash"] = _md5(net_file)
    if route_file:
        payload["meta"]["route_path"] = _rel(route_file)
        try:
            payload["meta"]["route_hash"] = _md5(route_file)
        except FileNotFoundError:
            pass

    for tls, ag in spec.per_agent.items():
        payload["per_agent"][tls] = {
            "head_to_phase": ag.head_to_phase,
            "legal_mask_phase": ag.legal_mask_phase,
            "pad_mask_head": ag.pad_mask_head,
            "mixing": ag.mixing,
            "stats": {
                "tls_id": ag.stats.tls_id,
                "traci_phases": ag.stats.traci_phases,
                "green_raw": ag.stats.green_raw,
                "green_semantic": ag.stats.green_semantic,
                "non_semantic_count": ag.stats.non_semantic_count,
                "a_loc": ag.stats.a_loc
            }
        }

    payload["meta"].setdefault("spec_format_version", "v1.1")

    json_path = os.path.join(out_dir, "tls_action_spec.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "tls_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tls_id","traci_phases","green_raw","green_semantic","a_loc","non_semantic_count"])
        for tls, ag in spec.per_agent.items():
            s = ag.stats
            w.writerow([s.tls_id, s.traci_phases, s.green_raw, s.green_semantic, s.a_loc, s.non_semantic_count])

    print(f"[EXPORT] JSON: {json_path}")
    print(f"[EXPORT] CSV : {csv_path}")
    return out_dir

def load_action_spec(json_path: str) -> ActionSpec:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    per_agent = {}
    for tls_id, d in data["per_agent"].items():
        st = d["stats"]
        per_agent[tls_id] = AgentSpec(
            tls_id=tls_id,
            head_to_phase=d["head_to_phase"],
            legal_mask_phase=d["legal_mask_phase"],
            pad_mask_head=d["pad_mask_head"],
            mixing=d.get("mixing", []),
            stats=TLSStats(
                tls_id=st["tls_id"],
                traci_phases=st["traci_phases"],
                green_raw=st["green_raw"],
                green_semantic=st["green_semantic"],
                non_semantic_count=st["non_semantic_count"],
                a_loc=st["a_loc"],
            )
        )
    return ActionSpec(global_action_dim=int(data["global_action_dim"]),
                      per_agent=per_agent, meta=data.get("meta", {}))

# =============== CLI 主入口（保持原有运行方式，需要 --net） ===============
if __name__ == "__main__":

    # 路网文件路径
    net_file   = r'env\sumo_fenglin_base_road\fenglin_y2z_t.net.xml'
    parser = argparse.ArgumentParser(description="Build TLS action spec (no hardcoded defaults).")
    # parser.add_argument("--net", required=True, help="Path to SUMO net.xml")
    parser.add_argument("--route", default=None, help="Optional route file")
    parser.add_argument("--out", default="lane_dir/outputs_specs", help="Directory to write outputs")
    parser.add_argument("--align-with-env", dest="align_env", action="store_true")
    parser.add_argument("--no-align-with-env", dest="align_env", action="store_false")
    parser.set_defaults(align_env=True)
    parser.add_argument("--topk", type=int, default=1, help="Top-k sparsity for mixing display")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui instead of sumo")
    args = parser.parse_args()

    if traci is None:
        print("[FATAL] 需要在 SUMO/TraCI 可用的环境下运行")
        sys.exit(2)

    net_file = os.path.abspath(net_file)
    route_file = os.path.abspath(args.route) if args.route else None
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    # 仅依赖你项目已有的 lane_dir.py（兼容不同函数名）
    try:
        import lane_dir.lane_dir as lane_dir
    except Exception as e:
        print(f"[FATAL] 无法导入 lane_dir.py：{e}")
        sys.exit(2)

    if hasattr(lane_dir, "build_lane_dir_map"):
        build_lane_dir_map = lane_dir.build_lane_dir_map
    elif hasattr(lane_dir, "build_lane_dir_map_with_meta"):
        def build_lane_dir_map(netf: str):
            return lane_dir.build_lane_dir_map_with_meta(netf)[0]
    elif hasattr(lane_dir, "build_all_tls_lane_dirs"):
        build_lane_dir_map = lane_dir.build_all_tls_lane_dirs
    else:
        print("[FATAL] lane_dir.py 未找到构建函数")
        sys.exit(3)

    SUMO_BIN = checkBinary("sumo-gui" if args.gui else "sumo")
    cmd = [SUMO_BIN, "-n", net_file, "--no-step-log", "true", "--no-warnings", "true"]
    if route_file and os.path.isfile(route_file):
        cmd += ["-r", route_file]
    traci.start(cmd)

    try:
        lane_dir_map = build_lane_dir_map(net_file)
        per_agent: Dict[str, AgentSpec] = {}
        for tls in traci.trafficlight.getIDList():
            m = lane_dir_map.get(tls)
            if not m:
                print(f"[WARN] lane_dir map 缺少 {tls}，跳过")
                continue
            ag = build_agent_spec_for_tls(
                tls_id=tls,
                lane_dir_map_tls=m,
                align_with_env=args.align_env,
                topk=args.topk,
                net_file=net_file
            )
            per_agent[tls] = ag
        spec = ActionSpec(global_action_dim=8, per_agent=per_agent, meta={})
        out_dir = export_action_spec(spec, out_root, net_file=net_file, route_file=route_file)

        # 自检
        loaded = load_action_spec(os.path.join(out_dir, "tls_action_spec.json"))
        assert loaded.global_action_dim == spec.global_action_dim
        assert set(loaded.per_agent.keys()) == set(spec.per_agent.keys())
        print("[OK] 训练导出与评估加载一致")
        print(f"[DONE] Spec 导出目录：{out_dir}")
    finally:
        try:
            traci.close()
        except Exception:
            pass
