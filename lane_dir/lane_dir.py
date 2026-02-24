
"""

职责：从 SUMO .net.xml 构建“每个 TLS 的 NESW × {左(l), 直(s)} 统一语义”，并输出可审计元数据。
方向与转向判定（通用、场景无关）：
  • 方向(Approach/leg)：先用“驶入切线角 θ_leg”判定 行进方向(motion∈{E/N/W/S})，再取互补作为“来自哪一侧”的 leg：
        leg = NESW_COMP[motion]
    若落在象限边界灰带(±G°)，再用多证据裁决（位置向量、直行对面投票、角距平滑）。
    之后做“网络一致性”：沿道路把对向两端(N↔S / E↔W)校正（仅改低置信侧），
    本实现以“几何角”为一等公民，并加入“横/纵轴护栏”，避免把明显横向的腿改到纵向（反之亦然）。
  • 转向(Movement)：优先用 connection.dir（'s'/'','l','r'），缺失则几何兜底：
      Δ = wrapTo180(out - in)；|Δ|≤STRAIGHT_MAX → s；Δ≥LEFT_MIN → l；|Δ|≥UTURN → u(统计)；其余→r(统计)

统一约定：
  - NESW 基于世界坐标：E=0°，N=90°，W=180°，S=270°。
  - φ* 仅用于角度审计，不参与 NESW 判定。
  - right_only：某方向仅存在右转（l/s 皆空）时会标注。
  - legacy[tls][dir]['l'/'s']：属于该方向的左/直示例 lanes 列表（仅展示前 K 条）。
  - meta[tls]['legs']：
      [{'edge_id','theta_leg','label','heading','margin_deg','ambiguous',
        'position_vec':(ux,uy),'straight_bias': 'horizontal'|'vertical'|None,
        'fixed_by_edge_consistency': bool, 'fixed_by_fill': bool, 'fixed_by_pairing': bool}, ...]
"""

import sys, os, math, xml.etree.ElementTree as ET
from collections import defaultdict

# ======= 可调常量（默认即可跨网稳定） =======
STRAIGHT_MAX_DEG = 20.0   # 直行阈值：相对偏转 ≤ 20°
LEFT_MIN_DEG     = 35.0   # 左转阈值：逆时针偏转 ≥ 35°
UTURN_DEG        = 150.0  # 折返阈值：|Δ| ≥ 150°
ALIGN_STEP_DEG   = 1.0    # φ* 搜索步长：在 [0, 90)° 穷举（用于角度审计）
AMBIGUOUS_BAND   = 12.0   # 灰带宽度 ±G°（象限边界两侧的模糊区）
CONSIS_EPS_DEG   = 20.0   # 网络一致性：对向匹配的容差（≈180°±ε）
ROUNDABOUT_R_TH  = 0.7    # 环岛样态：右转占比阈值
SAMPLE_SHOW_K    = 4      # 自检输出：每方向示例 lane_id 最大数量
FILL_EMPTY_DIR   = True   # 开启四向补位
FILL_BAND_DEG    = 20.0   # 只有 margin ≤ 阈值的“靠边腿”允许被挪用（默认 20° 更稳健）
DUP_SPLIT_BAND_DEG = 25.0  # 仅对“靠边腿”做重复拆分：margin≤25° 或本身 ambiguous


NESW_ORDER = ("north", "east", "south", "west")
NESW_COMP  = {"north":"south", "south":"north", "east":"west", "west":"east"}
NESW_BASE  = {"east":0.0, "north":90.0, "west":180.0, "south":270.0}

# ======= 轴护栏（提升跨网稳健性） =======
AXIS_GUARD_DEG   = 10.0   # 横/纵轴护栏：若改标会让腿远离其更接近的轴超过该阈值，则禁止跨轴改标

def _axis_of_label(lbl: str) -> str:
    return 'horizontal' if lbl in ('east','west') else 'vertical'

def _axis_dist_deg(theta: float, axis: str) -> float:
    """到横轴(0/180)或纵轴(90/270)的最小角距（度）"""
    if axis == 'horizontal':
        return min(abs(_wrap180(theta-0.0)), abs(_wrap180(theta-180.0)))
    else:
        return min(abs(_wrap180(theta-90.0)), abs(_wrap180(theta-270.0)))

def _same_axis(a: str, b: str) -> bool:
    return _axis_of_label(a) == _axis_of_label(b)

# ======= 小工具 =======
def _wrap180(a):
    while a <= -180.0: a += 360.0
    while a >   180.0: a -= 360.0
    return a

def _deg_norm(a):
    while a < 0.0: a += 360.0
    while a >= 360.0: a -= 360.0
    return a

def _bearing_xy(p_to, p_from):
    dx, dy = p_to[0]-p_from[0], p_to[1]-p_from[1]
    if dx==0.0 and dy==0.0: return 0.0
    deg = math.degrees(math.atan2(dy, dx))  # E=0°, CCW+
    return _deg_norm(deg)

def _estimate_phi_star(thetas):
    # 穷举 φ∈[0,90) 使 ∑ min_{k∈{0,90,180,270}} |wrap180((θ+φ)-k)| 最小（审计指标）
    best_phi = 0.0; best_skew = float('inf')
    for phi in [i*ALIGN_STEP_DEG for i in range(int(90/ALIGN_STEP_DEG))]:
        tot = 0.0
        for th in thetas:
            thp = _deg_norm(th + phi)
            d = min(abs(_wrap180(thp - b)) for b in (0.0,90.0,180.0,270.0))
            tot += d
        if tot < best_skew:
            best_skew, best_phi = tot, phi
    return best_phi, best_skew

def _neighbor_dirs(d):
    if d == "north": return ("east","west")
    if d == "east":  return ("north","south")
    if d == "south": return ("east","west")
    if d == "west":  return ("north","south")
    return ()

def _dir_center_deg(d):
    return NESW_BASE[d]

def _turn_flag(in_deg, out_deg, flag):
    # 优先使用 SUMO dir 标记
    if flag in ('s', ''):  # SUMO 有时直行记为空串
        return 's'
    if flag == 'l':
        return 'l'
    if flag == 'r':
        return 'r'
    # 几何兜底
    delta = _wrap180(out_deg - in_deg)
    if abs(delta) <= STRAIGHT_MAX_DEG: return 's'
    if delta >= LEFT_MIN_DEG:          return 'l'
    if abs(delta) >= UTURN_DEG:        return 'u'
    return 'r'

def _short_list(lst, k=SAMPLE_SHOW_K):
    if not lst: return "[]"
    if len(lst) <= k:
        return "[" + ",".join(lst) + "]"
    return "[" + ",".join(lst[:k]) + ",...]"

def _legend_print():
    print("=== 输出字段说明 ===")
    print("  [TLS] <id> | approaches=<进入边数> | phi*=<对齐角°> | effective_actions=<出现 l/s 的方向数>")
    print("  行格式:")
    print("    leg=<方向>  heading→{N|E|S|W}  lanes=<车道数> | left=<YES/NO> straight=<YES/NO>")
    print("      l=[示例lane_id...]  s=[示例lane_id...]  margin=<距象限边界的角差°>  [NOTE]")
    print("")
    print("  定义要点：")
    print("  • leg：按驶入切线角的“行进方向”取互补得到（来自哪一侧）。")
    print("  • heading：车辆行进方向（E/W/N/S），与 leg 互补。")
    print("  • margin：该方向与最近象限边界的角差；小=靠边，可能触发裁决/校正。")
    print("  • NOTE: right_only / fixed_by_fill / fixed_by_pairing / fixed_by_edge_consistency（兼容旧日志）。")
    print("")

# ======= 解析 .net.xml =======
def _parse_net(net_path: str):
    """
    Parse a SUMO .net.xml into lightweight structures for downstream NESW/turn analysis.

    Robust TLS (traffic-light-controlled junction) detection:
      1) Any <junction> whose 'type' starts with 'traffic_light' is treated as TLS.
      2) Any <junction> whose id appears in <tlLogic id="..."> is ALSO treated as TLS.

    Returns
    -------
    dict with keys:
        tls_ids, jxy, node_xy, edge_from_to, edge_lanes, lane_shape, connections
    """
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"未找到路网文件: {net_path}")
    root = ET.parse(net_path).getroot()

    # Junction coords (ALL junctions)
    node_xy = {}
    for j in root.findall('junction'):
        node_xy[j.get('id')] = (float(j.get('x')), float(j.get('y')))

    # TLS detection
    tls_nodes = []
    tls_set_by_logic = set()
    for t in root.findall('tlLogic'):
        tid = t.get('id')
        if tid:
            tls_set_by_logic.add(tid)
    for j in root.findall('junction'):
        j_id = j.get('id')
        j_type = j.get('type') or ''
        if j_type.startswith('traffic_light') or (j_id in tls_set_by_logic):
            tls_nodes.append(j)
    tls_ids = [j.get('id') for j in tls_nodes]
    jxy = {j.get('id'): (float(j.get('x')), float(j.get('y'))) for j in tls_nodes}

    # Edges and lanes
    edge_from_to, edge_lanes, lane_shape = {}, {}, {}
    for e in root.findall('edge'):
        if e.get('function') == 'internal':
            continue
        eid = e.get('id')
        from_node = e.get('from'); to_node = e.get('to')
        edge_from_to[eid] = (from_node, to_node)
        lids = []
        for li, ln in enumerate(e.findall('lane')):
            lid = f"{eid}_{li}"
            lids.append(lid)
            shape_attr = ln.get('shape')
            if shape_attr:
                pts = []
                for token in shape_attr.strip().split():
                    x_str, y_str = token.split(',')
                    pts.append((float(x_str), float(y_str)))
                lane_shape[lid] = pts
        edge_lanes[eid] = lids

    # Lane-level connections (NORMAL→NORMAL lanes only)
    normal_edge_ids = set(edge_from_to.keys())
    connections = []
    for c in root.findall('connection'):
        fe = c.get('from'); te = c.get('to')
        if (fe in normal_edge_ids) and (te in normal_edge_ids):
            fli = int(c.get('fromLane')) if c.get('fromLane') is not None else 0
            tli = int(c.get('toLane')) if c.get('toLane') is not None else 0
            dirflag = c.get('dir') or ''
            connections.append((fe, fli, te, tli, dirflag))

    return {
        "tls_ids": tls_ids,
        "jxy": jxy,
        "node_xy": node_xy,
        "edge_from_to": edge_from_to,
        "edge_lanes": edge_lanes,
        "lane_shape": lane_shape,
        "connections": connections
    }

# ======= 主流程 =======
def _label_from_theta(theta_deg: float):
    """
    先判定行进方向 motion，再取互补作为 leg（来自哪一侧）。
    返回:
      label: leg方向 'north'|'east'|'south'|'west'  （来自哪一侧）
      heading: 行进方向字母 'N'|'E'|'S'|'W'
      margin: 距离最近象限边界(45/135/225/315)的角差（度）
      ambiguous: 是否处于灰带(±AMBIGUOUS_BAND)
    """
    centers = {"east": 0.0, "north": 90.0, "west": 180.0, "south": 270.0}
    # motion：按 θ 最近主轴
    motion = min(centers, key=lambda k: abs(_wrap180(theta_deg - centers[k])))
    # leg = 来向 = motion 的互补
    label = NESW_COMP[motion]
    heading_char = {"east":"E","north":"N","west":"W","south":"S"}[motion]
    theta_n = _deg_norm(theta_deg)
    boundaries = (45.0, 135.0, 225.0, 315.0)
    margin = min(abs(_wrap180(theta_n - b)) for b in boundaries)
    ambiguous = margin <= AMBIGUOUS_BAND
    return label, heading_char, margin, ambiguous

def build_lane_dir_map_with_meta(net_path: str):
    G = _parse_net(net_path)
    tls_ids, jxy = G["tls_ids"], G["jxy"]
    node_xy = G["node_xy"]
    edge_from_to = G["edge_from_to"]
    edge_lanes   = G["edge_lanes"]
    lane_shape   = G["lane_shape"]

    # from-lane → [(to-lane, dir_flag)]
    outgoing = defaultdict(list)
    for fe, fli, te, tli, d in G["connections"]:
        try:
            from_lid = f"{fe}_{fli}"
            to_lid   = f"{te}_{tli}"
            outgoing[from_lid].append((to_lid, d))
        except Exception:
            pass

    legacy = {}  # tls -> dir -> {'l':[lanes],'s':[lanes]}
    meta   = {}

    # ---- 每个 TLS 单独处理 ----
    for tls in tls_ids:
        cx, cy = jxy[tls]

        # 进入该 TLS 的边/车道
        in_edges = []
        for eid, (frm, to) in edge_from_to.items():
            if to == tls:  # 指向该路口
                in_edges.append(eid)

        in_lanes = []
        lane_in_bearing = {}  # lane_id -> 朝向该路口的切线角
        lane_tip_pt = {}      # lane_id -> 靠近路口的末端点
        missing_shape = 0
        for eid in in_edges:
            for lid in edge_lanes.get(eid, []):
                pts = lane_shape.get(lid, [])
                if len(pts) < 2:
                    missing_shape += 1
                    continue
                a, b = pts[-2], pts[-1]  # 靠近路口的末段
                lane_in_bearing[lid] = _bearing_xy(b, a)  # 方向“朝路口”
                lane_tip_pt[lid]  = b
                in_lanes.append(lid)

        # 合并成“来向臂”：按进入边 edge 分组
        legs = []  # {edge_id, lanes:[lid], theta_leg(朝内), rep_point}
        for eid in in_edges:
            lids = [lid for lid in edge_lanes.get(eid, []) if lid in lane_in_bearing]
            if not lids:
                continue
            ths = [lane_in_bearing[lid] for lid in lids]
            pts = [lane_tip_pt[lid]      for lid in lids]
            theta_leg = _deg_norm(sum(ths)/len(ths))
            # 代表点：取末端点的均值
            rx = sum(p[0] for p in pts)/len(pts); ry = sum(p[1] for p in pts)/len(pts)
            legs.append({"edge_id": eid, "lanes": lids, "theta_leg": theta_leg, "rep": (rx, ry)})

        # φ*（仅用于角度审计/平滑，不用于决定 NESW）
        phi_star, total_dev = _estimate_phi_star([g["theta_leg"] for g in legs])

        # 直行“对面”投票所需：统计该 leg 的直行连接主要指向水平/垂直
        def _straight_bias_for_leg(lids, in_th):
            hori, vert = 0, 0
            for lid in lids:
                for to_lid, dflag in outgoing.get(lid, []):
                    pts2 = lane_shape.get(to_lid, [])
                    if len(pts2) < 2: 
                        continue
                    out_th = _bearing_xy(pts2[1], pts2[0])
                    flag = _turn_flag(in_th, out_th, dflag)
                    if flag == 's':
                        if _axis_dist_deg(out_th, 'horizontal') <= _axis_dist_deg(out_th, 'vertical'):
                            hori += 1
                        else:
                            vert += 1
            if hori == vert == 0:
                return None
            return 'horizontal' if hori >= vert else 'vertical'

        # 初判 + 灰带裁决
        legs_out = []
        for g in legs:
            # 位置向量：路口中心->上游（以边 from 节点近似）
            frm_node, _ = edge_from_to.get(g["edge_id"], (None, None))
            ux, uy = 0.0, 0.0
            if frm_node and (frm_node in node_xy):
                ux = node_xy[frm_node][0] - cx
                uy = node_xy[frm_node][1] - cy

            label, heading, margin, ambiguous = _label_from_theta(g["theta_leg"])
            sbias = _straight_bias_for_leg(g["lanes"], g["theta_leg"])

            legs_out.append({
                "edge_id": g["edge_id"],
                "lanes": g["lanes"],
                "theta_leg": round(g["theta_leg"], 3),
                "label": label,                  # ← 来向（来自哪一侧）
                "heading": heading,              # ← 行进方向（E/W/N/S）
                "margin_deg": round(margin, 3),  # 接近象限边界的“置信边距”
                "ambiguous": bool(ambiguous),
                "position_vec": (round(ux,3), round(uy,3)),
                "straight_bias": sbias,
                "fixed_by_edge_consistency": False,  # 兼容旧字段
                "fixed_by_fill": False,              # 四向补位触发
                "fixed_by_pairing": False,           # 对向配对校正触发
            })

        # 灰带多证据裁决（只用于 ambiguous 的 leg；位置优先，其次直行偏置，最后角距平滑）
        for L in legs_out:
            if not L["ambiguous"]:
                continue
            scores = {k:0.0 for k in NESW_ORDER}
            # 位置向量优先：绝对值大的分到对应轴类（横/纵）
            ux, uy = L["position_vec"]
            if abs(ux) >= abs(uy):
                scores["east"]  += 1.0
                scores["west"]  += 1.0
            else:
                scores["north"] += 1.0
                scores["south"] += 1.0
            # 直行对面投票：偏向更可能的轴类
            if L["straight_bias"] == 'horizontal':
                scores["east"]  += 0.5; scores["west"]  += 0.5
            elif L["straight_bias"] == 'vertical':
                scores["north"] += 0.5; scores["south"] += 0.5
            # 角距平滑：越接近各自中心轴得分越高（小加权，避免压过主证据）
            for k in NESW_ORDER:
                c = NESW_BASE[k]
                scores[k] += -0.1*abs(_wrap180(L["theta_leg"] - c))
            best_motion = max(scores.items(), key=lambda kv: kv[1])[0]
            new_label = NESW_COMP[best_motion]             # ← 行进方向取互补，作为“来向”leg
            L["label"] = new_label
            L["heading"] = {"north":"S","east":"W","south":"N","west":"E"}[new_label]  # 同步 heading（可选）

            # heading 仍是“行进方向”，保持不变

        # 同向重复腿优先分流到同轴空桶方向
        _split_duplicate_legs(legs_out)

        # ---- 四向补位（仅同轴补位，避免横↔纵跨象限挪用）----
        if FILL_EMPTY_DIR:
            _fill_empty_directions(legs_out)
            
        # movement 投票（l/s/r/u）——lane 级投票后归入其 leg 的 label
        legacy[tls] = {k:{'l':[], 's':[]} for k in NESW_ORDER}
        cnt_r = cnt_u = 0
        broken_conn = 0
        lanes_left = {k:set() for k in NESW_ORDER}
        lanes_straight = {k:set() for k in NESW_ORDER}
        lanes_right = {k:set() for k in NESW_ORDER}
        turns_by_dir = {k:{} for k in NESW_ORDER}
        leg_by_lane = {}
        for L in legs_out:
            for lid in L["lanes"]:
                leg_by_lane[lid] = L["label"]
        for L in legs_out:
            for lid in L["lanes"]:
                ins = outgoing.get(lid, [])
                if not ins:
                    broken_conn += 1; continue
                votes = {'l':0,'s':0,'r':0,'u':0}
                pts = lane_shape.get(lid, [])
                if len(pts) < 2: continue
                in_th = _bearing_xy(pts[-1], pts[-2])  # 反向，保证“朝中心”
                for to_lid, dflag in ins:
                    pts2 = lane_shape.get(to_lid, [])
                    if len(pts2) < 2:
                        continue
                    out_th = _bearing_xy(pts2[1], pts2[0])
                    flag = _turn_flag(in_th, out_th, dflag)
                    votes[flag] += 1
                # 联合口径：同一 from-lane 具备多种能力时，同时计入各集合
                if votes['l'] > 0:
                    lanes_left[ leg_by_lane[lid] ].add(lid)
                if votes['s'] > 0:
                    lanes_straight[ leg_by_lane[lid] ].add(lid)
                if votes['r'] > 0:
                    lanes_right[ leg_by_lane[lid] ].add(lid)
                if votes['u'] > 0:
                    cnt_u += 1
                turns_by_dir[ leg_by_lane[lid] ][lid] = votes

        # 汇总 per-dir
        per_dir = {}
        for k in NESW_ORDER:
            per_dir[k] = {
                "lanes_total": len(lanes_left[k] | lanes_straight[k] | lanes_right[k]),
                "left_cnt": len(lanes_left[k]),
                "straight_cnt": len(lanes_straight[k]),
                "right_cnt": len(lanes_right[k]),
            }
            legacy[tls][k]['l'] = sorted(list(lanes_left[k]))
            legacy[tls][k]['s'] = sorted(list(lanes_straight[k]))

        # 审计辅助：环岛样态与 right_only
        roundabout_like = False
        total_right = sum(per_dir[k]["right_cnt"] for k in NESW_ORDER)
        total_turns = sum(per_dir[k]["lanes_total"] for k in NESW_ORDER)
        if total_turns>0 and (total_right/total_turns) >= ROUNDABOUT_R_TH:
            roundabout_like = True
        right_only = any(per_dir[k]["left_cnt"]==0 and per_dir[k]["straight_cnt"]==0 and per_dir[k]["right_cnt"]>0 for k in NESW_ORDER)

        meta[tls] = {
            "approach_count": len(legs_out),
            "phi_star": round(phi_star, 1),
            "direction_skew": round(total_dev, 3),
            "per_dir": per_dir,
            "turn_votes": turns_by_dir,
            "missing_shape_count": missing_shape,
            "broken_conn_count": broken_conn,
            "right_only": right_only,
            "roundabout_like": roundabout_like,
            "effective_actions": sum(1 for k in NESW_ORDER if (len(legacy[tls][k]['l'])+len(legacy[tls][k]['s']))>0),
            "legs": legs_out,
        }

    # ---- 第二次：网络一致性校正（以几何角为准 + 轴护栏）----
    tls_set = set(tls_ids)
    for tls in tls_ids:
        legs_out = meta[tls]['legs']
        for L in legs_out:
            eid = L["edge_id"]
            from_node, to_node = G["edge_from_to"].get(eid, (None, None))
            if from_node in tls_set and to_node == tls:
                other = from_node  # 相邻路口
                opp = _deg_norm(L["theta_leg"] + 180.0)
                cand = None; best = 1e9
                for R in meta[other]['legs']:
                    d = abs(_wrap180(opp - R["theta_leg"]))
                    if d < best:
                        cand, best = R, d
                if cand and best <= CONSIS_EPS_DEG:
                    # 基于“几何角”推导两端的几何 leg（来向）
                    geo_here, _, _, _  = _label_from_theta(L["theta_leg"])
                    geo_there, _, _, _ = _label_from_theta(cand["theta_leg"])

                    # 如果几何上不互补，则把 margin 小（低置信）的一侧改成“对端几何 leg”的互补
                    if NESW_COMP.get(geo_here) != geo_there:
                        want_here  = NESW_COMP[geo_there]
                        want_there = NESW_COMP[geo_here]

                        def _guard_allows_change(theta, cur_lbl, new_lbl):
                            cur_axis  = _axis_of_label(cur_lbl)
                            new_axis  = _axis_of_label(new_lbl)
                            if cur_axis == new_axis:
                                return True  # 同轴(E↔W/N↔S)自由改
                            # 跨轴：只有当改过去的轴更“近”且留有护栏余量才允许
                            return _axis_dist_deg(theta, new_axis) + AXIS_GUARD_DEG < _axis_dist_deg(theta, cur_axis)

                        if L["margin_deg"] <= cand["margin_deg"]:
                            if _guard_allows_change(L["theta_leg"], L["label"], want_here):
                                L["label"] = want_here
                            else:
                                L["label"] = geo_here
                            L["fixed_by_pairing"] = True
                            L["fixed_by_edge_consistency"] = True
                        else:
                            if _guard_allows_change(cand["theta_leg"], cand["label"], want_there):
                                cand["label"] = want_there
                            else:
                                cand["label"] = geo_there
                            cand["fixed_by_pairing"] = True
                            cand["fixed_by_edge_consistency"] = True

    return legacy, meta


def build_all_tls_lane_dirs(net_path: str):
    legacy, _ = build_lane_dir_map_with_meta(net_path)
    return legacy

def _fill_empty_directions(legs_out):
    """
    若某个方向为空，则从其相邻方向中，挑 margin 小、且“theta_leg 最接近该方向中心轴”的腿挪过来。
    只挪 ambiguous=True 或 margin ≤ FILL_BAND_DEG 的腿；并强制“同轴补位”（横→横，纵→纵）。
    """
    by_dir = {d: [] for d in ("north","east","south","west")}
    for L in legs_out:
        by_dir[L["label"]].append(L)

    empties = [d for d,lst in by_dir.items() if len(lst)==0]
    if not empties:
        return

    for d in empties:
        neigh = _neighbor_dirs(d)
        cand = []
        for n in neigh:
            for L in by_dir[n]:
                if L["ambiguous"] or L["margin_deg"] <= FILL_BAND_DEG:
                    # 几何轴护栏：仅允许同“轴类”的腿被补位（横→横，纵→纵）
                    geo_lbl, _, _, _ = _label_from_theta(L["theta_leg"])
                    if _same_axis(geo_lbl, d):
                        dev = abs(_wrap180(L["theta_leg"] - _dir_center_deg(d)))
                        cand.append((dev, L))
        if not cand:  # 实在没有靠边腿可挪，放弃补位
            continue
        cand.sort(key=lambda x: x[0])
        # 挑“最接近该方向中心轴”的腿：改其 label
        _, bestL = cand[0]
        bestL["label"] = d
        bestL["fixed_by_fill"] = True
        bestL["fixed_by_edge_consistency"] = True  # 兼容旧日志：补位提示

def _split_duplicate_legs(legs_out):
    """
    若某个方向(label)下有多条腿，且其“同轴邻居方向”存在空桶，
    则把其中更靠近邻居主轴的一条腿挪到该邻居，以形成更直观的四向划分。
    仅对靠边腿（ambiguous 或 margin≤DUP_SPLIT_BAND_DEG）生效；仅在同轴内挪动。
    """
    def _neighbors_same_axis(lbl):
        # 同轴邻居：南/北的邻居是东/西；东/西的邻居是南/北
        return ("east","west") if lbl in ("north","south") else ("north","south")

    # 迭代处理，直至没有可分流的情况
    while True:
        # 按方向收集
        by_dir = {d: [] for d in ("north","east","south","west")}
        for L in legs_out:
            by_dir[L["label"]].append(L)

        empties = {d for d,lst in by_dir.items() if len(lst)==0}
        changed = False

        # 遍历“重复腿”的方向
        for lbl, lst in by_dir.items():
            if len(lst) <= 1:
                continue
            targets = empties.intersection(_neighbors_same_axis(lbl))
            if not targets:
                continue

            best_tuple = None  # (improve_score, L, target_dir)
            for L in lst:
                # 只动靠边腿
                if not (L.get("ambiguous", False) or (L.get("margin_deg", 999.0) <= DUP_SPLIT_BAND_DEG)):
                    continue
                theta = L["theta_leg"]
                # 当前与本轴中心的偏差
                dev_cur = abs(_wrap180(theta - NESW_BASE[lbl]))
                # 在目标候选里挑最近者
                best_t, best_dev = None, 1e9
                for t in targets:
                    dev_t = abs(_wrap180(theta - NESW_BASE[t]))
                    if dev_t < best_dev:
                        best_t, best_dev = t, dev_t
                # 轴护栏：只做同轴挪动且要“明显更近”
                if best_dev + AXIS_GUARD_DEG >= dev_cur:
                    continue
                improve_score = best_dev - dev_cur  # 越小越好（负值更优）
                if (best_tuple is None) or (improve_score < best_tuple[0]):
                    best_tuple = (improve_score, L, best_t)

            if best_tuple is not None:
                _, L, target = best_tuple
                # 执行分流
                L["label"] = target
                L["fixed_by_fill"] = True               # 复用既有标志
                L["fixed_by_edge_consistency"] = True   # 兼容旧日志
                changed = True
                break  # 做一次移动后重建 by_dir/empties，再循环

        if not changed:
            break

def _print_validation(legacy, meta, top_n=10):
    tls_ids = list(legacy.keys())
    print("[CHECK] 构建 NESW×{l,s} 统一语义 ...")
    print(f"[OK] 路口(TLS)总数: {len(tls_ids)}\n")
    _legend_print()

    show_ids = tls_ids[:top_n] if top_n is not None else tls_ids
    for tls in show_ids:
        info = meta[tls]
        print(f"[TLS] {tls} | approaches={info['approach_count']} | phi*={info['phi_star']}° | effective_actions={info['effective_actions']}")
        for d in NESW_ORDER:
            L = legacy[tls][d]['l']; S = legacy[tls][d]['s']
            legs_d = [x for x in info['legs'] if x['label']==d]
            margin = min([x['margin_deg'] for x in legs_d]) if legs_d else '-'
            note_right_only = (len(L)==0 and len(S)==0 and len(legacy[tls][d].get('r', []))>0)
            note = ""
            if note_right_only:
                note += " NOTE: right_only"
            if legs_d:
                if any(x.get('fixed_by_fill') for x in legs_d):
                    note += " NOTE: fixed_by_fill"
                if any(x.get('fixed_by_pairing') for x in legs_d):
                    note += " NOTE: fixed_by_pairing"
                elif any(x.get('fixed_by_edge_consistency') for x in legs_d):
                    note += " NOTE: fixed_by_edge_consistency"
            heading = {"north":"S","east":"W","south":"N","west":"E"}[d]
            lanes_total = info['per_dir'][d]['lanes_total'] if d in info['per_dir'] else len(set(L)|set(S))
            print(f"  leg={d.capitalize():5}  heading→{heading}  lanes={lanes_total:>2} | left={'YES' if len(L)>0 else 'NO ':3} straight={'YES' if len(S)>0 else 'NO ':3}  | l={_short_list(L)} s={_short_list(S)}  margin={margin}{note}")
        warns = []
        if info['right_only']: warns.append("right_only")
        if info['roundabout_like']: warns.append("roundabout_like")
        if info['missing_shape_count']>0: warns.append(f"missing_shape={info['missing_shape_count']}")
        if info['broken_conn_count']>0: warns.append(f"broken_conn={info['broken_conn_count']}")
        if info['effective_actions']<2: warns.append("few_effective_actions")
        if warns: print("  WARN:", ", ".join(warns))
        print("-"*90)
    print("\n[DONE] 自检完成。如需打印全部 TLS：python lane_dir.py <net.xml> ALL")

# ======= 入口 =======
if __name__ == "__main__":
    net = sys.argv[1] if len(sys.argv)>=2 else r"env\map_ff\ff.net.xml"
    show_all = (len(sys.argv)>=3 and str(sys.argv[2]).upper()=="ALL")
    try:
        legacy_map, meta_map = build_lane_dir_map_with_meta(net)
        _print_validation(legacy_map, meta_map, top_n=None if show_all else 10)
    except Exception as e:
        print("[ERR] 自检失败：", repr(e))
