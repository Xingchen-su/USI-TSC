# spec_guard.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import glob
import hashlib
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Callable, Iterable, Dict, Any, Tuple, Optional
from utils import action_spec_utils as asu  # 仅做数据层工具：取A、掩码softmax等


"""模块用途
保证训练/评估在启动时，拿到一份“与当前 net.xml 匹配”的 tls_action_spec.json：
1) 若已存在且匹配 → 直接用；
2) 若不存在/不匹配 → (训练端可选) 自动调用 reorder 生成；
3) 对齐环境：逐路口核对动作空间、掩码与映射表，提前发现错配。
"""


# ---------- 基础工具 ----------

def md5_file(path: str) -> str:
    """计算文件 MD5（用于与 spec.meta.net_hash 比对）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _rel(path: str, project_root: str = ".") -> str:
    '''相对路径显示（统一正斜杠）'''
    import os
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(project_root))
    return rel.replace("\\", "/")


def _spec_matches_net(spec: Dict[str, Any], net_xml: str) -> bool:
    """若 spec 有 meta.net_hash，则与 net.xml MD5 比对；否则宽松放行并提示。"""
    meta = spec.get("meta", {})
    net_hash = meta.get("net_hash")
    if not net_hash:
        print("[spec_guard][INFO] 当前 spec 缺少 meta.net_hash，未做哈希校验（宽松通过）。")
        return True
    ok = (net_hash == md5_file(net_xml))
    if not ok:
        print(f"[spec_guard][WARN] net_hash 不匹配: spec={net_hash} vs net={md5_file(net_xml)}")
    return ok


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_spec_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.basename(path) == "tls_action_spec.json"


def _find_latest_spec(root_dir: str, net_hash: Optional[str] = None) -> Optional[str]:
    """
    在 root_dir 内递归查找最新的 tls_action_spec.json；
    若给了 net_hash，则优先找 meta.net_hash == net_hash 的那份。
    """
    candidates = []
    for p in glob.glob(os.path.join(root_dir, "**", "tls_action_spec.json"), recursive=True):
        try:
            spec = _load_json(p)
            mh = spec.get("meta", {}).get("net_hash")
            mtime = os.path.getmtime(p)
            candidates.append((p, mtime, mh))
        except Exception:
            continue
    if not candidates:
        return None
    if net_hash:
        matched = [c for c in candidates if c[2] == net_hash]
        if matched:
            return sorted(matched, key=lambda x: x[1], reverse=True)[0][0]
    # 回退：最新的那份
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

def _print_choice(spec: Dict[str, Any], spec_path: str, net_xml: str, project_root: str = ".") -> None:
    """集中打印：选择的 spec，及其与当前 net.xml 的指纹对比（全用相对路径）"""
    meta = spec.get("meta", {})
    sp_rel = _rel(spec_path, project_root)
    rd_rel = meta.get("run_dir") or _rel(os.path.dirname(spec_path), project_root)
    built_at = meta.get("built_at")

    spec_net_path = meta.get("net_path") or _rel(net_xml, project_root)   # 规范内已是相对路径
    spec_net_hash = meta.get("net_hash")
    curr_net_path = _rel(net_xml, project_root)
    curr_net_hash = md5_file(net_xml)
    matched = (spec_net_hash == curr_net_hash) if spec_net_hash else None

    print("\n[spec_guard] ===== ACTION SPEC SELECTION =====")
    print(f"[CHOSEN ] spec: {sp_rel}")
    print(f"[SOURCE ] dir:  {rd_rel}   built_at: {built_at}")
    print(f"[SPEC  ] net:   {spec_net_path}   md5: {spec_net_hash}")
    print(f"[CURR  ] net:   {curr_net_path}   md5: {curr_net_hash}")
    if matched is True:
        print("[RESULT] MATCH: ✅  (spec.meta.net_hash == current net.xml md5)")
    elif matched is False:
        print("[RESULT] MATCH: ❌  (hash mismatch)")
    else:
        print("[RESULT] MATCH: N/A (spec has no net_hash)")
    print("[spec_guard] ==================================\n")


# ---------- 调用 reorder 并定位输出 ----------

def _run_reorder_and_pick(
    reorder_py: str,
    net_xml: str,
    out_root: str,
    python_exec: str = sys.executable,
    timeout_sec: int = 600,
    extra_args: Tuple[str, ...] = (),
) -> str:
    """
    调用 reorder 生成新输出；由于其通常按时间戳建子目录，这里在运行后
    通过“扫描 out_root 下最新的 tls_action_spec.json”来确定真实路径。
    """
    os.makedirs(out_root, exist_ok=True)
    before = set(glob.glob(os.path.join(out_root, "**", "tls_action_spec.json"), recursive=True))

    cmd = [python_exec, reorder_py, "--net", net_xml, "--out", out_root, *extra_args]
    print("[spec_guard][INFO] 运行 reorder:", " ".join(cmd))
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout_sec
    )
    print("[spec_guard][INFO] reorder 输出：\n" + (ret.stdout or ""))
    if ret.returncode != 0:
        raise RuntimeError("[spec_guard] reorder 执行失败，详见上方输出。")

    # 查找新增的 spec；若没有新增，取最新的一份
    after = set(glob.glob(os.path.join(out_root, "**", "tls_action_spec.json"), recursive=True))
    new = list(after - before)
    if new:
        # 选择 mtime 最新
        newest = sorted(new, key=lambda p: os.path.getmtime(p), reverse=True)[0]
        return newest

    # 没找到“新增”文件，退而求其次：取 out_root 下最新的
    latest = _find_latest_spec(out_root)
    if latest:
        return latest
    raise FileNotFoundError("[spec_guard] 未能定位到新生成的 tls_action_spec.json")


# ---------- 对外主入口 ----------

def ensure_action_spec(
    net_xml: str,
    desired: str,
    reorder_py: str = "reorder_mask_builder.py",
    out_root: str = "outputs_specs",
    auto_generate: bool = True,
    python_exec: str = sys.executable,
    reorder_extra_args: Tuple[str, ...] = (),
) -> Tuple[Dict[str, Any], str]:
    """获取与 net.xml 匹配的 action_spec；desired 可为具体 .json 或目录。"""
    assert os.path.isfile(net_xml), f"net.xml 不存在：{net_xml}"

    def _load(p: str) -> Dict[str, Any]:
        return _load_json(p)

    # A) desired=文件
    if desired.lower().endswith(".json"):
        if os.path.exists(desired):
            spec = _load(desired)
            if _spec_matches_net(spec, net_xml):
                _print_choice(spec, desired, net_xml)
                return spec, desired
            if not auto_generate:
                raise AssertionError("[spec_guard] 指定的 action_spec 与 net.xml 不匹配；请先运行 reorder。")
        # 不存在或不匹配且允许自动生成
        spec_path = _run_reorder_and_pick(reorder_py, net_xml, out_root, python_exec, extra_args=reorder_extra_args)
        spec = _load(spec_path)
        if not _spec_matches_net(spec, net_xml):
            raise AssertionError("[spec_guard] 自动生成的 action_spec 仍与 net.xml 不匹配，请检查 reorder。")
        _print_choice(spec, spec_path, net_xml)
        return spec, spec_path

    # B) desired=目录：优先找哈希匹配；否则取最新
    if not os.path.isdir(desired):
        raise NotADirectoryError(f"[spec_guard] 既不是 .json 文件也不是目录：{desired}")

    expect = md5_file(net_xml)
    cand = _find_latest_spec(desired, net_hash=expect) or _find_latest_spec(desired)
    if cand:
        spec = _load(cand)
        _print_choice(spec, cand, net_xml)
        return spec, cand

    if not auto_generate:
        raise FileNotFoundError("[spec_guard] 目录内未找到匹配 net 的 action_spec；请先运行 reorder。")

    spec_path = _run_reorder_and_pick(reorder_py, net_xml, desired, python_exec, extra_args=reorder_extra_args)
    spec = _load(spec_path)
    if not _spec_matches_net(spec, net_xml):
        raise AssertionError("[spec_guard] 自动生成的 action_spec 仍与 net.xml 不匹配，请检查 reorder。")
    _print_choice(spec, spec_path, net_xml)
    return spec, spec_path


def verify_env_compat(
    spec: Dict[str, Any],
    tls_ids: Iterable[str],
    get_phase_count: Callable[[str], int],
) -> None:
    """
    与环境一致性核对（稳健&兼容版）：
    - 兼容两种 head_to_phase 形态：
      * 全长度：len(head_to_phase) == global_action_dim
      * 压缩式：len(head_to_phase) == pad_mask_head.sum()  （只列出可用 head 的目标）
    - 其他校验同前：pad_mask_head 长度、a_loc/本地动作域一致、映射目标落在“真实相位域”，
      legal_mask_phase（若有）需合法；对明显 1-based 的表做校验时临时 -1 处理。
    """
    per = spec.get("per_agent", {})
    A = asu.action_dim(spec)

    for tls in tls_ids:
        assert tls in per, f"[spec_guard] spec 缺少路口：{tls}"
        item = per[tls]

        # 1) 语义头掩码长度
        head_mask = item["pad_mask_head"]
        assert len(head_mask) == A, f"[spec_guard] {tls}.pad_mask_head 长度 {len(head_mask)} ≠ global_action_dim {A}"
        head_mask_sum = int(sum(1 for v in head_mask if v))

        # 2) 本地动作域一致性
        n_env_local = int(get_phase_count(tls))
        stats = item.get("stats", {})
        if "a_loc" in stats:
            n_spc = int(stats["a_loc"])
            assert n_spc == head_mask_sum, f"[spec_guard] {tls}: a_loc={n_spc} ≠ 掩码可用数={head_mask_sum}"
            assert n_env_local == n_spc, f"[spec_guard] {tls}: env本地相位={n_env_local} ≠ a_loc={n_spc}"

        # 3) 真实相位域长度（优先用 legal_mask_phase，其次用 stats.traci_phases，最后兜底本地域）
        if "legal_mask_phase" in item:
            real_domain_len = int(len(item["legal_mask_phase"]))
        else:
            stats = item.get("stats", {})
            if "traci_phases" in stats:
                real_domain_len = int(stats["traci_phases"])   # ← 用你已有字段
            else:
                real_domain_len = n_env_local  # 兜底，不推荐，但保证不崩

        # 4) head_to_phase 兼容校验（全长度/压缩式）
        h2p_raw = item["head_to_phase"]
        L = len(h2p_raw)
        if L == A:
            # 全长度
            seq = list(range(A))
            h2p_full = list(h2p_raw)
        elif L == head_mask_sum:
            # 压缩式：只为可用 head 给映射 —— 展开成全长度，掩码=False 的位置填占位 -1
            h2p_full = [-1] * A
            idx = 0
            for h, use in enumerate(head_mask):
                if use:
                    h2p_full[h] = int(h2p_raw[idx])
                    idx += 1
        else:
            raise AssertionError(
                f"[spec_guard] {tls}.head_to_phase 长度 {L} 既不等于 global_action_dim {A}，"
                f"也不等于可用 head 数 {head_mask_sum}"
            )

        # 4.1 1-based 容错（只对被使用的 head 生效）
        used_targets = [int(v) for v, use in zip(h2p_full, head_mask) if use]
        if used_targets and min(used_targets) >= 1 and max(used_targets) == real_domain_len:
            print(f"[spec_guard][INFO] 检测到 {tls} 的 head_to_phase 可能为 1-based，校验时按 0-based 处理。")
            h2p_checked = [ (v-1 if (v >= 1 and use) else v) for v, use in zip(h2p_full, head_mask) ]
        else:
            h2p_checked = h2p_full

        # 4.2 目标域越界与合法性
        for h, use in enumerate(head_mask):
            if not use:
                continue
            tgt = int(h2p_checked[h])
            assert 0 <= tgt < real_domain_len, \
                f"[spec_guard] {tls}: head {h} 映射目标 {tgt} 超出 0..{real_domain_len-1}"
        if "legal_mask_phase" in item:
            lmp = item["legal_mask_phase"]
            for h, use in enumerate(head_mask):
                if not use:
                    continue
                tgt = int(h2p_checked[h])
                assert lmp[tgt], f"[spec_guard] {tls}: head {h} → phase {tgt} 非法 (legal_mask_phase=0)"


    # [ADD-LOG] 精确 SUMMARY（全量统计，不再使用“≈”）
    try:
        A = asu.action_dim(spec)
        per = spec.get("per_agent", {})

        usable_counts = []
        h2p_full_cnt, h2p_comp_cnt, h2p_other_cnt = 0, 0, 0
        real_domain_lens = []

        for tls in tls_ids:
            item = per[tls]
            mask = item["pad_mask_head"]
            usable = int(sum(1 for v in mask if v))
            usable_counts.append(usable)

            L = len(item["head_to_phase"])
            if L == A:
                h2p_full_cnt += 1
            elif L == usable:
                h2p_comp_cnt += 1
            else:
                h2p_other_cnt += 1

            if "legal_mask_phase" in item:
                real_domain_lens.append(len(item["legal_mask_phase"]))
            elif "stats" in item and "traci_phases" in item["stats"]:
                real_domain_lens.append(int(item["stats"]["traci_phases"]))
            else:
                # 兜底：用本地动作域
                real_domain_lens.append(int(get_phase_count(tls)))

        # usable_heads 统计：若一致打印“=N”，否则打印“min..max / mean”
        u_min, u_max = min(usable_counts), max(usable_counts)
        if u_min == u_max:
            usable_str = f"={u_min}"
        else:
            u_mean = sum(usable_counts) / max(1, len(usable_counts))
            usable_str = f"{u_min}..{u_max} (mean={u_mean:.2f})"

        # 真实域长度统计：同理
        r_min, r_max = min(real_domain_lens), max(real_domain_lens)
        if r_min == r_max:
            real_str = f"={r_min}"
        else:
            r_mean = sum(real_domain_lens) / max(1, len(real_domain_lens))
            real_str = f"{r_min}..{r_max} (mean={r_mean:.2f})"

        total_tls = len(list(tls_ids))
        fmt_parts = []
        if h2p_full_cnt:
            fmt_parts.append(f"full {h2p_full_cnt}/{total_tls}")
        if h2p_comp_cnt:
            fmt_parts.append(f"compressed {h2p_comp_cnt}/{total_tls}")
        if h2p_other_cnt:
            fmt_parts.append(f"other {h2p_other_cnt}/{total_tls}")
        h2p_fmt_str = ", ".join(fmt_parts) if fmt_parts else "n/a"

        print(f"[SUMMARY] global_action_dim={A}  usable_heads{usable_str}  head_to_phase=({h2p_fmt_str})  real_phase_domain{real_str}")
    except Exception as e:
        print(f"[spec_guard][WARN] 打印摘要失败：{e}")

    print("[spec_guard][OK] 规范与环境一致性校验通过。")

