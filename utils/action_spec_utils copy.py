

import json
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn.functional as F

Spec = Dict[str, Any]

"""模块用途

统一加载 reorder 导出的动作规范（actual_phases/perm/semantic_order），
并提供：构造掩码、带掩码的 softmax、语义→真实相位的重排。
训练与评估两侧都应仅通过本模块使用这些功能。

提供：
1) action_dim(spec): 取语义动作头总维度
2) head_mask(spec, tls): 取该路口的语义头掩码 [A]
3) masked_softmax_logits(logits, mask): 先遮 -inf 再 softmax
4) map_head_to_phase(spec, tls, a_sem): 语义 head → 真实相位
5) legal_phase_mask(spec, tls): 可选，返回真实相位合法性掩码
"""

def load_spec(path: str) -> Spec:
    """读取规范 JSON（新版结构：global_action_dim + per_agent）。"""
    with open(path, "r", encoding="utf-8") as f:
        spec: Spec = json.load(f)
    assert "per_agent" in spec and "global_action_dim" in spec, "缺少 per_agent 或 global_action_dim"
    return spec

def action_dim(spec: Spec) -> int:
    """返回全局语义动作头维度。"""
    return int(spec["global_action_dim"])

def head_mask(spec: Spec, tls_id: str) -> torch.Tensor:
    """返回该路口的语义头掩码 [A]（bool）。"""
    arr = spec["per_agent"][tls_id]["pad_mask_head"]
    t = torch.tensor(arr, dtype=torch.bool)
    assert t.dim() == 1, "pad_mask_head 需为一维"
    return t

def masked_softmax_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """被 mask=False 的位置置 -inf，再 softmax（支持 [A] 或 [B,A]）。"""
    x = logits.clone()
    x[~mask] = float("-inf")
    return F.softmax(x, dim=-1)

def map_head_to_phase(spec: Spec, tls_id: str, a_sem: int) -> int:
    """将语义 head 索引映射为环境真实相位索引；兼容全长度与压缩式两种表。"""
    item = spec["per_agent"][tls_id]
    table = item["head_to_phase"]
    A = int(spec["global_action_dim"])
    mask = item["pad_mask_head"]

    if len(table) == A:
        # 全长度表：直接索引
        assert 0 <= a_sem < A, "a_sem 超出 global_action_dim"
        return int(table[a_sem])

    # 压缩式：len(table) == 可用 head 数。需要按照掩码把 a_sem 映射到“第几个 True”。
    if len(table) == int(sum(1 for v in mask if v)):
        assert 0 <= a_sem < A, "a_sem 超出 global_action_dim"
        if not mask[a_sem]:
            raise ValueError(f"a_sem={a_sem} 指向被掩掉的 head，无法映射 head_to_phase")
        rank = sum(1 for i in range(a_sem) if mask[i])  # 该 head 在 True 序列中的排名
        return int(table[rank])

    raise AssertionError(
        f"{tls_id}.head_to_phase 长度既不等于 global_action_dim={A}，也不等于可用 head 数"
    )

def legal_phase_mask(spec: Spec, tls_id: str) -> Optional[torch.Tensor]:
    """可选：返回真实相位合法性掩码 [N_env]；没有则 None。"""
    item = spec["per_agent"][tls_id]
    if "legal_mask_phase" not in item:
        return None
    return torch.tensor(item["legal_mask_phase"], dtype=torch.bool)
