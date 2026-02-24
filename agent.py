

import torch,traci
import numpy as np
import torch.nn.functional as F

from utils.action_spec_utils import head_mask as _head_mask 
from utils.action_spec_utils import masked_softmax_logits as _masked_softmax_logits
from utils.action_spec_utils import map_head_to_phase as _map_head_to_phase



# ================= 兜底实现（仅在 utils 未提供时启用） =================
def _to_bool_mask(mask, device):
    if isinstance(mask, torch.Tensor):
        m = mask.to(device=device)
        if m.dtype != torch.bool:
            m = m.bool()
        return m
    return torch.as_tensor(mask, dtype=torch.bool, device=device)
    
def _fallback_masked_softmax_logits(logits: torch.Tensor, mask) -> torch.Tensor:
    """
    计算带掩码的 softmax（mask=False 的位置不参与归一化）。
    用于在掩码格式不规则或类型不确定时的“安全回退版本”。
    兼容单样本 [A] 和批次 [B, A] 两种输入形状。

    参数：
        logits : torch.Tensor
            策略网络输出的未归一化得分。
        mask : list / np.array / torch.Tensor
            掩码：True 或 1 表示该动作可用；False 或 0 表示屏蔽。

    返回：
        torch.Tensor
            同形状概率分布，屏蔽位置的概率为 0。
    """

    # === 单样本情况：[A] ===
    if logits.dim() == 1:
        device = logits.device
        # 确保 mask 转成布尔类型的张量（兼容列表或其他类型）
        m = _to_bool_mask(mask, device)
        # very_neg = 最小可表示数值，用作被屏蔽项的填充值（≈ -inf）
        very_neg = torch.finfo(logits.dtype).min
        # 屏蔽无效动作：mask=False 的地方填 very_neg
        masked = logits.masked_fill(~m, very_neg)
        # 若所有动作都被屏蔽，则返回全 0（防止 softmax NaN）
        if (~m).all():
            masked = torch.zeros_like(masked)
        # 在最后一维执行 softmax（归一化概率分布）
        return F.softmax(masked, dim=-1)

    # === 批次情况：[B, A] ===
    else:
        device = logits.device
        # 扩展 mask 维度为 [1, A] 以便广播到每个 batch
        m = _to_bool_mask(mask, device).unsqueeze(0)
        very_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~m, very_neg)
        return F.softmax(masked, dim=-1)


def _safe_phase_from_spec(spec: dict, tls_id: str, a_sem: int) -> int:
    """
    返回“合法的真实相位索引”（0..n_ph-1）：
      1) 先看 head_to_phase[a_sem]；
      2) 不存在/为 -1/越界 或 legal_mask_phase[x]==0 → 回退到第一个 legal==1 的相位；
      3) 找不到就 0。
    """
    per = spec["per_agent"][tls_id]
    legal = per.get("legal_mask_phase", [])
    n_ph = len(legal)
    h2p = per.get("head_to_phase", [])
    a_env = None
    if isinstance(h2p, list) and 0 <= int(a_sem) < len(h2p):
        try:
            a_env = int(h2p[int(a_sem)])
        except Exception:
            a_env = None
    if a_env is None or a_env < 0 or (n_ph > 0 and a_env >= n_ph) or (n_ph > 0 and legal[a_env] == 0):
        for j in range(n_ph):
            if legal[j] == 1:
                return j
        return 0
    return a_env

def _fallback_map_head_to_phase(spec: dict, tls_id: str, a_sem: int) -> int:
    return _safe_phase_from_spec(spec, tls_id, a_sem)

# 优先使用 utils 中实现；如无则用兜底
def head_mask(spec, tls_id):
    """返回该路口的语义头掩码:pad_mask_head [A]（bool）。"""
    return _head_mask(spec, tls_id)

def masked_softmax_logits(logits, mask):
    if _masked_softmax_logits is not None:
        return _masked_softmax_logits(logits, mask)
    return _fallback_masked_softmax_logits(logits, mask)

def map_head_to_phase(spec, tls_id, a_sem):
    """
    将语义动作索引 a_sem 映射为环境中的真实相位编号。
    优先使用快速映射函数，若结果非法或出错则自动回退至安全版本，
    确保始终返回有效的相位索引。
    """
    if _map_head_to_phase is not None:
        try:
            val = _map_head_to_phase(spec, tls_id, a_sem)
            per = spec["per_agent"][tls_id]
            legal = per.get("legal_mask_phase", [])
            if not isinstance(val, int):
                return _safe_phase_from_spec(spec, tls_id, a_sem)
            if val < 0 or (len(legal) > 0 and (val >= len(legal) or legal[val] == 0)):
                return _safe_phase_from_spec(spec, tls_id, a_sem)
            return int(val)
        except Exception:
            return _safe_phase_from_spec(spec, tls_id, a_sem)
    return _fallback_map_head_to_phase(spec, tls_id, a_sem)


# ======== 把“真实相位索引”压缩为“环境动作索引（0..K-1）” ========
_phase2action_cache = {}  # tls_id -> {phase_idx -> action_idx}

def _phase_to_action_index(spec: dict, tls_id: str, phase_idx: int) -> int:
    """
    将真实相位号（0..n_ph-1）映射为环境的动作索引（0..K-1）。
    K 等于 legal_mask_phase 中 1 的个数；动作序按 legal_mask_phase 中 1 的自然顺序。
    """
    per = spec["per_agent"][tls_id]
    legal = per.get("legal_mask_phase", [])
    key = tls_id
    mapping = _phase2action_cache.get(key)
    if mapping is None:
        mapping = {}
        order = [i for i, v in enumerate(legal) if v == 1]
        for k, j in enumerate(order):
            mapping[j] = k
        _phase2action_cache[key] = mapping
    # 如果目标不在合法表里，回退到第一个合法动作
    if phase_idx in mapping:
        return mapping[phase_idx]
    # 回退：第一个合法动作；再不行就 0
    order = [i for i, v in enumerate(legal) if v == 1]
    return 0 if len(order) == 0 else 0


class MacLight:
    def __init__(
        self,
        policy_net,
        critic_net,
        actor_lr: float = 1e-4,    # 略微提升，策略更快跟上信号
        critic_lr: float = 1e-3,   # 明显降低，减小 noisy reward 对 value 的放大
        gamma: float = 0.9,
        lmbda: float = 0.9,
        epochs: int = 12,
        eps: float = 0.2,
        device: str = 'cpu',
        kl_target: float = 0.02,
        max_grad_norm: float = 0.5,
        value_coef: float = 0.25,  # value loss 在 total_loss 里的权重
    ):
        self.actor = policy_net.to(device)
        self.critic = critic_net.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

        # —— 与 train 的“两参调用”对齐的规范缓存 —— #
        self._spec = None
        self._mask_cache = {}  # tls_id -> torch.bool 掩码（置于正确 device）
        self.head4_target = 0.4 # 目标：希望单个 agent 在一个 batch 内 head4 使用比例不要长期远高于 0.4
        self.head4_lambda = 0.0  # 置 0 以关闭 head4 专用正则


        self.kl_target = kl_target
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef

    def bind_spec(self, spec: dict):
        self._spec = spec
        self._mask_cache.clear()
        # 清理相位→动作索引缓存（避免跨图复用）
        _phase2action_cache.clear()
        try:
            for tls_id in spec["per_agent"].keys():
                m = head_mask(spec, tls_id)
                m = _to_bool_mask(m, self.device)
                self._mask_cache[tls_id] = m
        except Exception:
            pass
        return True

    def actor_forward(self, state_np):
        """
        功能:
        1)将传入的 state 转为 tensor；
        2)将输入传入 actor ,得到 policy 网络的输出结果；
        3)返回 policy 的输出结果.

        返回值：
            一个二维 tensor 张量
        """
        if isinstance(state_np, torch.Tensor):
            x = state_np.to(self.device, dtype=torch.float32).flatten()
        else:
            x = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).flatten()
        logits = self.actor(x)   # 期望 shape=[A]
        return logits


    def act_train(self, state_np, tls_id, spec=None):
        """
        根据当前状态选择训练相位的动作。

        流程：
        1) 前向传播 → 得到策略网络输出 logits；
        2) 获取或生成该路口的语义掩码 mask；
        3) 计算 masked softmax 概率，只在合法语义上归一化；
        4) 按概率采样语义动作 a_sem；
        5) 映射为真实相位号 phase_idx（含安全兜底）；
        6) 再压缩为环境可执行动作索引 a_env；
        7) 返回 (a_env, a_sem, probs)。

        返回：
            a_env : int   → 环境动作编号（用于执行）
            a_sem : int   → 语义动作编号（用于训练对齐）
            probs : Tensor → 动作概率分布
        """
        spec = spec or self._spec
        logits = self.actor_forward(state_np)
        m = self._mask_cache.get(tls_id)

        if m is None:
            m = _to_bool_mask(head_mask(spec, tls_id), self.device)
            self._mask_cache[tls_id] = m
        probs = masked_softmax_logits(logits, m)
        dist = torch.distributions.Categorical(probs=probs)
        a_sem = int(dist.sample().item())                        # 从 action_dim 维度中随机一个动作
        phase_idx = int(map_head_to_phase(spec, tls_id, a_sem))  # 通过 head_to_phase 得到真实相位号（含安全兜底）
        a_env = _phase_to_action_index(spec, tls_id, phase_idx)  # 压缩为环境动作索引
        return a_env, a_sem, probs

    def act_eval(self, state_np, tls_id, spec=None):
        spec = spec or self._spec
        if spec is None:
            raise RuntimeError("MacLight.act_eval: 缺少 spec；请先调用 agent.bind_spec(spec)。")
        logits = self.actor_forward(state_np)
        m = self._mask_cache.get(tls_id)
        if m is None:
            m = _to_bool_mask(head_mask(spec, tls_id), self.device)
            self._mask_cache[tls_id] = m
        probs = masked_softmax_logits(logits, m)
        a_sem = int(torch.argmax(probs).item())
        phase_idx = int(map_head_to_phase(spec, tls_id, a_sem))
        a_env = _phase_to_action_index(spec, tls_id, phase_idx)
        return a_env, a_sem, probs

    # 批量接口；注意这里只返回概率，不做环境索引转换
    def batch_masked_probs(self, states_tensor, tls_ids, spec=None):
        spec = spec or self._spec
        if spec is None:
            raise RuntimeError("MacLight.batch_masked_probs: 缺少 spec；请先 bind_spec(spec)。")
        logits = self.actor(states_tensor.to(self.device))  # [B, A]
        probs_list = []
        for i, tls_id in enumerate(tls_ids):
            m = self._mask_cache.get(tls_id)
            if m is None:
                m = _to_bool_mask(head_mask(spec, tls_id), self.device)
                self._mask_cache[tls_id] = m
            p_i = masked_softmax_logits(logits[i], m).unsqueeze(0)     # [1, A]
            probs_list.append(p_i)
        return torch.cat(probs_list, dim=0)                             # [B, A]

    def batch_log_probs(self, states_tensor, actions_sem_tensor, tls_ids, spec=None):
        spec = spec or self._spec
        if spec is None:
            raise RuntimeError("MacLight.batch_log_probs: 缺少 spec；请先 bind_spec(spec)。")
        probs = self.batch_masked_probs(states_tensor, tls_ids, spec=spec)  # [B, A]
        if actions_sem_tensor.ndim == 1:
            actions_sem_tensor = actions_sem_tensor.unsqueeze(1)            # [B,1]
        gathered = probs.gather(1, actions_sem_tensor.to(self.device))
        gathered = torch.clamp(gathered, min=1e-12)
        return torch.log(gathered)                                          # [B,1]

    def update(self, transition_dict, agent_name, spec):
        """
        Args:
            transition_dict: 包含状态、动作、奖励等转换数据的字典
            agent_name: 当前代理的名称（交通灯ID）
            spec: 动作规范配置字典
        
        Returns:
            tuple: (actor_loss, critic_loss) 最后一次迭代的损失值
        """
        
        # ==================== 1. 数据准备和验证 ====================
        if spec is None:
            raise RuntimeError("PPO更新需要有效的动作规范(spec)")
        
        
        # 获取动作掩码，用于过滤无效动作
        mask = self._mask_cache.get(agent_name)
        if mask is None:
            mask = _to_bool_mask(head_mask(spec, agent_name), self.device)
            self._mask_cache[agent_name] = mask

        # ==================== 2. 转换输入数据为张量 ====================
        def _safe_to_tensor(data, dtype, device):
            """安全地将数据转换为张量，避免重复转换警告"""
            if torch.is_tensor(data):
                return data.detach().to(device=device, dtype=dtype)
            return torch.tensor(data, dtype=dtype, device=device)
        
        # 转换所有必需的输入数据
        states      = _safe_to_tensor(transition_dict['states'][agent_name],      torch.float32, self.device)
        actions_env = _safe_to_tensor(transition_dict['actions'][agent_name],     torch.int64,   self.device).view(-1,1)
        actions_sem = _safe_to_tensor(transition_dict['actions_sem'][agent_name], torch.int64,   self.device).view(-1, 1)
        old_logp    = _safe_to_tensor(transition_dict['old_logp_sem'][agent_name],torch.float32, self.device).view(-1, 1)
        rewards     = _safe_to_tensor(transition_dict['rewards'][agent_name],     torch.float32, self.device).view(-1, 1)
        dones       = _safe_to_tensor(transition_dict['dones'][agent_name],       torch.bool,    self.device).view(-1, 1)
        next_states = _safe_to_tensor(transition_dict['next_states'][agent_name], torch.float32, self.device)


        # 处理全局嵌入
        global_emb_data = transition_dict.get('global_emb', [])
        if len(global_emb_data) > 0:
            global_emb = _safe_to_tensor(global_emb_data, torch.float32, self.device)
            g_now, g_next = global_emb[:-1], global_emb[1:]
        else:
            g_now = g_next = None

        # ==================== 3. 计算目标值（无梯度计算） ====================
        with torch.no_grad():
            # 计算当前状态和下一状态的价值估计
            if g_next is not None:
                v_next = self.critic(next_states, g_next)
                v_curr = self.critic(states, g_now)
            else:
                v_next = self.critic(next_states)
                v_curr = self.critic(states)

            # 计算TD目标：奖励 + 折扣因子 * 下一状态价值 * 非终止标志
            not_done = (~dones).float()
            td_target = rewards + self.gamma * not_done * v_next
            
            # 计算TD误差和优势函数
            td_delta = td_target - v_curr
            advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.detach().cpu())
            advantage = advantage.to(self.device)
            
            # 标准化优势函数以提高训练稳定性
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        A = mask.numel()
        if actions_sem.min().item() < 0 or actions_sem.max().item() >= A:
            raise RuntimeError(f"[{agent_name}] actions_sem 越界：min={actions_sem.min().item()}, "
                            f"max={actions_sem.max().item()}, A={A}")
        # ==================== 4. PPO多轮优化 ====================
        last_actor_loss = last_critic_loss = 0.0
        for epoch in range(self.epochs):
            # 计算当前策略的概率分布
            current_actor_output = self.actor(states)
            if current_actor_output.dim() == 2:
                current_probs = _fallback_masked_softmax_logits(current_actor_output, mask)
            else:
                current_probs = masked_softmax_logits(current_actor_output, mask)

            
            # 计算当前策略的对数概率
            current_log_probs = torch.log(torch.clamp(current_probs.gather(1, actions_sem), min=1e-12))
            ratio = torch.exp(current_log_probs - old_logp)  # 重要性采样比率
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # 取最小值确保保守更新

            # 计算近似 KL：E[log π_old(a|s) - log π_new(a|s)]
            with torch.no_grad():
                approx_kl = torch.mean(old_logp - current_log_probs).item()

            # KL 超出安全阈值：提前停止本次 agent 的多轮更新
            if approx_kl > self.kl_target * 2.0:
                # 可选：打印一次调试信息，方便之后你用日志确认
                # print(f"[KL-stop] {agent_name} epoch={epoch} approx_kl={approx_kl:.4f} > {self.kl_target*2:.4f}")
                break

            
            # Critic损失：价值函数拟合TD目标
            if g_now is not None:
                v_pred = self.critic(states, g_now)
            else:
                v_pred = self.critic(states)
            critic_loss = torch.mean(F.mse_loss(v_pred, td_target))
            
            # 合并损失并执行反向传播
            # （可选）熵正则：来自掩码后的分布（与你之前设计一致）
            dist = torch.distributions.Categorical(probs=current_probs)
            entropy = dist.entropy().mean()

            # 启用适度熵正则，防止策略过快塌缩到单一 head
            entropy_coef = 0.005

            # head4 使用结构的轻量正则（当前 batch 维度）
            head_reg = 0.0
            try:
                A = current_probs.size(1)
                # 只有当 head4 存在且正则强度大于 0 时才启用
                if A > 4 and self.head4_lambda > 0.0:
                    # actions_sem: [T, 1]，统计本 batch 内各 head 使用的频率
                    # 注意：actions_sem 是离散索引，不涉及梯度
                    counts = torch.bincount(actions_sem.view(-1), minlength=A).float()
                    p4 = counts[4] / (counts.sum() + 1e-6)
                    # 超过目标部分才惩罚：max(p4 - target, 0)^2
                    excess = torch.clamp(p4 - self.head4_target, min=0.0)
                    head_reg = self.head4_lambda * (excess ** 2)
            except Exception:
                head_reg = 0.0

            # 总损失：Actor + Critic - 熵奖励 + head4 正则
            total_loss = actor_loss + self.value_coef * critic_loss - entropy_coef * entropy + head_reg        
            
            # 清空梯度并执行反向传播
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # 梯度裁剪（actor + critic 一起做）
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # 记录最后一次迭代的损失
            last_actor_loss = float(actor_loss.item())
            last_critic_loss = float(critic_loss.item())

        return last_actor_loss, last_critic_loss


    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        adv = 0.0
        for delta in td_delta[::-1]:
            adv = gamma * lmbda * adv + delta
            advantage_list.append(adv)
        advantage_list.reverse()
        advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
        advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
        return advantage_list
