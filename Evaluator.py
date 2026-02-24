import torch
import sys
import os
import pandas as pd
import time
import datetime


class Evaluator:
    def __init__(self):

        self.episode = 0
        self.time_flag = time.time()
        print(
            f"\n| `seed`: Random seed of algorithm."
            f"\n| `episode`: Number of current episode."
            f"\n| `time`: Time spent (minute) from the start of training to this moment."
            f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
            f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
            f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
            f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
            f"\n| `VaeLoss`: VAE loss value."
            f"\n| {'seed':>3}  {'episode':>5}  {'time':>5}  |  {'avgR':>7}  {'stdR':>9}     | {'objC':>8}  {'objA':>8}  |  {'VaeLoss':>8}  "
        )

    def evaluate_and_save(
        self,
        writer,
        return_list,
        waiting_list,
        queue_list,
        speed_list,
        time_list,
        seed_list,
        ckpt_path,
        file_path,
        episode,
        agent,
        seed,
        actor_loss_list=False,
        critic_loss_list=False,
        **kwargs
    ):
        """
        训练过程中，每个 episode 结束后调用一次。

        说明：
        - return_list / waiting_list 等都是“从第 0 个 episode 到当前 episode 的全历史”；
        - 每次根据全历史重建 DataFrame 并覆盖保存 CSV；
        - 四个指标：
            ATT_list: 平均旅行时间（秒/车）
            AWT_list: 平均等待时间（秒/车）
            AS_list : 平均速度
            TP_list : 吞吐量（完成车辆数）
        """
        vae_loss_list = kwargs.get("vae_loss_list", None)
        att_list = kwargs.get("att_list", None)
        awt_list = kwargs.get("awt_list", None)
        as_list = kwargs.get("as_list", None)
        tp_list = kwargs.get("tp_list", None)

        path_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ---- 1. 处理路径，兼容 Windows 和简单写法 ----
        if not ckpt_path:
            ckpt_path = "ckpt"   # 兜底

        norm_ckpt_path = os.path.normpath(ckpt_path)
        parts = [p for p in norm_ckpt_path.split(os.sep) if p]

        if len(parts) >= 2:
            # .../<mission>/<alg>
            mission_name = parts[-2]
            alg_name = parts[-1]
        elif len(parts) == 1:
            mission_name = parts[0]
            alg_name = parts[0]
        else:
            mission_name = "default_mission"
            alg_name = "default_alg"

        system_type = sys.platform

        # ---- 2. 创建目录 ----
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

        if not os.path.exists(norm_ckpt_path):
            os.makedirs(norm_ckpt_path, exist_ok=True)

        log_file_path = os.path.join(file_path, f"{seed}_{alg_name}_{system_type}.csv")
        ckpt_file_path = os.path.join(norm_ckpt_path, f"{seed}_{alg_name}_{system_type}.pt")

        # ---- 3. 组织要保存的数据 ----
        return_save = pd.DataFrame()
        return_save["Algorithm"] = [alg_name] * len(return_list)
        return_save["Seed"] = seed_list
        return_save["Return"] = return_list
        return_save["waiting_list"] = waiting_list
        return_save["queue_list"] = queue_list
        return_save["speed_list"] = speed_list

        # v3.2: 直接按 episode 写入 loss 序列，长度应与 return_list 对齐
        return_save["Actor loss"] = actor_loss_list if actor_loss_list else None
        return_save["Critic loss"] = critic_loss_list if critic_loss_list else None

        return_save["VAE loss"] = vae_loss_list if vae_loss_list is not None else None

        # ==== 四个指标列：ATT / AWT / AS / TP ====
        return_save["ATT"] = att_list if att_list is not None else None
        return_save["AWT"] = awt_list if awt_list is not None else None
        return_save["AS"] = as_list if as_list is not None else None
        return_save["TP"] = tp_list if tp_list is not None else None

        return_save["Log time"] = time_list

        # ---- 4. 保存 model & csv ----
        if writer > 0:
            if agent is not None:
                save_dict = {"agent": agent}
                if kwargs.get("vae", None):
                    save_dict["vae"] = kwargs.get("vae")
                torch.save(save_dict, ckpt_file_path)

            # 覆盖写 CSV
            return_save.to_csv(log_file_path, index=False, encoding="utf-8-sig")

        # ---- 5. 打印信息----
        self.episode = episode + 1
        used_time = (time.time() - self.time_flag) / 60
        avg_r = return_save["Return"].mean()
        std_r = return_save["Return"].std()
        actor_loss = return_save["Actor loss"].mean()
        critic_loss = return_save["Critic loss"].mean()
        vae_loss = return_save["VAE loss"].mean()
        print(
            f"| {seed:3d}  {self.episode:5d}    {used_time:5.2f}   "
            f"| {avg_r:9.2f}  {std_r:9.2f}    "
            f"| {critic_loss:8.1f}  {actor_loss:8.1f}  "
            f"| {vae_loss:8.1f}   "
        )
