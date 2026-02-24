
use_gui =False

import os
if use_gui == False:    os.environ['LIBSUMO_AS_TRACI'] = '1'
import sys
import argparse
import random, datetime
import numpy as np
import torch
import sumo_rl

# ====== 项目路径 ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _rel_path(p: str) -> str:
    """仅用于打印：把绝对路径转为相对项目根目录的路径"""
    try:
        return os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
    except Exception:
        return p

# # ====== 启动调试 ======
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

    
# ====== 导入主要模块 ======
from Reward import CustomRewardHelper, RewardScheduler
from agent import MacLight
from train import train_ours_agent
from utils.tools import MARLWrap
from Evaluator import Evaluator
from random_block import BlockStreet
from lane_dir.lane_dir import build_all_tls_lane_dirs
from ReEnv import CustomObservationFunction
from net import VAE, PolicyNet, ValueNet
from utils.spec_guard import ensure_action_spec


# ====== 默认配置 ======

sumofile = {
    "colo":{
        "net": r"env\cologne8\cologne8.net.xml" ,
        "rou": r"env\cologne8\cologne8.rou.xml" ,
    },
    "fenglin":{
        "net": r"env\sumo_fenglin_base_road\fenglin_y2z_t.net.xml" ,
        "rou": r"env\sumo_fenglin_base_road\fenglin_y2z_t.rou.xml" ,
    },
    "ingo":{
        "net": r"env\ingolstadt21\ingolstadt21.net.xml",
        "rou": r"env\ingolstadt21\ingolstadt21.rou.xml",
    },
    "normal":{
        "net": r"env\map_ff\ff.net.xml",
        "rou": r"env\map_ff\ff_normal.rou.xml",
    },
    "hard":{
        "net": r"env\map_ff\ff.net.xml",
        "rou": r"env\map_ff\ff_hard.rou.xml",
    },
    "block":{
        "net": r"env\map_ff\ff.net.xml",
        "rou": r"env\map_ff\ff_normal.rou.xml",
    },
}

DEFAULT_TASK  = "regular"
DEFAULT_MODEL = "Clean"
DEFAULT_SPEC  = "lane_dir/outputs_specs"
path_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ====== 主入口 ======
def main():
    parser = argparse.ArgumentParser(description="MacLight Runner (multi-seed)")
    parser.add_argument('--scenario',   type=str, default='colo', help='Choose a scenario name from: colo / fenglin / ingo / normal / hard / block')
    parser.add_argument('--task',       type=str, choices=['regular', 'block'], default=DEFAULT_TASK)
    parser.add_argument('--model',      type=str, default=DEFAULT_MODEL)
    parser.add_argument('--spec_path',  type=str, default=DEFAULT_SPEC)
    parser.add_argument('--episodes',   type=int, default=150)
    parser.add_argument('--seconds',    type=int, default=3600)
    parser.add_argument('--block_num',  type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--delta_time', type=int, default=5)
    parser.add_argument('--representation',     action='store_true', default=True,  help='Whether or not to use VAE')
    parser.add_argument('-s', '--seed', nargs='+', default=[42], type=int,help='Set random seed list, e.g. -s 42 46 50')
    args = parser.parse_args()

    net_file = os.path.abspath(sumofile[args.scenario]['net']).replace("\\", "/")
    route_file = os.path.abspath(sumofile[args.scenario]['rou']).replace("\\", "/")
    if args.scenario == 'block':    args.task = 'block'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n========== CONFIG ==========")
    print(f"net_file   : {_rel_path(net_file)}")
    print(f"route_file : {_rel_path(route_file) if route_file else '[none]'}")
    print(f"episodes   : {args.episodes} | seconds/ep={args.seconds}")
    print(f"task       : {args.task} | block_num={args.block_num if args.task=='block' else 0}")
    print(f"model_name : {args.model}")
    print(f"representation(VAE): {args.representation} | latent_dim={args.latent_dim}")
    print(f"spec_path  : {args.spec_path}")
    print(f"device     : {device}")
    print(f"seeds      : {args.seed}")
    print("============================\n")

    if 'colo' in net_file:
        begin_time = 25200
    elif 'ingo' in net_file:
        begin_time = 57600
    else:
        begin_time = 0
    print(f"begin:{begin_time}")

    # ===== 多种子循环 =====
    for seed in args.seed:
        print(f"\n[Start] task={args.task} | model={args.model} | seed={seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        helper = CustomRewardHelper(
            alpha=0.05,           # 先偏向热身
            qcap_per_lane=12.0,
            w_pressure=0.4,
            penal_switch=0.05,
            band_ratio=(0.10, 0.15),
            alpha_band=1.0,
            ewma_rho=0.0         # 先关去抖；稳定后可开 0.2
        )
        scheduler = RewardScheduler(helper, qL=0.10, qU=0.15, eps=0.01, step=0.1, patience=50, backoff=100)

        # 1) 语义映射
        lane_dir_map = build_all_tls_lane_dirs(net_file)
        spec, _ = ensure_action_spec(net_xml=net_file, 
                                        desired=args.spec_path,            # 命令行/默认传入的路径
                                        auto_generate=True                 # 若不存在或不匹配，自动调用 reorder 生成
                                    )

        # 2) 构建环境
        env = sumo_rl.parallel_env(
            net_file=      net_file,
            route_file=    route_file,
            begin_time=    begin_time,
            num_seconds=   args.seconds,
            observation_class=lambda ts: CustomObservationFunction(ts, lane_dir_map=lane_dir_map, tls_action_spec=spec),
            reward_fn      = helper, 
            use_gui=       use_gui,
            sumo_warnings= False,
            # delta_time=    10,
            additional_sumo_cmd="--no-step-log"
            
        )

        agent_names = env.possible_agents
        state_dim = max(env.observation_space(a).shape[0] for a in agent_names)
        hidden_dim = 2 * state_dim
        action_dim = int(8)
        latent_dim = args.latent_dim

        if args.task == "block":
            env = BlockStreet(env, args.block_num, args.seconds)

        # 超参数
        n_tls = len(agent_names)
        print(f"[INFO] number of TLS: {n_tls}")

        # 根据路口数分档
        if n_tls >= 20:
            # 像 ingo 这种较大网络
            actor_lr = 5e-5
            critic_lr = 8e-4
            epochs    = 6        # 减少每个 episode 上的 PPO 迭代轮数
        elif n_tls >= 10:
            # 中等规模（normal 这类）
            actor_lr = 1e-4
            critic_lr = 1e-3
            epochs    = 10
        else:
            # 小图：ff / colo 等，保持原来较快的学习速度
            actor_lr = 1e-4
            critic_lr = 1e-3
            epochs    = 10

        alg_args = dict(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lmbda=0.95,
            gamma=0.99,
            device=device,
            epochs=epochs,
            eps=0.2,
            agent_name=agent_names
        )


        # 输出目录（每个 seed 独立）
        ckp_path = f'ckpt/{args.task}/{args.model}/{path_time}'
        file_path = f"data/{args.task}/{args.model}/{path_time}"
        os.makedirs(ckp_path, exist_ok=True)
        os.makedirs(file_path, exist_ok=True)

        evaluator = Evaluator()
        magent = MARLWrap("I", MacLight, alg_args, PolicyNet, ValueNet,
                          state_dim, hidden_dim, action_dim, latent_dim=latent_dim)
        
        # VAE 可选构建
        if args.representation:
            vae = VAE( state_dim=state_dim, latent_dim=latent_dim).to(device)
        else:
            vae = None

        # 训练
        _, train_time = train_ours_agent(
            env=env,
            agents=magent,
            agent_name=agent_names,
            representation=args.representation,
            writer=1,   
            total_episodes=args.episodes,
            seed=seed,
            ckpt_path=ckp_path,
            file_path=file_path,
            evaluator=evaluator,
            max_state=state_dim,
            latent_dim=latent_dim,
            net_file=net_file,
            spec_path=args.spec_path,
            min_green_steps=2,
            hidden_dim=hidden_dim,
            vae=vae,
            reward_helper=helper,          
            reward_scheduler=scheduler,    
            epoch_episodes=1               
        )

        print(f"[Done] seed={seed} | episodes={args.episodes} | time≈{train_time:.1f} min")

    print("\n>>> All seeds finished. <<<\n")


if __name__ == "__main__":
    main()
