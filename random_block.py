import gymnasium as gym
import torch
import traci

# 随机阻塞车道
class BlockStreet:
    def __init__(self, env, block_num=8, seconds=3600) -> None:
        self.env = env
        self.time = 0
        # blockable edges can be refer to light yellow parts in `./doc.map_indicator.pdf`
        self.block_num = block_num
        self.end_time = seconds
        self.agent_name = env.possible_agents

        self.blockable_edges = ['B2C2', 'B3C3', 'C1C2', 'C2B2', 'C2C1',
                                'C2C3', 'C2D2', 'C3B3', 'C3C2', 'C3C4',
                                'C3D3', 'C4C3', 'D1D2', 'D2C2', 'D2D1',
                                'D2D3', 'D2E2', 'D3C3', 'D3D2', 'D3D4',
                                'D3E3', 'D4D3', 'E2D2', 'E3D3',]
        self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]

    def reset(self, seed=None):
        self.time = 0
        return self.env.reset(seed=seed) if seed else self.env.reset()

    def step(self, action):
        '''
        Randomly block 4 road sections every 200 seconds, \n
        unblock them at the end of 200 seconds and set new blocking targets
        '''

        if self.time % 200 != 0:
            for edge_id in self.rd_id:  # 阻塞通行
                traci.edge.setMaxSpeed(self.blockable_edges[edge_id], 0.5)  # m/s
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                traci.vehicle.rerouteTraveltime(vehicle_id)  # 车辆重新规划路径
        else:
            for edge_id in self.rd_id:  # 恢复通行
                traci.edge.setMaxSpeed(
                    self.blockable_edges[edge_id], 13.89)  # m/s
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                traci.vehicle.rerouteTraveltime(vehicle_id)
            self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]  # 重新抽n个车道

        self.time += 5
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.time >= self.end_time:
            done = {agt: True for agt in self.agent_name}
        return next_state, reward, done, truncated, info

    def close(self):
        self.env.close()   