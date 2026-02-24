
# 统计：ATT, AWT, AS, TP + 每一步的 waiting / queue / speed 序列

from collections import defaultdict

class SimulationMetrics:
    WAIT_SPEED_THRESH = 0.1  # m/s, 小于这个算在等

    def __init__(self):
        self.reset()

    def reset(self):
        # --- 给 ATT / AWT / TP 用的 ---
        self._depart_time = {}             # veh_id -> depart_time
        self._wait_time = defaultdict(float)
        self.total_travel_time = 0.0
        self.total_wait_time = 0.0
        self.completed_trips = 0

        # --- 给 AS 用的（全局平均速度）---
        self.speed_sum = 0.0
        self.speed_count = 0

        # --- 给你要的三条“每一步”序列用的 ---
        self.waiting_series = []   # 每一步：多少辆车速度 < 0.1
        self.queue_series = []     # 每一步：所有 lane 的 halting 车之和
        self.speed_series = []     # 每一步：正在路网里的车的平均速度

    def step(self, traci_conn):
        sim_time = traci_conn.simulation.getTime()
        step_len = traci_conn.simulation.getDeltaT() / 1000.0

        # 1) 新进来的车
        for vid in traci_conn.simulation.getDepartedIDList():
            self._depart_time[vid] = sim_time
            # 确保有等待时间记录
            if vid not in self._wait_time:
                self._wait_time[vid] = 0.0

        # 2) 当前在网里的车，用来统计“这一步”的情况
        running_veh = traci_conn.vehicle.getIDList()
        waiting_cnt_step = 0
        speed_sum_step = 0.0
        speed_cnt_step = 0

        for vid in running_veh:
            v = traci_conn.vehicle.getSpeed(vid)
            # 全局平均速度的累加
            self.speed_sum += v
            self.speed_count += 1

            # 这一步的平均速度
            speed_sum_step += v
            speed_cnt_step += 1

            # 等待判定
            if v < self.WAIT_SPEED_THRESH:
                waiting_cnt_step += 1
                self._wait_time[vid] += step_len

        # 3) 这一帧的“排队长度” -> 所有 lane 的 halting 数之和
        queue_cnt_step = 0
        for lane_id in traci_conn.lane.getIDList():
            queue_cnt_step += traci_conn.lane.getLastStepHaltingNumber(lane_id)

        # 4) 这一帧的“平均速度”
        if speed_cnt_step > 0:
            avg_speed_step = speed_sum_step / speed_cnt_step
        else:
            avg_speed_step = 0.0

        # 存三条序列
        self.waiting_series.append(waiting_cnt_step)
        self.queue_series.append(queue_cnt_step)
        self.speed_series.append(avg_speed_step)

        # 5) 到达的车 -> 用来算 ATT / AWT / TP
        for vid in traci_conn.simulation.getArrivedIDList():
            depart_t = self._depart_time.pop(vid, sim_time)
            travel_time = sim_time - depart_t
            self.total_travel_time += travel_time

            wait_t = self._wait_time.pop(vid, 0.0)
            self.total_wait_time += wait_t

            self.completed_trips += 1

    def get_metrics(self):
        # ATT / AWT
        if self.completed_trips > 0:
            att = self.total_travel_time / self.completed_trips
            awt = self.total_wait_time / self.completed_trips
        else:
            att = 0.0
            awt = 0.0

        # 全局平均速度 AS
        if self.speed_count > 0:
            avg_speed = self.speed_sum / self.speed_count
        else:
            avg_speed = 0.0

        return {
            "ATT": att,
            "AWT": awt,
            "AS": avg_speed,
            "TP": self.completed_trips,

            # 三条序列也一并返回，待会存 CSV 要用
            "waiting_series": self.waiting_series,
            "queue_series": self.queue_series,
            "speed_series": self.speed_series,
        }
