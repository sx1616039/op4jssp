import os
import numpy as np
import matplotlib.pyplot as plt
import random


class JobEnv:
    def __init__(self, case_name, path='../all_data_set/', no_op=False):
        self.case_name = case_name
        file = path + case_name + ".txt"
        with open(file, 'r') as f:
            user_line = f.readline()
            data = user_line.split('\t')
            self.m_n = list(map(int, data))
            data = f.read()
            data = str(data).replace('\n', '\t')
            data = str(data).split('\t')
            if data.__contains__(""):
                data.remove("")
            job = list(map(int, data))
            self.job = np.array(job).reshape(self.m_n[0], self.m_n[1] * 2)

        self.job_num = self.m_n[0]
        self.machine_num = self.m_n[1]
        if no_op:
            self.action_num = self.job_num+1
        else:
            self.action_num = self.job_num

        self.current_time = 0  # current time
        self.finished_jobs = None

        self.next_time_on_machine = None
        self.job_on_machine = None
        self.current_op_of_job = None
        self.assignable_job = None
        # state features are above 4 variables
        self.state_num = self.machine_num * 2 + self.job_num + self.action_num
        self.result_dict = {}
        self.state = None

        self.max_op_len = 0
        self.max_job_len = 0
        self.total_process_time = 0
        # find maximum operation length of all jobs
        for j in range(self.job_num):
            job_len = 0
            for i in range(self.machine_num):
                self.total_process_time += self.job[j][i * 2 + 1]
                job_len += self.job[j][i * 2 + 1]
                if self.max_op_len < self.job[j][i * 2 + 1]:
                    self.max_op_len = self.job[j][i * 2 + 1]
            if self.max_job_len < job_len:
                self.max_job_len = job_len

        self.last_release_time = None
        self.done = False
        self.reward = 0
        self.no_op_cnt = 0

    def reset(self):
        self.current_time = 0  # current time
        self.next_time_on_machine = np.repeat(0, self.machine_num)
        self.job_on_machine = np.repeat(-1, self.machine_num)  # -1 implies idle machine
        self.current_op_of_job = np.repeat(0, self.job_num)  # current operation state of job
        self.assignable_job = np.ones(self.action_num, dtype=bool)  # whether a job is assignable
        self.finished_jobs = np.zeros(self.job_num, dtype=bool)

        self.last_release_time = np.repeat(0, self.job_num)
        self.state = []
        self.result_dict = {}
        self.done = False
        self.no_op_cnt = 0
        return self._get_state()

    def _get_state(self):
        self.state = []
        self.state = np.append(self.state, self.assignable_job)
        self.state = np.append(self.state, (self.next_time_on_machine - self.current_time) / self.max_op_len)
        self.state = np.append(self.state, self.job_on_machine / self.job_num)
        self.state = np.append(self.state, self.current_op_of_job / self.machine_num)
        return self.state

    def step(self, action):
        self.done = False
        self.reward = 0
        if action == self.job_num:
            self.no_op_cnt += 1
            self.reward -= self.time_advance()
            self.release_machine()
        else:
            # action is operation
            self.allocate_job(action)

        if self.stop():
            self.done = True
        return self._get_state(), self.reward/self.max_op_len, self.done

    def allocate_job(self, job_id):
        stage = self.current_op_of_job[job_id]
        machine_id = self.job[job_id][stage * 2]
        process_time = self.job[job_id][stage * 2 + 1]

        self.job_on_machine[machine_id] = job_id
        start_time = self.next_time_on_machine[machine_id]
        self.next_time_on_machine[machine_id] += process_time
        end_time = start_time + process_time
        self.result_dict[job_id + 1, machine_id + 1] = start_time, end_time, process_time

        self.last_release_time[job_id] = self.current_time
        self.assignable_job[job_id] = False
        # assignable jobs whose current machine are employed will not be assignable
        for x in range(self.job_num):
            if self.assignable_job[x] and self.job[x][self.current_op_of_job[x] * 2] == machine_id:
                self.assignable_job[x] = False
        # there is no assignable jobs after assigned a job and time advance is needed
        self.reward += process_time
        while sum(self.assignable_job) == 0 and not self.stop():
            self.reward -= self.time_advance()
            self.release_machine()

    def time_advance(self):
        hole_len = 0
        self.current_time = self.find_second_min()
        for machine in range(self.machine_num):
            dist_need_to_advance = self.current_time - self.next_time_on_machine[machine]
            if dist_need_to_advance > 0:
                self.next_time_on_machine[machine] += dist_need_to_advance
                hole_len += dist_need_to_advance
        return hole_len

    def release_machine(self):
        for k in range(self.machine_num):
            cur_job_id = self.job_on_machine[k]
            if cur_job_id >= 0 and self.current_time >= self.next_time_on_machine[k]:
                self.job_on_machine[k] = -1
                self.last_release_time[cur_job_id] = self.current_time
                for x in range(self.job_num):  # release jobs on this machine
                    if not self.finished_jobs[x] and self.job[x][self.current_op_of_job[x] * 2] == k:
                        self.assignable_job[x] = True
                self.current_op_of_job[cur_job_id] += 1
                if self.current_op_of_job[cur_job_id] >= self.machine_num:
                    self.finished_jobs[cur_job_id] = True
                    self.assignable_job[cur_job_id] = False
                else:
                    next_machine = self.job[cur_job_id][self.current_op_of_job[cur_job_id] * 2]
                    if self.job_on_machine[next_machine] >= 0:  # 如果下一工序的机器被占用，则作业不可分配
                        self.assignable_job[cur_job_id] = False

    def stop(self):
        if sum(self.current_op_of_job) < self.machine_num * self.job_num:
            return False
        return True

    def find_second_min(self):
        min_time = min(self.next_time_on_machine)
        second_min_value = 100000
        for value in self.next_time_on_machine:
            if min_time < value < second_min_value:
                second_min_value = value
        if second_min_value == 100000:
            return min_time
        return second_min_value

    def draw_gantt(self, file_name):
        font_dict = {
            "style": "oblique",
            "weight": "bold",
            "color": "white",
            "size": 14
        }
        machine_labels = [" "]  # 生成y轴标签
        for i in range(self.machine_num):
            machine_labels.append("machine " + str(i + 1))
        plt.figure(1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
        colors = ['#%06X' % random.randint(0, 256 ** 3 - 1) for _ in range(800)]
        for k, v in self.result_dict.items():
            plt.barh(y=k[1] - 1, width=v[2], left=v[0], edgecolor="black", color=colors[round(k[0])])
            plt.text(((v[0] + v[1]) / 2), k[1] - 1, str(round(k[0])), fontdict=font_dict)
        plt.yticks([i - 1 for i in range(self.machine_num + 1)], machine_labels)
        plt.title(self.case_name)
        plt.xlabel("time")
        plt.ylabel("machine")
        if not os.path.exists('gantt'):
            os.makedirs('gantt')
        plt.savefig("gantt/" + file_name + ".png")
        plt.close()
