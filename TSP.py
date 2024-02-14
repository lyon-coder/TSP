import numpy as np
import matplotlib
import asyncio
import time
# from IPython.display import clear_output
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from GA import *
from DP import *
from IQL import *

class TSP(object):
    """
    Q-Learning 求解TSP问题
    author: sufer zen
    """
    def __init__(self,
                 num_cities = 200,
                 map_size = (8000.0, 6000.0),
                 alpha = 2,
                 beta = 1, 
                 learning_rate = 1e-3,
                 eps = 0.1,
                 ):
        '''
        Args:
            num_cities (int): 城市数目
            map_size (int, int): 地图尺寸（宽，高）
            alpha (float): 一个超参，值越大，越优先探索最近的点
            beta (float): 一个超参，值越大，越优先探索可能导向总距离最优的点
            learning_rate (float): 学习率
            eps (float): 探索率，值越大，探索性越强，但越难收敛 
        '''
        self.start_city_id = 0
        self.num_cities =num_cities
        self.map_size = map_size
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.learning_rate = learning_rate
        self.cities = self.generate_cities()
        self.distances = self.get_dist_matrix()
        self.mean_distance = self.distances.mean()
        self.qualities = np.zeros([num_cities, num_cities])
        self.normalizers = np.zeros(num_cities)
        self.best_path = None
        self.best_path_length = np.inf


    def generate_cities(self):
        '''
        随机生成城市（坐标）
        Returns:
            cities: [[x1, x2, x3...], [y1, y2, y3...]] 城市坐标
        '''
        max_width, max_height = self.map_size
        cities = np.random.random([2, self.num_cities]) \
            * np.array([max_width, max_height]).reshape(2, -1)
        return cities
    
    def get_cities(self,):
        return self.cities

    def get_dist_matrix(self):
        '''
        根据城市坐标，计算距离矩阵
        '''
        dist_matrix = np.zeros([self.num_cities, self.num_cities])
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i == j:
                    continue
                xi, xj = self.cities[0, i], self.cities[0, j]
                yi, yj = self.cities[1, i], self.cities[1, j]
                dist_matrix[i, j] = np.sqrt((xi-xj)**2 + (yi-yj)**2)
        return dist_matrix

    def rollout(self, start_city_id=None):
        '''
        从 start_city 出发，根据策略，在城市间游走，直到所有城市都走了一遍
        '''
        cities_visited = []
        action_probs = []
        if start_city_id is None:
            start_city_id = np.random.randint(self.num_cities)
        current_city_id = start_city_id
        cities_visited.append(current_city_id)
        while len(cities_visited) < self.num_cities:
            current_city_id, action_prob = self.choose_next_city(cities_visited)
            cities_visited.append(current_city_id)
            action_probs.append(action_prob)
        cities_visited.append(cities_visited[0])
        action_probs.append(1.0)

        path_length = self.calc_path_length(cities_visited)
        if path_length < self.best_path_length:
            self.best_path = cities_visited
            self.best_path_length = path_length
        rewards = self.calc_path_rewards(cities_visited, path_length)
        return cities_visited, action_probs, rewards

    def choose_next_city(self, cities_visited):
        '''
        根据策略选择下一个城市
        '''
        current_city_id = cities_visited[-1]
        
        # 对 quality 取指数，计算 softmax 概率用
        probabilities = np.exp(self.qualities[current_city_id])

        # 将已经走过的城市概率设置为零
        for city_visited in cities_visited:
            probabilities[city_visited] = 0

        # 计算 softmax 概率
        probabilities = probabilities/probabilities.sum()
        
        if np.random.random() < self.eps:
            # 以 eps 概率按softmax概率密度进行随机采样
            next_city_id = np.random.choice(range(len(probabilities)), p=probabilities)
        else:
            # 以 (1 - eps) 概率选择当前最优策略
            next_city_id = probabilities.argmax()

        # 计算当前决策/action 的概率
        if probabilities.argmax() == next_city_id:
            action_prob = probabilities[next_city_id]*self.eps + (1-self.eps)
        else:
            action_prob = probabilities[next_city_id]*self.eps
            
        return next_city_id, action_prob

    def calc_path_rewards(self, path, path_length):
        '''
        计算给定路径的奖励/rewards
        Args:
            path (list[int]): 路径，每个元素代表城市的 id
            path_length (float): 路径长路
        Returns:
            rewards: 每一步的奖励，总距离以及当前这一步的距离越大，奖励越小
        '''
        rewards = []
        for fr, to in zip(path[:-1], path[1:]):
            dist = self.distances[fr, to]
            reward = (self.mean_distance/path_length)**self.beta
            reward = reward*(self.mean_distance/dist)**self.alpha
            rewards.append(np.log(reward))
        return rewards

    def calc_path_length(self, path):
        '''
        计算路径长度
        '''
        path_length = 0
        for fr, to in zip(path[:-1], path[1:]):
            path_length += self.distances[fr, to]
        return path_length
    
    def calc_updates_for_one_rollout(self, path, action_probs, rewards):
        '''
        对于给定的一次 rollout 的结果，计算其对应的 qualities 和 normalizers 
        '''
        qualities = []
        normalizers = []
        for fr, to, reward, action_prob in zip(path[:-1], path[1:], rewards, action_probs):
            log_action_probability = np.log(action_prob)
            qualities.append(- reward*log_action_probability)
            normalizers.append(- (reward + 1)*log_action_probability)
        return qualities, normalizers

    def update(self, path, new_qualities, new_normalizers):
        '''
        用渐近平均的思想，对 qualities 和 normalizers 进行更新
        '''
        lr = self.learning_rate
        for fr, to, new_quality, new_normalizer in zip(
            path[:-1], path[1:], new_qualities, new_normalizers):
            self.normalizers[fr] = (1-lr)*self.normalizers[fr] + lr*new_normalizer
            self.qualities[fr, to] = (1-lr)*self.qualities[fr, to] + lr*new_quality
    
    async def train_for_one_rollout(self, start_city_id):
        '''
        对一次 rollout 的结果进行训练的流程
        '''
        path, action_probs, rewards = self.rollout(start_city_id=start_city_id)
        new_qualities, new_normalizers = self.calc_updates_for_one_rollout(path, action_probs, rewards)
        self.update(path, new_qualities, new_normalizers)

    async def train_for_one_epoch(self):
        '''
        对一个 epoch 的结果进行训练的流程，
        一个 epoch 对应于从每个 city 出发进行一次 rollout
        '''
        # tasks = [self.train_for_one_rollout(start_city_id) for start_city_id in range(self.num_cities)]
        tasks = [self.train_for_one_rollout(self.start_city_id)]
        await asyncio.gather(*tasks)

    async def train(self, num_epochs=1000, display=False):
        '''
        总训练流程
        '''
        Q_start = time.time()
        for epoch in range(num_epochs):
            await self.train_for_one_epoch()
            if display:
                self.draw(epoch)
        Q_end = time.time()
        print("Q Learning time is ", Q_end - Q_start)
        self.draw_result(num_epochs)

    def draw(self, epoch):
        '''
        绘图
        '''
        # _ = plt.scatter(*self.cities)
        # for fr, to in zip(self.best_path[:-1], self.best_path[1:]):
        #     x1, y1 = self.cities[:, fr]
        #     x2, y2 = self.cities[:, to]
        #     dx, dy = x2-x1, y2-y1
        #     plt.arrow(x1, y1, dx, dy, width=0.01*min(self.map_size), 
        #               edgecolor='orange', facecolor='white', animated=True, 
        #               length_includes_head=True)
        # nrs = np.exp(self.qualities)
        # for i in range(self.num_cities):
        #     nrs[i, i] = 0
        # gap = np.abs(np.exp(self.normalizers) - nrs.sum(-1)).mean()
        # plt.title(f'epoch {epoch}: path length = {self.best_path_length:.2f}, normalizer error = {gap:.3f}')
        # plt.savefig('tsp.png')
        # if s:
        #     plt.show()
        # plt.close()
        # # clear_output(wait=True)
        
    def draw_result(self, epoch):
        for fr, to in zip(self.best_path[:-1], self.best_path[1:]):
            print(fr, "----->", to)
        print("path length: ", self.best_path_length)
        _ = plt.scatter(*self.cities)
        for fr, to in zip(self.best_path[:-1], self.best_path[1:]):
            x1, y1 = self.cities[:, fr]
            x2, y2 = self.cities[:, to]
            dx, dy = x2-x1, y2-y1
            plt.arrow(x1, y1, dx, dy, width=0.01*min(self.map_size), 
                    edgecolor='orange', facecolor='white', animated=True, 
                    length_includes_head=True)
        nrs = np.exp(self.qualities)
        for i in range(self.num_cities):
            nrs[i, i] = 0
        gap = np.abs(np.exp(self.normalizers) - nrs.sum(-1)).mean()
        plt.title(f'epoch {epoch}: path length = {self.best_path_length:.2f}, normalizer error = {gap:.3f}')
        plt.savefig('tsp.png')
        plt.close()

if __name__ == '__main__':
    tsp = TSP()
    cities = tsp.get_cities()
    episodes = 2000
    # Q Learning
    asyncio.run(tsp.train(episodes))

    # GA
    # GA参数
    generation = episodes # 迭代次数
    popsize = 100  # 种群大小
    tournament_size = 5  # 锦标赛小组大小
    pc = 0.95  # 交叉概率
    pm = 0.1  # 变异概率

    CityCoordinates = [(cities[0][i], cities[1][i]) for i in range(len(cities[0]))]

    TSP_GA(CityCoordinates, generation, popsize, tournament_size, pc, pm)

    # # DP
    # TSP_DP(CityCoordinates)

    # IQL
    iql = TSP_IQL(cities)
    asyncio.run(iql.train(episodes))


