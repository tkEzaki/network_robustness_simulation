import numpy as np
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
import time
from scipy.sparse import csr_matrix
import sys
import random
import math
import pickle
import time

np.set_printoptions(threshold=np.inf)

EPSILON = 0.0001


class Network:
    def __init__(self, network_size, network_type="lattice"):
        {"lattice": self.set_lattice,
         "random": self.set_random_net,
         "hub_and_spoke": self.set_hub_and_spoke,
         "lean_reinforced_hub_and_spoke": self.set_lean_reinforced_hub_and_spoke,
         "reinforced_hub_and_spoke": self.set_reinforced_hub_and_spoke
         }[network_type](network_size)
        self.set_adjacency_matrix_uniform()
        # self.set_shortest_path()
        self.set_shortest_path_and_link_centrality()
        self.link_usage_record = np.zeros((self.node_num, self.node_num))  # 各リンクが使用された回数を記録する
        self.node_usage_record = np.zeros(self.node_num)
        self.link_usage_record_recent = np.zeros((self.node_num, self.node_num))  # 各リンクが使用された回数を記録する
        self.node_usage_record_recent = np.zeros(self.node_num)

    def set_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)), \
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)

    def set_reinforced_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num + 4
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)), \
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)

        self.nodes[25].x, self.nodes[25].y = 0.25, 0.25
        self.nodes[26].x, self.nodes[26].y = -0.25, 0.25
        self.nodes[27].x, self.nodes[27].y = -0.25, -0.25
        self.nodes[28].x, self.nodes[28].y = 0.25, -0.25

        self.nodes[25].neighbors = [1, 4]
        self.nodes[26].neighbors = [1, 2]
        self.nodes[27].neighbors = [2, 3]
        self.nodes[28].neighbors = [3, 4]

        self.nodes[1].neighbors.extend([25, 26])
        self.nodes[2].neighbors.extend([26, 27])
        self.nodes[3].neighbors.extend([27, 28])
        self.nodes[4].neighbors.extend([28, 25])

    def set_lean_reinforced_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)),\
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)

        self.nodes[1].neighbors.extend([2, 4])
        self.nodes[2].neighbors.extend([1, 3])
        self.nodes[3].neighbors.extend([2, 4])
        self.nodes[4].neighbors.extend([3, 1])

    def set_lattice(self, L):
        self.node_num = L * L
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(L):
            for j in range(L):
                self.nodes[i * L + j].x, self.nodes[i * L + j].y \
                    = - 0.5 + 1 / (L - 1) * i, 0.5 - 1 / (L - 1) * j

        # bulk
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                node_id_center = i * L + j
                node_id_left = i * L + j - 1
                node_id_right = i * L + j + 1
                node_id_up = (i - 1) * L + j
                node_id_down = (i + 1) * L + j
                self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # top boundary
        for j in range(1, L - 1):
            node_id_center = 0 * L + j
            node_id_left = 0 * L + j - 1
            node_id_right = 0 * L + j + 1
            # node_id_up = (L - 1) * L + j
            node_id_down = 1 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_down]

        # down boundary
        for j in range(1, L - 1):
            node_id_center = (L - 1) * L + j
            node_id_left = (L - 1) * L + j - 1
            node_id_right = (L - 1) * L + j + 1
            node_id_up = (L - 2) * L + j
            # node_id_down = 0 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up]

        # left boundary
        for i in range(1, L - 1):
            node_id_center = i * L + 0
            # node_id_left = i * L + L - 1
            node_id_right = i * L + 0 + 1
            node_id_up = (i - 1) * L + 0
            node_id_down = (i + 1) * L + 0
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_right, node_id_up, node_id_down]

        # right boundary
        for i in range(1, L - 1):
            node_id_center = i * L + (L - 1)
            node_id_left = i * L + (L - 1) - 1
            # node_id_right = i * L + 0
            node_id_up = (i - 1) * L + (L - 1)
            node_id_down = (i + 1) * L + (L - 1)
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_up, node_id_down]

        # left up corner
        node_id_center = 0
        # node_id_left = (L - 1)
        node_id_right = 1
        # node_id_up = (L - 1) * L
        node_id_down = 1 * L
        # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_down]

        # left down corner
        node_id_center = (L - 1) * L
        # node_id_left = (L - 1) * L + L - 1
        node_id_right = (L - 1) * L + 1
        node_id_up = (L - 2) * L
        # node_id_down = 0
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_up]

        # right up corner
        node_id_center = L - 1
        node_id_left = L - 2
        # node_id_right = 0
        # node_id_up = (L - 1) * L + L - 1
        node_id_down = 1 * L + L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_down]

        # right down corner
        node_id_center = (L - 1) * L + L - 1
        node_id_left = (L - 1) * L + L - 2
        # node_id_right = (L - 1) * L
        node_id_up = (L - 2) * L + L - 1
        # node_id_down = 1 * L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_up]

    def set_lattice_periodic(self, L):
        self.node_num = L * L
        self.nodes = [Node() for i in range(self.node_num)]
        # bulk
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                node_id_center = i * L + j
                node_id_left = i * L + j - 1
                node_id_right = i * L + j + 1
                node_id_up = (i - 1) * L + j
                node_id_down = (i + 1) * L + j
                self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # top boundary
        for j in range(1, L - 1):
            node_id_center = 0 * L + j
            node_id_left = 0 * L + j - 1
            node_id_right = 0 * L + j + 1
            node_id_up = (L - 1) * L + j
            node_id_down = 1 * L + j
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # down boundary
        for j in range(1, L - 1):
            node_id_center = (L - 1) * L + j
            node_id_left = (L - 1) * L + j - 1
            node_id_right = (L - 1) * L + j + 1
            node_id_up = (L - 2) * L + j
            node_id_down = 0 * L + j
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # left boundary
        for i in range(1, L - 1):
            node_id_center = i * L + 0
            node_id_left = i * L + L - 1
            node_id_right = i * L + 0 + 1
            node_id_up = (i - 1) * L + 0
            node_id_down = (i + 1) * L + 0
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # right boundary
        for i in range(1, L - 1):
            node_id_center = i * L + (L - 1)
            node_id_left = i * L + (L - 1) - 1
            node_id_right = i * L + 0
            node_id_up = (i - 1) * L + (L - 1)
            node_id_down = (i + 1) * L + (L - 1)
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # left up corner
        node_id_center = 0
        node_id_left = (L - 1)
        node_id_right = 1
        node_id_up = (L - 1) * L
        node_id_down = 1 * L
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # left down corner
        node_id_center = (L - 1) * L
        node_id_left = (L - 1) * L + L - 1
        node_id_right = (L - 1) * L + 1
        node_id_up = (L - 2) * L
        node_id_down = 0
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # right up corner
        node_id_center = L - 1
        node_id_left = L - 2
        node_id_right = 0
        node_id_up = (L - 1) * L + L - 1
        node_id_down = 1 * L + L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # right down corner
        node_id_center = (L - 1) * L + L - 1
        node_id_left = (L - 1) * L + L - 2
        node_id_right = (L - 1) * L
        node_id_up = (L - 2) * L + L - 1
        node_id_down = 1 * L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        return

    def set_random_net(self, node_num, link_num=40):
        if node_num > link_num:
            print("Warning! Random network is not connected!")
        elif (node_num - 1) * node_num / 2 < link_num:
            print("Warning! Too many links in random network.")

        self.node_num = node_num
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(self.node_num):
            self.nodes[i].x, self.nodes[i].y = \
                0.5 * math.cos(2 * math.pi / self.node_num * i), 0.5 * math.sin(2 * math.pi / self.node_num * i)

        neighbors_list = [[] for i in range(self.node_num)]
        # at least 1 link per node
        for i in range(self.node_num):
            selected_node = random.randrange(self.node_num)
            while selected_node == i or i in neighbors_list[selected_node]:
                selected_node = random.randrange(self.node_num)

            neighbors_list[i].append(selected_node)
            neighbors_list[selected_node].append(i)

        for link in range(link_num - self.node_num):
            selected_node_1 = random.randrange(self.node_num)
            selected_node_2 = random.randrange(self.node_num)
            while selected_node_1 == selected_node_2 \
                    or selected_node_2 in neighbors_list[selected_node_1]:
                selected_node_1 = random.randrange(self.node_num)

            neighbors_list[selected_node_1].append(selected_node_2)
            neighbors_list[selected_node_2].append(selected_node_1)

        for i in range(self.node_num):
            self.nodes[i].neighbors = neighbors_list[i].copy()
        # not yet developed

        return

    def set_adjacency_matrix_uniform(self):
        self.adjacency_matrix = np.zeros((self.node_num, self.node_num))
        self.connectivity_matrix = np.zeros((self.node_num, self.node_num))
        self.link_list = []
        for i in range(self.node_num):
            for j in self.nodes[i].neighbors:
                self.adjacency_matrix[i][j] = 1 + (np.random.rand() - 0.5)
                self.connectivity_matrix[i][j] = 1
                self.link_list.append((i, j))

    def set_shortest_path(self):
        distance, last_visited = shortest_path(self.adjacency_matrix, directed=True, return_predecessors=True)
        # print(last_visited)
        for i in range(self.node_num):  # origin
            for j in range(self.node_num):  # destination
                if i == j:
                    self.nodes[i].shortest_paths.append([i, i])
                else:
                    current_node = j
                    path = []
                    for k in range(self.node_num):
                        path.append(current_node)
                        current_node = last_visited[i][current_node]
                        if current_node == i:  # fin
                            path.append(current_node)
                            self.nodes[i].shortest_paths.append(list(reversed(path)))
                            break

        return

    def set_shortest_path_and_link_centrality(self):
        distance, last_visited = shortest_path(self.adjacency_matrix, directed=True, return_predecessors=True)
        # print(last_visited)
        link_centrality = np.zeros(self.adjacency_matrix.shape)
        for i in range(self.node_num):  # origin
            for j in range(self.node_num):  # destination
                if i == j:
                    self.nodes[i].shortest_paths.append([i, i])
                else:
                    current_node = j
                    path = []
                    for k in range(self.node_num):
                        # update from the destination in the reversed direction
                        previous_node = last_visited[i][current_node]
                        link_centrality[previous_node][current_node] += 1
                        path.append(current_node)
                        current_node = previous_node
                        if current_node == i:  # fin
                            path.append(current_node)
                            self.nodes[i].shortest_paths.append(list(reversed(path)))
                            break

        self.link_centrality = link_centrality

        # print(self.link_centrality)
        return

    def set_capacity_dist(self, capacity_type, capacity_coef):
        if capacity_type == "WBC":
            self.link_capacity = self.connectivity_matrix.copy() \
                + self.link_centrality * capacity_coef \
                / np.sum(self.link_centrality)
        elif capacity_type == "PPL":
            self.link_capacity = self.connectivity_matrix.copy() * (1 + capacity_coef)
        else:
            print("capacity_type: {} unknown.".format(capacity_type))


class HubAndSpokeNetwork(Network):
    def __init__(self, leaf_num=5, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_hub_and_spoke(leaf_num)
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.generate_pattern = generate_pattern
        self.link_usage_record = np.zeros((self.node_num, self.node_num))  # 各リンクが使用された回数を記録する
        self.node_usage_record = np.zeros(self.node_num)

    def set_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)), \
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)


class LatticeNetwork(Network):

    def __init__(self, L: int, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_lattice(L)
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.generate_pattern = generate_pattern

        self.link_usage_record = np.zeros((self.node_num, self.node_num))  # 各リンクが使用された回数を記録する
        self.node_usage_record = np.zeros(self.node_num)

    def set_lattice(self, L):
        self.node_num = L * L
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(L):
            for j in range(L):
                self.nodes[i * L + j].x, self.nodes[i * L + j].y \
                    = - 0.5 + 1 / (L - 1) * i, 0.5 - 1 / (L - 1) * j

        # bulk
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                node_id_center = i * L + j
                node_id_left = i * L + j - 1
                node_id_right = i * L + j + 1
                node_id_up = (i - 1) * L + j
                node_id_down = (i + 1) * L + j
                self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # top boundary
        for j in range(1, L - 1):
            node_id_center = 0 * L + j
            node_id_left = 0 * L + j - 1
            node_id_right = 0 * L + j + 1
            # node_id_up = (L - 1) * L + j
            node_id_down = 1 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_down]

        # down boundary
        for j in range(1, L - 1):
            node_id_center = (L - 1) * L + j
            node_id_left = (L - 1) * L + j - 1
            node_id_right = (L - 1) * L + j + 1
            node_id_up = (L - 2) * L + j
            # node_id_down = 0 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up]

        # left boundary
        for i in range(1, L - 1):
            node_id_center = i * L + 0
            # node_id_left = i * L + L - 1
            node_id_right = i * L + 0 + 1
            node_id_up = (i - 1) * L + 0
            node_id_down = (i + 1) * L + 0
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_right, node_id_up, node_id_down]

        # right boundary
        for i in range(1, L - 1):
            node_id_center = i * L + (L - 1)
            node_id_left = i * L + (L - 1) - 1
            # node_id_right = i * L + 0
            node_id_up = (i - 1) * L + (L - 1)
            node_id_down = (i + 1) * L + (L - 1)
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_up, node_id_down]

        # left up corner
        node_id_center = 0
        # node_id_left = (L - 1)
        node_id_right = 1
        # node_id_up = (L - 1) * L
        node_id_down = 1 * L
        # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_down]

        # left down corner
        node_id_center = (L - 1) * L
        # node_id_left = (L - 1) * L + L - 1
        node_id_right = (L - 1) * L + 1
        node_id_up = (L - 2) * L
        # node_id_down = 0
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_up]

        # right up corner
        node_id_center = L - 1
        node_id_left = L - 2
        # node_id_right = 0
        # node_id_up = (L - 1) * L + L - 1
        node_id_down = 1 * L + L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_down]

        # right down corner
        node_id_center = (L - 1) * L + L - 1
        node_id_left = (L - 1) * L + L - 2
        # node_id_right = (L - 1) * L
        node_id_up = (L - 2) * L + L - 1
        # node_id_down = 1 * L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_up]


class RandomNetwork(Network):
    def __init__(self, node_num, link_num=40, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_random_net(node_num, link_num)
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.generate_pattern = generate_pattern

        self.link_usage_record = np.zeros((self.node_num, self.node_num))  # 各リンクが使用された回数を記録する
        self.node_usage_record = np.zeros(self.node_num)

    def set_random_net(self, node_num, link_num=40):
        if node_num > link_num:
            print("Warning! Random network is not connected!")
        elif (node_num - 1) * node_num / 2 < link_num:
            print("Warning! Too many links in random network.")

        self.node_num = node_num
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(self.node_num):
            self.nodes[i].x, self.nodes[i].y = \
                0.5 * math.cos(2 * math.pi / self.node_num * i), 0.5 * math.sin(2 * math.pi / self.node_num * i)

        neighbors_list = [[] for i in range(self.node_num)]
        # at least 1 link per node
        for i in range(self.node_num):
            selected_node = random.randrange(self.node_num)
            while selected_node == i or i in neighbors_list[selected_node]:
                selected_node = random.randrange(self.node_num)

            neighbors_list[i].append(selected_node)
            neighbors_list[selected_node].append(i)

        for link in range(link_num - self.node_num):
            selected_node_1 = random.randrange(self.node_num)
            selected_node_2 = random.randrange(self.node_num)
            while selected_node_1 == selected_node_2 \
                    or selected_node_2 in neighbors_list[selected_node_1]:
                selected_node_1 = random.randrange(self.node_num)

            neighbors_list[selected_node_1].append(selected_node_2)
            neighbors_list[selected_node_2].append(selected_node_1)

        for i in range(self.node_num):
            self.nodes[i].neighbors = neighbors_list[i].copy()
        # not yet developed

        return


class Node:
    neighbors = []
    quantity = 0
    shortest_paths = []
    current_packets = []

    def __init__(self):
        self.neighbors = []
        self.quantity = 0
        self.shortest_paths = []
        self.current_packet = []
        self.x = 0
        self.y = 0


class Packet:
    def __init__(self, origin, destination, packet_id, path_plan=[], distance=0, steps=0, total_cost=0, is_traced=False,
                 trace_id=-1):
        self.origin = origin
        self.destination = destination
        self.path_plan = path_plan
        self.current_location = origin
        self.packet_id = packet_id
        self.distance = distance
        self.remaining_path_length = len(path_plan)
        self.shortest_remaining_path_length = self.remaining_path_length
        self.steps = steps
        self.total_cost = total_cost
        self.is_traced = is_traced
        self.trace_id = trace_id

    def update_remaining_path_length(self):
        self.remaining_path_length = len(self.path_plan)


class PathPlanner:
    def __init__(self, full_network: Network, capacity_type="link_centrality", capacity_coef=0.0):
        self.path_usage = []
        self.blocked_path = []
        self.full_network = full_network
        self.disturbed_links = []
        if capacity_type == "link_centrality":
            self.link_capacity = self.full_network.connectivity_matrix.copy() \
                + self.full_network.link_centrality * capacity_coef \
                / np.sum(self.full_network.link_centrality)
        elif capacity_type == "double_core":
            self.link_capacity = self.full_network.connectivity_matrix.copy() * (1 + capacity_coef)
            self.link_capacity[0][1] *= 2
            self.link_capacity[0][2] *= 2
            self.link_capacity[0][3] *= 2
            self.link_capacity[0][4] *= 2
            self.link_capacity[1][0] *= 2
            self.link_capacity[2][0] *= 2
            self.link_capacity[3][0] *= 2
            self.link_capacity[4][0] *= 2
        elif capacity_type == "triple_core":
            self.link_capacity = self.full_network.connectivity_matrix.copy() * (1 + capacity_coef)
            self.link_capacity[0][1] *= 3
            self.link_capacity[0][2] *= 3
            self.link_capacity[0][3] *= 3
            self.link_capacity[0][4] *= 3
            self.link_capacity[1][0] *= 3
            self.link_capacity[2][0] *= 3
            self.link_capacity[3][0] *= 3
            self.link_capacity[4][0] *= 3
        else:
            self.link_capacity = self.full_network.connectivity_matrix.copy() * (1 + capacity_coef)

    def add_path_plan(self, path):
        for i in range(len(path) - 1):
            if len(self.path_usage) > i:
                self.path_usage[i].append((path[i], path[i + 1]))
            else:
                self.path_usage.append([])
                self.path_usage[i].append((path[i], path[i + 1]))
        self.compute_blocked_path()  # compute blocked path

    def compute_blocked_path(self):
        new_blocked_path = []
        for t in range(len(self.path_usage)):
            new_capacity_matrix = self.link_capacity.copy()
            blocked_links = []
            for link in self.path_usage[t]:
                new_capacity_matrix[link[0]][link[1]] -= 1.0

            for link in self.full_network.link_list:
                if new_capacity_matrix[link[0]][link[1]] < 1:
                    blocked_links.append(link)

            new_blocked_path.append(blocked_links)

        for t in range(len(self.disturbed_links)):
            if t < len(new_blocked_path):
                new_blocked_path[t].extend(self.disturbed_links[t])
            else:
                new_blocked_path.append(self.disturbed_links[t])

        self.blocked_path = new_blocked_path

    def update_path_usage(self):
        if len(self.path_usage) > 0:
            self.path_usage.pop(0)

    def check_conflict(self, path):
        conflict = False
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in self.path_usage[i]:
                conflict = True
            if conflict:
                break
        return conflict

    def create_time_expanded_network(self, waiting_cost=1, wait_in_advance=False):
        node_num = self.full_network.node_num
        # t_max = len(self.path_usage)
        t_max = len(self.blocked_path)
        nodes = self.full_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.full_network.adjacency_matrix

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = 0.00000001

        for t in range(t_max):
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]
            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + 0.0001
                for j in nodes[i].neighbors:
                    if not (i, j) in occupied_paths:
                        expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def create_time_expanded_network_wait(self, fixed_path, waiting_cost=1, wait_in_advance=False):
        node_num = self.full_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.full_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.full_network.adjacency_matrix
        fixed_paths_tuple = []

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = 0.00000001

        for t in range(len(fixed_path) - 1):
            fixed_paths_tuple.append((fixed_path[t], fixed_path[t + 1]))

        for t in range(t_max):
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]

            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + 0.0001
                for j in nodes[i].neighbors:
                    if (i, j) in fixed_paths_tuple:
                        if not (i, j) in occupied_paths:
                            expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                if (i, j) in fixed_paths_tuple:
                    expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def get_and_register_optimal_path(self, origin, destination, waiting_cost=1):
        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network(waiting_cost=waiting_cost))

        t_max = len(self.blocked_path)
        node_num = self.full_network.node_num

        distance, last_visited = shortest_path(expanded_adjacency_matrix,
                                               indices=origin,
                                               directed=True, return_predecessors=True)

        for t_scan in range(t_max + 1):
            if not np.isinf(distance[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original = self.__convert_location_expand_to_original(node_num, path)
                self.add_path_plan(path_original)

                return distance[t_scan * node_num + destination], path_original

        raise Exception("No shortest path found!")

    def get_and_register_wait_path(self, origin, destination, waiting_cost=1, fixed_path=None):
        if fixed_path is None:
            free_shortest_path = self.full_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network_wait(free_shortest_path,
                                                                                      waiting_cost=waiting_cost))
        distance, last_visited = shortest_path(expanded_adjacency_matrix,
                                               indices=origin,
                                               directed=True, return_predecessors=True)

        t_max = len(self.blocked_path)
        node_num = self.full_network.node_num

        for t_scan in range(t_max + 1):
            if not np.isinf(distance[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original = self.__convert_location_expand_to_original(node_num, path)
                self.add_path_plan(path_original)
                return distance[t_scan * node_num + destination], path_original

        raise Exception("No shortest path found!")

    def compare_paths(self, origin, destination, waiting_cost=1, fixed_path=None):
        if fixed_path is None:
            free_shortest_path = self.full_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix_wait = csr_matrix(self.create_time_expanded_network_wait(free_shortest_path,
                                                                                           waiting_cost=waiting_cost))

        expanded_adjacency_matrix_adaptive = csr_matrix(self.create_time_expanded_network(waiting_cost=waiting_cost))

        distance_wait, last_visited_wait = shortest_path(expanded_adjacency_matrix_wait,
                                                         indices=origin,
                                                         directed=True, return_predecessors=True)

        distance_adaptive, last_visited_adaptive = shortest_path(expanded_adjacency_matrix_adaptive,
                                                                 indices=origin,
                                                                 directed=True, return_predecessors=True)

        t_max = len(self.blocked_path)
        node_num = self.full_network.node_num

        for t_scan in range(t_max + 1):
            if not np.isinf(distance_wait[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited_wait[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original_wait = self.__convert_location_expand_to_original(node_num, path)
                break

        for t_scan in range(t_max + 1):
            if not np.isinf(distance_adaptive[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited_adaptive[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original_adaptive = self.__convert_location_expand_to_original(node_num, path)
                break
        if len(path_original_adaptive) > len(path_original_wait):
            print("--------------------")
            print(path_original_wait)
            print(path_original_adaptive)
            print("--------------------")

    def __convert_location_expand_to_original(self, node_num, path_expanded):
        path_original = []
        for i in path_expanded:
            path_original.append(i % node_num)

        return path_original


class PathPlannerNew:
    def __init__(self, network: Network, waiting_cost: float):
        self.path_usage = []
        self.blocked_path = []
        self.static_network = network
        self.disturbed_links = []
        self.waiting_cost = waiting_cost

    def add_path_plan(self, path):
        for i in range(len(path) - 1):
            if len(self.path_usage) > i:
                self.path_usage[i].append((path[i], path[i + 1]))
            else:
                self.path_usage.append([])
                self.path_usage[i].append((path[i], path[i + 1]))
        self.compute_blocked_path()  # compute blocked path

    def compute_blocked_path(self):
        new_blocked_path = []
        for t in range(len(self.path_usage)):
            new_capacity_matrix = self.static_network.link_capacity.copy()
            blocked_links = []
            for link in self.path_usage[t]:
                new_capacity_matrix[link[0]][link[1]] -= 1.0

            for link in self.static_network.link_list:
                if new_capacity_matrix[link[0]][link[1]] < 1:
                    blocked_links.append(link)

            new_blocked_path.append(blocked_links)

        for t in range(len(self.disturbed_links)):
            if t < len(new_blocked_path):
                new_blocked_path[t].extend(self.disturbed_links[t])
            else:
                new_blocked_path.append(self.disturbed_links[t])

        self.blocked_path = new_blocked_path

    def update_path_usage(self):
        if len(self.path_usage) > 0:
            self.path_usage.pop(0)

    def check_conflict(self, path):
        conflict = False
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in self.path_usage[i]:
                conflict = True
            if conflict:
                break
        return conflict

    def create_time_expanded_network_all(self, waiting_cost=None, wait_in_advance=False):
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        for t in range(t_max):
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]
            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON  # > 0
                for j in nodes[i].neighbors:
                    if not (i, j) in occupied_paths:
                        expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def create_time_expanded_network(self, origin, waiting_cost=None, wait_in_advance=False):
        # Unlike create_time_expanded_network_all,
        # this function wires links that are only reachable from the origin at each time step
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        reachable_nodes = [origin]
        for t in range(t_max):
            newly_reached_nodes = []
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]
            for i in range(len(nodes)):
                if i in reachable_nodes:
                    expanded_adjacency_matrix[t * node_num + i][
                        (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON  # > 0
                    for j in nodes[i].neighbors:
                        if not (i, j) in occupied_paths:
                            expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]
                            newly_reached_nodes.append(j)

            reachable_nodes.extend(newly_reached_nodes)

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def create_time_expanded_network_fixed_path(self, fixed_path, waiting_cost=None, wait_in_advance=False):
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix
        fixed_paths_tuple = []

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        for t in range(len(fixed_path) - 1):
            fixed_paths_tuple.append((fixed_path[t], fixed_path[t + 1]))

        for t in range(t_max):
            occupied_paths = self.blocked_path[t]

            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON
                for j in nodes[i].neighbors:
                    if (i, j) in fixed_paths_tuple:
                        if not (i, j) in occupied_paths:
                            expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                if (i, j) in fixed_paths_tuple:
                    expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def get_and_register_optimal_path(self, origin, destination, waiting_cost=None):
        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network(origin, waiting_cost=waiting_cost))
        distance, path_plan = self.find_fastest_path(expanded_adjacency_matrix, origin, destination)
        self.add_path_plan(path_plan)
        return distance, path_plan

    def get_and_register_fixed_path(self, origin, destination, waiting_cost=None, fixed_path=None):
        if fixed_path is None:
            free_shortest_path = self.static_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network_fixed_path(free_shortest_path,
                                                                                            waiting_cost=waiting_cost))
        distance, path_plan = self.find_fastest_path(expanded_adjacency_matrix, origin, destination)
        self.add_path_plan(path_plan)
        return distance, path_plan

    def compare_paths(self, origin, destination, waiting_cost=None, fixed_path=None):
        if fixed_path is None:
            free_shortest_path = self.static_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix_wait = csr_matrix(self.create_time_expanded_network_fixed_path(free_shortest_path,
                                                                                                 waiting_cost=waiting_cost))

        expanded_adjacency_matrix_adaptive = csr_matrix(self.create_time_expanded_network(waiting_cost=waiting_cost))

        _, path_original_wait = self.find_fastest_path(expanded_adjacency_matrix_wait)
        _, path_original_adaptive = self.find_fastest_path(expanded_adjacency_matrix_adaptive)

        if len(path_original_adaptive) > len(path_original_wait):
            print("--------------------")
            print(path_original_wait)
            print(path_original_adaptive)
            print("--------------------")

    def find_fastest_path(self, expanded_adjacency_matrix, origin, destination):
        distance, last_visited = shortest_path(expanded_adjacency_matrix,
                                               indices=origin,
                                               directed=True, return_predecessors=True)

        t_max = len(self.blocked_path)
        node_num = self.static_network.node_num

        for t_scan in range(t_max + 1):
            if not np.isinf(distance[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original = self.__convert_location_expand_to_original(node_num, path)
                return distance[t_scan * node_num + destination], path_original

        raise Exception("No shortest path found!")

    def __convert_location_expand_to_original(self, node_num, path_expanded):
        path_original = []
        for i in path_expanded:
            path_original.append(i % node_num)

        return path_original


def read_network_data(path: str):
    with open(path, "rb") as f:
        network_data = pickle.load(f)
    network_data["networks"][0].link_usage_record = np.zeros((network_data["networks"][0].node_num, network_data["networks"][0].node_num))  # 各リンクが使用された回数を記録する
    network_data["networks"][0].node_usage_record = np.zeros(network_data["networks"][0].node_num)


if __name__ == "__main__":
    # print("e")
    # n = Network(5, network_type="lattice")
    # pp = PathPlanner(n, capacity_type="link_centrality", capacity_coef=100)
    # print(pp.link_capacity)
    # # ちゃんとキャパシティが機能しているかチェック
    # read_network_data("results//figures//network_data.pkl")
    ln = LatticeNetwork(5)
    hsn = HubAndSpokeNetwork(5)
    rn = RandomNetwork(20, 40)
    pp = PathPlannerNew(hsn, 0.5)
    print(pp.get_and_register_optimal_path(0, 24))
    pold = PathPlanner(hsn)
    print(pold.get_and_register_optimal_path(0, 24))

    times_1 = []
    times_2 = []
    for k in range(100):
        time_0 = time.time()
        pp = PathPlannerNew(hsn, 1.0)

        for i in range(5):
            pp.get_and_register_optimal_path(0, 24)

        time_1 = time.time()

        pold = PathPlanner(hsn)
        for i in range(5):
            pold.get_and_register_optimal_path(0, 24)

        time_2 = time.time()
        times_1.append(time_1 - time_0)
        times_2.append(time_2 - time_1)

    print(np.mean(times_1), np.std(times_1), np.mean(times_2), np.std(times_2))
