import network
import numpy as np
import random
import time
import datetime
from operator import attrgetter

np.set_printoptions(threshold=np.inf)


class OptimalTransportSimulator():
    def __init__(self, network_size, demand_rate, waiting_cost, fixed_network=None,
                 network_type="lattice", capacity_type="link_centrality", capacity_coef=0.0):
        if fixed_network is None:
            self.static_network = network.Network(network_size, network_type=network_type)
        else:
            self.static_network = fixed_network

        self.path_planner = network.PathPlanner(
            self.static_network, capacity_type=capacity_type, capacity_coef=capacity_coef
        )
        self.packets = []
        self.next_packet_id = 0
        self.waiting_cost = waiting_cost
        self.demand_rate = demand_rate
        self.cost_records = []
        self.move_records = []
        self.path_length_records = []
        self.num_packets_records = []
        self.recording = False
        self.monitor = False
        self.disturbed_links_plan = []
        self.traced_steps_records = []
        self.traced_cost_records = []

        # print("Network constructed.")

    def add_single_optimal_packet(self, origin, destination, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        distance, path_plan = self.path_planner.get_and_register_optimal_path(origin, destination,
                                                                              waiting_cost=self.waiting_cost)

        packet = network.Packet(origin, destination, self.next_packet_id, path_plan=path_plan, distance=distance,
                                steps=steps, total_cost=total_cost, is_traced=is_traced, trace_id=trace_id)
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (origin, destination), ", path:", path_plan)

    def add_single_optimal_packet_random(self, generate_pattern=None, is_traced=False, packet_size=1, trace_id=-1):
        if generate_pattern is None:
            origin = random.randrange(self.static_network.node_num)
            destination = random.randrange(self.static_network.node_num)
            while destination == origin:
                destination = random.randrange(self.static_network.node_num)
        else:
            origin = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[0])[0]
            destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]
            while destination == origin:
                destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]

        for i in range(packet_size):
            self.add_single_optimal_packet(origin, destination, is_traced=is_traced, trace_id=trace_id)

    def add_single_fixed_packet(self, origin, destination, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        # packet delivered on a predefined route
        distance, path_plan = self.path_planner.get_and_register_wait_path(
            origin, destination, waiting_cost=self.waiting_cost
        )

        packet = network.Packet(
            origin, destination, self.next_packet_id, path_plan=path_plan,
            distance=distance, steps=steps, total_cost=total_cost, is_traced=is_traced, trace_id=trace_id
        )
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (origin, destination), ", path:", path_plan)

    def add_single_wait_packet(self, fixed_path, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        # packet delivered on a given route
        distance, path_plan = self.path_planner.get_and_register_wait_path(
            fixed_path[0], fixed_path[-1], waiting_cost=self.waiting_cost, fixed_path=fixed_path
        )

        packet = network.Packet(fixed_path[0], fixed_path[-1], self.next_packet_id, path_plan=path_plan,
                                distance=distance, steps=steps, total_cost=total_cost,
                                is_traced=is_traced, trace_id=trace_id
                                )
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (fixed_path[0], fixed_path[-1]), ", path:", path_plan)

    def add_single_fixed_packet_random(self, generate_pattern=None, packet_size=1, is_traced=False, trace_id=-1):
        if generate_pattern is None:
            origin = random.randrange(self.static_network.node_num)
            destination = random.randrange(self.static_network.node_num)
            while destination == origin:
                destination = random.randrange(self.static_network.node_num)
        else:
            origin = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[0])[0]
            destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]
            while destination == origin:
                destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]

        for i in range(packet_size):
            self.add_single_fixed_packet(origin, destination, is_traced=is_traced, trace_id=trace_id)

    def add_optimal_packets(self):
        update_list = random.sample(list(range(self.static_network.node_num)), self.static_network.node_num)
        for node in update_list:
            if random.random() < self.demand_rate:
                destination = random.randrange(self.static_network.node_num)
                while destination == node:
                    destination = random.randrange(self.static_network.node_num)
                self.add_single_optimal_packet(node, destination)

    def add_fixed_packets(self):
        update_list = random.sample(list(range(self.static_network.node_num)), self.static_network.node_num)
        for node in update_list:
            if random.random() < self.demand_rate:
                destination = random.randrange(self.static_network.node_num)
                while destination == node:
                    destination = random.randrange(self.static_network.node_num)
                self.add_single_fixed_packet(node, destination)

    def move_packets(self):
        arrived_packets = []
        moved_packets = []

        for i in range(len(self.packets)):
            if not self.packets[i].current_location == \
                    self.packets[i].path_plan[1]:
                moved_packets.append(i)
            self.packets[i].path_plan.pop(0)
            self.packets[i].current_location = self.packets[i].path_plan[0]
            self.packets[i].update_remaining_path_length()
            self.packets[i].shortest_remaining_path_length = \
                len(self.static_network
                    .nodes[self.packets[i].current_location]
                    .shortest_paths[self.packets[i].destination])

            if len(self.packets[i].path_plan) == 1:  # arrived
                arrived_packets.append(i)

        num_deleted = 0
        for i in arrived_packets:
            del self.packets[i - num_deleted]
            num_deleted += 1

        if self.recording:
            self.move_records.append(len(moved_packets))
            self.num_packets_records.append(len(self.packets) + len(arrived_packets))

        if self.monitor:
            print("Packets moved.", len(self.packets), "packets")

        return len(arrived_packets), len(moved_packets)

    def move_and_regenerate_packets(self, packet_type="optimal", generate_pattern=None):
        arrived_packets = []
        moved_packets = []
        num_arrived_traced_packets = 0

        for i in range(len(self.packets)):
            self.packets[i].steps += 1
            self.static_network.node_usage_record[self.packets[i].current_location] += 1

            if not self.packets[i].current_location == self.packets[i].path_plan[1]:  # moved
                moved_packets.append(i)
                self.packets[i].total_cost += \
                    self.static_network.adjacency_matrix[self.packets[i].path_plan[0]][self.packets[i].path_plan[1]]
                self.static_network.link_usage_record[self.packets[i].path_plan[0], self.packets[i].path_plan[1]] += 1
            else:
                self.packets[i].total_cost += self.waiting_cost

            self.packets[i].path_plan.pop(0)
            self.packets[i].current_location = self.packets[i].path_plan[0]
            self.packets[i].update_remaining_path_length()
            self.packets[i].shortest_remaining_path_length = \
                len(self.static_network
                    .nodes[self.packets[i].current_location]
                    .shortest_paths[self.packets[i].destination])

            if len(self.packets[i].path_plan) == 1:  # arrived
                arrived_packets.append(i)
                if self.recording:
                    if self.packets[i].is_traced:
                        num_arrived_traced_packets += 1
                        if len(self.traced_steps_records) <= self.packets[i].trace_id:
                            self.traced_steps_records.append([])
                            self.traced_cost_records.append([])
                            self.traced_steps_records[self.packets[i].trace_id].append(self.packets[i].steps)
                            self.traced_cost_records[self.packets[i].trace_id].append(self.packets[i].total_cost)
                        else:
                            self.traced_steps_records[self.packets[i].trace_id].append(self.packets[i].steps)
                            self.traced_cost_records[self.packets[i].trace_id].append(self.packets[i].total_cost)
                    else:
                        self.path_length_records.append(self.packets[i].steps)
                        self.cost_records.append(self.packets[i].total_cost)

        if len(self.path_planner.blocked_path) > 0:
            self.path_planner.blocked_path.pop(0)

        if len(self.disturbed_links_plan) > 0:
            self.disturbed_links_plan.pop(0)
            self.path_planner.disturbed_links.pop(0)

        num_deleted = 0
        for i in arrived_packets:
            del self.packets[i - num_deleted]
            num_deleted += 1

        if self.recording:
            self.move_records.append(len(moved_packets))
            self.num_packets_records.append(len(self.packets) + len(arrived_packets))

        self.path_planner.update_path_usage()

        if self.monitor:
            print("Packets moved.", len(self.packets), "packets. Replenish new packets.---move_and_regenerate_packets")

        # regenerate particles
        for i in range(num_deleted - num_arrived_traced_packets):
            {"optimal": self.add_single_optimal_packet_random,
             "fixed": self.add_single_fixed_packet_random,
             "adaptive": self.add_single_optimal_packet_random}[packet_type](generate_pattern=generate_pattern)

        if self.monitor:
            print("Replenishment done.", len(self.packets), "packets---move_and_regenerate_packets")

        return len(arrived_packets), len(moved_packets)

    def generate_traced_packets(self, packet_size, packet_type, trace_id, generate_pattern=None):

        {"optimal": self.add_single_optimal_packet_random,
         "fixed": self.add_single_fixed_packet_random,
         "adaptive": self.add_single_optimal_packet_random}[packet_type](
            generate_pattern=generate_pattern, packet_size=packet_size, is_traced=True, trace_id=trace_id
        )

    def regenerate_all_paths(self, packet_type, reverse=False, id_sort=False, capacity_type="link_centrality",
                             capacity_coef=0.0):
        if packet_type == "adaptive":
            self.packets.sort(key=attrgetter("shortest_remaining_path_length"), reverse=reverse)

        if id_sort:
            self.packets.sort(key=attrgetter("packet_id"), reverse=reverse)

        self.path_planner = network.PathPlanner(self.static_network, capacity_type=capacity_type,
                                                capacity_coef=capacity_coef)  # new
        self.path_planner.blocked_path = self.disturbed_links_plan.copy()
        self.path_planner.disturbed_links = self.disturbed_links_plan.copy()

        packets_temp = self.packets.copy()
        self.packets = []
        self.next_packet_id -= len(packets_temp)

        if self.monitor:
            print("Re-optimize path.---regenerate_all_paths")

        for packet in packets_temp:
            origin = packet.current_location
            destination = packet.destination
            {"optimal": self.add_single_optimal_packet,
             "fixed": self.add_single_fixed_packet}[packet_type](
                origin, destination, steps=packet.steps, total_cost=packet.total_cost,
                is_traced=packet.is_traced, trace_id=packet.trace_id
            )

        if self.monitor:
            print("Re-optimization done.---regenerate_all_paths")

    def disturbance_update_all_paths(self, packet_type, disturbance_type="random", fixed_disturbance=[],
                                     disturbance_length=5, disturbance_num=2, reverse=False, id_sort=False,
                                     capacity_type="link_centrality", capacity_coef=0.0):
        disturbed_links = []
        for i in range(disturbance_num):
            disturbed_node = random.randrange(self.static_network.node_num)
            next_node = random.choice(self.static_network.nodes[disturbed_node].neighbors)
            while (disturbed_node, next_node) in disturbed_links:
                disturbed_node = random.randrange(self.static_network.node_num)
                next_node = random.choice(self.static_network.nodes[disturbed_node].neighbors)
            disturbed_links.append((disturbed_node, next_node))

        disturbed_links_plan = [disturbed_links.copy() for i in range(disturbance_length)]
        if self.monitor:
            print("Disturbed_link_updated:", disturbed_links_plan, "---disturbance_update_all_paths")

        # sort before copy
        if id_sort:
            self.packets.sort(key=attrgetter("packet_id"), reverse=reverse)
        else:
            self.packets.sort(key=attrgetter("shortest_remaining_path_length"), reverse=reverse)

        # recreate
        self.path_planner = network.PathPlanner(self.static_network, capacity_type=capacity_type,
                                                capacity_coef=capacity_coef)  # new
        packets_temp = self.packets.copy()
        self.packets = []
        self.next_packet_id -= len(packets_temp)
        self.path_planner.blocked_path = disturbed_links_plan.copy()
        self.path_planner.disturbed_links = disturbed_links_plan.copy()

        self.disturbed_links_plan = disturbed_links_plan.copy()

        if self.monitor:
            print("Re-calculate path plans.---disturbance_update_all_paths")

        for packet in packets_temp:
            if packet_type == "optimal" or packet_type == "fixed":
                self.add_single_wait_packet(
                    packet.path_plan, steps=packet.steps, total_cost=packet.total_cost,
                    is_traced=packet.is_traced, trace_id=packet.trace_id
                )
            elif packet_type == "adaptive":
                origin = packet.current_location
                destination = packet.destination
                self.add_single_optimal_packet(
                    origin, destination, steps=packet.steps, total_cost=packet.total_cost,
                    is_traced=packet.is_traced, trace_id=packet.trace_id
                )

        if self.monitor:
            print("Path plans recalculated.---disturbance_update_all_paths")

    def compute_average_cost(self, threshold=0):
        return np.mean(self.cost_records[threshold:])

    def compute_std_cost(self, threshold=0):
        return np.std(self.cost_records[threshold:])

    def compute_average_move(self, threshold=0):
        return np.mean(self.move_records[threshold:])

    def compute_std_move(self, threshold=0):
        return np.mean(self.move_records[threshold:])

    def compute_average_path_length(self, threshold=0):
        return np.mean(self.path_length_records[threshold:])

    def compute_std_path_length(self, threshold=0):
        return np.std(self.path_length_records[threshold:])

    def compute_average_num_packets(self, threshold=0):
        return np.mean(self.num_packets_records[threshold:])

    def compute_average_steps_traced_packets(self):
        if len(self.traced_steps_records) > 1:
            steps_ndarray = np.array(self.traced_steps_records[:-1])
            return steps_ndarray.mean(axis=0)
        else:
            return [0]

    def compute_std_steps_traced_packets(self):
        if len(self.traced_steps_records) > 1:
            steps_ndarray = np.array(self.traced_steps_records[:-1])
            return steps_ndarray.std(axis=0)
        else:
            return [0]

    def compute_average_cost_traced_packets(self):
        if len(self.traced_cost_records) > 1:
            cost_ndarray = np.array(self.traced_cost_records[:-1])
            return cost_ndarray.mean(axis=0)
        else:
            return [0]

    def compute_std_cost_traced_packets(self):
        if len(self.traced_cost_records) > 1:
            cost_ndarray = np.array(self.traced_cost_records[:-1])
            return cost_ndarray.std(axis=0)
        else:
            return [0]


def simulator_closed_system(
    given_network=None, network_size=5, num_packets=1, waiting_cost=0, packet_type="optimal",
    id_sort=False, network_type="lattice", capacity_type="link_centrality",
    capacity_coef=0.0, with_disturbance=False, disturbance_type="random", fixed_disturbance=[],
    disturbance_length=5, disturbance_num=2, generate_pattern=None, is_traced=False, packet_size=1,
    generate_pattern_trace=None, T_max=1000, condition_memo="memo"
):
    time_start = time.time()
    ots = OptimalTransportSimulator(network_size, 0, waiting_cost, fixed_network=given_network, network_type=network_type,
                                    capacity_type=capacity_type, capacity_coef=capacity_coef)
    ots.recording = True
    ots.monitor = False

    for i in range(num_packets):  # initialize packets
        {"optimal": ots.add_single_optimal_packet_random,
         "fixed": ots.add_single_fixed_packet_random,
         "adaptive": ots.add_single_optimal_packet_random}[packet_type]()

    if is_traced:
        current_trace_id = 0
        ots.generate_traced_packets(
            packet_size, packet_type, trace_id=current_trace_id, generate_pattern=generate_pattern_trace
        )

    if with_disturbance:
        ots.disturbance_update_all_paths(
            packet_type, disturbance_type=disturbance_type,
            fixed_disturbance=fixed_disturbance, disturbance_length=disturbance_length,
            disturbance_num=disturbance_num, id_sort=id_sort,
            capacity_type=capacity_type, capacity_coef=capacity_coef
        )

    for t in range(T_max):
        ots.move_and_regenerate_packets(packet_type=packet_type, generate_pattern=generate_pattern)
        if is_traced:
            if len(ots.packets) == num_packets:
                current_trace_id += 1
                ots.generate_traced_packets(
                    packet_size, packet_type, trace_id=current_trace_id, generate_pattern=generate_pattern_trace
                )

        if packet_type == "adaptive":
            ots.regenerate_all_paths(packet_type="optimal", reverse=False, capacity_type=capacity_type,
                                     capacity_coef=capacity_coef)

        if with_disturbance:
            if t % disturbance_length == 0:  # update disturbance
                ots.disturbance_update_all_paths(
                    packet_type, disturbance_type=disturbance_type,
                    fixed_disturbance=fixed_disturbance, disturbance_length=disturbance_length,
                    disturbance_num=disturbance_num, id_sort=id_sort,
                    capacity_type=capacity_type, capacity_coef=capacity_coef
                )

    # temp
    print(ots.traced_steps_records, ots.traced_cost_records)
    # ttemp

    print(num_packets, waiting_cost, packet_type, ots.compute_average_steps_traced_packets(),
          ots.compute_std_steps_traced_packets(),
          ots.compute_average_num_packets(), ots.compute_average_move(),
          ots.compute_average_path_length(), ots.compute_average_cost())
    return {
        "condition_memo": condition_memo,
        "execution_time": time.time() - time_start,
        "date_of_execution": str(datetime.datetime.today()),
        "network_size": network_size, "num_packets": num_packets, "waiting_cost": waiting_cost,
        "packet_type": packet_type,
        "id_sort": id_sort, "network_type": network_type,
        "capacity_type": capacity_type, "capacity_coeff": capacity_coef, "with_disturbance": with_disturbance,
        "disturbance_type": disturbance_type,
        "disturbance_num": disturbance_num,
        "fixed_disturbance": fixed_disturbance, "disturbance_length": disturbance_length,
        "generate_pattern": generate_pattern, "is_traced": is_traced, "packet_size": packet_size,
        "generate_pattern_trace": generate_pattern_trace, "T_max": T_max,
        "average_num_packets": ots.compute_average_num_packets(),
        "average_move": ots.compute_average_move(),
        "std_move": ots.compute_std_move(),
        "average_path_length": ots.compute_average_path_length(),
        "std_path_length": ots.compute_std_path_length(),
        "average_cost": ots.compute_average_cost(),
        "std_cost": ots.compute_std_cost(),
        "average_path_length_traced": ots.compute_average_steps_traced_packets(),
        "std_path_length_traced": ots.compute_std_steps_traced_packets(),
        "average_cost_traced": ots.compute_average_cost_traced_packets(),
        "std_cost_traced": ots.compute_std_cost_traced_packets(),
        "node_usage": ots.static_network.node_usage_record / T_max,
        "link_usage": ots.static_network.link_usage_record / T_max,
        "network_data": {
            "network_structure": ots.static_network,
            "capacity_dist": ots.path_planner.link_capacity
        }
    }


def simulator_closed_system_given_network(
        given_network,
        network_size=5, num_packets=1, waiting_cost=0, packet_type="optimal",
        id_sort=False, network_type="lattice", capacity_type="single_packet_per_link",
        capacity_coef=0.0, with_disturbance=False, disturbance_type="random", fixed_disturbance=[],
        disturbance_length=5, disturbance_num=2, generate_pattern=None, is_traced=False, packet_size=1,
        generate_pattern_trace=None, T_max=1000, condition_memo="memo"
):
    ots = OptimalTransportSimulator(network_size, 0, waiting_cost, fixed_network=given_network, network_type=network_type,
                                    capacity_type=capacity_type, capacity_coef=capacity_coef)
    ots.static_network.link_usage_record = np.zeros((ots.static_network.node_num, ots.static_network.node_num))
    ots.static_network.node_usage_record = np.zeros(ots.static_network.node_num)

    ots.recording = True
    ots.monitor = False

    for i in range(num_packets):
        {"optimal": ots.add_single_optimal_packet_random,
         "fixed": ots.add_single_fixed_packet_random,
         "adaptive": ots.add_single_optimal_packet_random}[packet_type]()

    if is_traced:
        current_trace_id = 0
        ots.generate_traced_packets(
            packet_size, packet_type, trace_id=current_trace_id, generate_pattern=generate_pattern_trace
        )

    if with_disturbance:
        ots.disturbance_update_all_paths(
            packet_type, disturbance_type=disturbance_type,
            fixed_disturbance=fixed_disturbance, disturbance_length=disturbance_length,
            disturbance_num=disturbance_num, id_sort=id_sort,
            capacity_type=capacity_type, capacity_coef=capacity_coef
        )

    for t in range(T_max):
        ots.move_and_regenerate_packets(packet_type=packet_type, generate_pattern=generate_pattern)
        if is_traced:
            if len(ots.packets) == num_packets:
                current_trace_id += 1
                ots.generate_traced_packets(
                    packet_size, packet_type, trace_id=current_trace_id, generate_pattern=generate_pattern_trace
                )

        if packet_type == "adaptive":
            ots.regenerate_all_paths(packet_type="optimal", reverse=False, capacity_type=capacity_type,
                                     capacity_coef=capacity_coef)

        if with_disturbance:
            if t % disturbance_length == 0:  # update disturbance
                ots.disturbance_update_all_paths(
                    packet_type, disturbance_type=disturbance_type,
                    fixed_disturbance=fixed_disturbance, disturbance_length=disturbance_length,
                    disturbance_num=disturbance_num, id_sort=id_sort,
                    capacity_type=capacity_type, capacity_coef=capacity_coef
                )

    return ots.static_network.node_usage_record, ots.static_network.link_usage_record


if __name__ == '__main__':
    # example
    results = []
    num_packets_list = [50]
    network_size = 5
    waiting_cost = 0.5
    start = time.time()
    for num_packets in num_packets_list:
        condition = {
            "network_size": network_size, "num_packets": num_packets, "waiting_cost": waiting_cost,
            "packet_type": "adaptive",
            "id_sort": True, "network_type": "hub_and_spoke",
            "capacity_type": "uniform", "capacity_coef": 1, "with_disturbance": True, "disturbance_type": "random",
            "fixed_disturbance": [], "disturbance_length": 5, "disturbance_num": 12,
            "generate_pattern": None, "is_traced": False, "packet_size": 2,
            "generate_pattern_trace": None, "T_max": 1000
        }
        print(simulator_closed_system(**condition))

    print(time.time() - start)
