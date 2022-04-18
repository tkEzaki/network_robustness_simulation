import simulate_optimal_transport
from multiprocessing import Pool
import time
import uuid
import pickle
import os


def simulator_wrapper(kwargs):
    results = simulate_optimal_transport.simulator_closed_system(**kwargs)
    print(results)
    with open("results/pool/10_trace/" + str(uuid.uuid4()) + ".pkl", "wb") as tf:
        pickle.dump(results, tf)


if __name__ == "__main__":
    network_info_list = [
        {"network_type": "random",
         "network_size": 25,
         "capacity_conditions": [
             {"capacity_type": "link_centrality",
              "capacity_coef": 0},
             {"capacity_type": "link_centrality",
              "capacity_coef": 80},
             {"capacity_type": "uniform",
              "capacity_coef": 1}
         ]
         },
        {"network_type": "lattice",
            "network_size": 5,
            "capacity_conditions": [
                {"capacity_type": "link_centrality",
                 "capacity_coef": 0},
                {"capacity_type": "link_centrality",
                 "capacity_coef": 80},
                {"capacity_type": "uniform",
                 "capacity_coef": 1}
            ]
         },
        {"network_type": "hub_and_spoke",
            "network_size": 5,
            "capacity_conditions": [
                {"capacity_type": "link_centrality",
                 "capacity_coef": 0},
                {"capacity_type": "link_centrality",
                 "capacity_coef": 48},
                {"capacity_type": "uniform",
                 "capacity_coef": 1}
            ]
         }
    ]

    num_packets_list = [1, 20, 40, 60, 80, 100]
    waiting_cost_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    packet_size_list = [2, 5]

    packet_type_list = ["fixed", "optimal", "adaptive"]
    T_max = 1000
    core_num = os.cpu_count() - 2

    start = time.time()
    conditions = []

    for packet_type in packet_type_list:
        for network_info in network_info_list:
            network_size = network_info["network_size"]
            network_type = network_info["network_type"]
            capacity_conditions = network_info["capacity_conditions"]
            for waiting_cost in waiting_cost_list:
                for capacity_condition in capacity_conditions:
                    for packet_size in packet_size_list:
                        for num_packets in num_packets_list:
                            memo = "trace " + packet_type + str(num_packets) + " " + network_type + " " + capacity_condition[
                                "capacity_type"] + " " + str(capacity_condition["capacity_coef"]) + "packet_size " + str(packet_size)
                            condition = {
                                "network_size": network_size, "num_packets": num_packets, "waiting_cost": waiting_cost,
                                "packet_type": packet_type,
                                "id_sort": False, "network_type": network_type,
                                "capacity_type": capacity_condition["capacity_type"],
                                "capacity_coef": capacity_condition["capacity_coef"],
                                "with_disturbance": False, "disturbance_type": "random",
                                "fixed_disturbance": [], "disturbance_length": 5, "disturbance_num": 2,
                                "generate_pattern": None, "is_traced": True, "packet_size": packet_size,
                                "generate_pattern_trace": None, "condition_memo": memo, "T_max": T_max
                            }

                            conditions.append(condition)

    print(len(conditions), "conditions executed with", core_num, "cores.")
    with Pool(core_num) as pool:
        pool.map(simulator_wrapper, conditions)

    print(time.time() - start)
