import numpy as np
import random
import sys
import pandas as pd
import time

input_file = sys.argv[1]
generation_num = int(sys.argv[2])
population_size = int(sys.argv[3])
crossover_prob = float(sys.argv[4])
mutation_prob = float(sys.argv[5])

print("Name of the graph file: ", input_file) 
print("Number of generations: ", generation_num) 
print("Population size: ", population_size) 
print("Crossover probability: ", crossover_prob) 
print("Mutation probability: ", mutation_prob) 
print()

def BuildAdjacencyMatrix(number_of_nodes, edges):
    adj_matrix = np.zeros(shape=(number_of_nodes, number_of_nodes))
    for pair in edges:
        source, dest = [int(x) for x in pair.split(" ")]
        adj_matrix[source, dest] = 1
    return adj_matrix

def GenerateWeightVector(node_weights):
    weight_vec = np.zeros(shape=(len(node_weights)))
    for pair in node_weights:
        node, weight = [x for x in pair.split(" ")]
        node = int(node)
        weight = float(weight)
        weight = weight if weight > 0 else 0.0001
        weight_vec[node] = weight
    return weight_vec

def GetRandomGeneration(population_size):
    return np.random.randint(2, size=(population_size,len(adj_matrix)))

def ReadInputFile(input_file):
    f = open(input_file,"r")
    f_lines = f.readlines()

    # Get number of nodes and number of edges from first two lines.
    number_of_nodes = int(f_lines[0])
    number_of_edges = int(f_lines[1].replace(".", ""))

    # Generate weight vector
    node_weights = f_lines[2:number_of_nodes+2]
    weight_vec = GenerateWeightVector(node_weights=node_weights)

    #Build 2D Adjacency matrix
    edges = f_lines[number_of_nodes+2:]
    adj_matrix = BuildAdjacencyMatrix(number_of_nodes=number_of_nodes, edges=edges)

    return number_of_nodes, number_of_edges, weight_vec, adj_matrix

num_of_nodes, num_of_edges, weight_vec, adj_matrix = ReadInputFile(input_file=input_file)

def FindFlipNodes(_temp_adj_matrix, not_covered_edges_idx):

    _temp_adj_triu_idx = np.tril_indices_from(_temp_adj_matrix)

    employable_edges_idx = np.argwhere(_temp_adj_matrix[_temp_adj_triu_idx] == 1)
    employable_nodes_idx = _temp_adj_triu_idx[0][employable_edges_idx]
    efficiency_matrix = pd.DataFrame(np.array(np.unique(employable_nodes_idx, return_counts=True)).T,
                        columns=["NodeID", "Efficiency"], dtype=object)
    efficiency_matrix["NodeID"] = efficiency_matrix["NodeID"].astype(np.int)
    efficiency_matrix["Efficiency"] = efficiency_matrix["Efficiency"]/weight_vec[efficiency_matrix["NodeID"]]

    return efficiency_matrix.sort_values(by=['Efficiency'], ascending=False)

def CheckVertexCover(_temp_adj_matrix):

    # Not Covered Edges Indices
    return np.argwhere(_temp_adj_matrix != 0)

def ModifyAdjMatrixForSample(_temp_adj_matrix, nodes_employed_idx):
    _temp_adj_matrix[nodes_employed_idx] = _temp_adj_matrix[:, nodes_employed_idx] = 0

def CheckRepair(generation):

    start_repair = end_repair = 0
    for cur_sample, sample_idx in zip(generation, range(0, len(generation))):
        start_repair = time.time()
        print(f"Repairing: {sample_idx}")
        _temp_adj_matrix = adj_matrix.copy()
        nodes_employed_idx = np.argwhere(cur_sample==1)
        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=nodes_employed_idx)
    
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        if(not_covered_edges_idx.size != 0):
            start_exec_repair = time.time()
            ExecuteRepair(cur_sample=cur_sample, _temp_adj_matrix=_temp_adj_matrix, not_covered_edges_idx=not_covered_edges_idx)
            end_exec_repair = time.time()
            print(f"Time for Execute Repair: {end_exec_repair - start_exec_repair }")

        end_repair=time.time()
        print(f"Total Time: {end_repair - start_repair}")

    return generation

def ExecuteRepair(cur_sample, _temp_adj_matrix, not_covered_edges_idx):
    is_repaired = False
    flip_time_arr = []
    while(is_repaired == False):  

        start_flip = time.time()
        flip_eff_mat = FindFlipNodes(_temp_adj_matrix, not_covered_edges_idx)
        end_flip = time.time()
        flip_time_arr.append(end_flip - start_flip)
        
        if(np.random.randn() > 0.8):
            rand_flip_idx = np.random.randint(len(flip_eff_mat))
            cur_sample[flip_eff_mat.iloc[rand_flip_idx]["NodeID"]] = 1
        else:
            cur_sample[flip_eff_mat.iloc[0].NodeID] = 1

        nodes_employed_idx = np.argwhere(cur_sample==1)
        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=nodes_employed_idx)
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        is_repaired = not_covered_edges_idx.size == 0
    print(f"Mean flip time: {np.mean(flip_time_arr)}, Total Flips = {len(flip_time_arr)}, Total Flip Time = {np.sum(flip_time_arr)}")
generation = GetRandomGeneration(population_size=population_size)
generation = CheckRepair(generation=generation)
pass

def TournementSelection():
    best = None
    for i in range(2):
        NodeID = random.randint(0, population_size - 1)
        if (best == None or weight_vec[NodeID] < weight_vec[best]):
            best = NodeID
    return best 