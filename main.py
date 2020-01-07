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

def CheckVertexCover(_temp_adj_matrix):

    # Not Covered Edges Indices
    return np.nonzero(_temp_adj_matrix)

def ModifyAdjMatrixForSample(_temp_adj_matrix, nodes_employed_idx, new_employed_idx=None):
    if (new_employed_idx is None):
        _temp_adj_matrix[nodes_employed_idx] = _temp_adj_matrix[:, nodes_employed_idx] = 0
    else:
        _temp_adj_matrix[new_employed_idx, :] = _temp_adj_matrix[:, new_employed_idx] = 0

def CheckRepair(generation):

    start_repair = end_repair = 0
    for cur_sample, sample_idx in zip(generation, range(0, len(generation))):
        start_repair = time.time()
        print(f"Repairing: {sample_idx}")
        _temp_adj_matrix = adj_matrix.copy()
        nodes_employed_idx = np.nonzero(cur_sample)
        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=nodes_employed_idx)
    
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        if(len(not_covered_edges_idx[0]) != 0):
            start_exec_repair = time.time()
            ExecuteRepair(cur_sample=cur_sample, _temp_adj_matrix=_temp_adj_matrix, not_covered_edges_idx=not_covered_edges_idx)
            end_exec_repair = time.time()
            print(f"Time for Execute Repair: {end_exec_repair - start_exec_repair }")

        end_repair=time.time()
        print(f"Total Time: {end_repair - start_repair}")

    return generation

def ExecuteRepair(cur_sample, _temp_adj_matrix, not_covered_edges_idx):
    is_repaired = False
    one_step_exec_arr = []
    while(is_repaired == False):  
        start_exec = time.time()
        new_employed_idx = np.random.choice(not_covered_edges_idx[:][0], size=np.random.randint(low=2, high=50))
        cur_sample[new_employed_idx] = 1

        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=None, new_employed_idx=new_employed_idx)
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        is_repaired = len(not_covered_edges_idx[0]) == 0
        end_exec = time.time()
        one_step_exec_arr.append(end_exec-start_exec)
    print(f"Mean Exec Cover: {np.mean(one_step_exec_arr)}")


generation = GetRandomGeneration(population_size=population_size)
generation = CheckRepair(generation=generation)


def TournementSelection():
    best = None
    for i in range(2):
        NodeID = random.randint(0, population_size - 1)
        if (best == None or weight_vec[NodeID] < weight_vec[best]):
            best = NodeID
    return best 