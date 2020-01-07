import numpy as np
import time
import sys
import pandas as pd
import random


input_file = sys.argv[1]
generation_num = int(sys.argv[2])
population_size = int(sys.argv[3])
crossover_prob = float(sys.argv[4])
mutation_prob = sys.argv[5]

orig_stdout = sys.stdout
f = open(f"out_{input_file}_{generation_num}_{population_size}_{crossover_prob}_{mutation_prob.replace('/', '_over_')}.txt", "w+")
sys.stdout = f

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
        weight = weight if weight > 0 else 0.00001
        weight_vec[node] = weight
    return weight_vec

def GetRandomGeneration(population_size):
    return np.random.randint(2, size=(population_size,len(adj_matrix)))

def ReadInputFile(input_file):
    global mutation_prob
    f = open(input_file,"r")
    f_lines = f.readlines()
    # Get number of nodes and number of edges from first two lines.
    number_of_nodes = int(f_lines[0])
    number_of_edges = int(f_lines[1].replace(".", ""))

    # Update Mutation Probability
    mutation_prob = 1/number_of_nodes if mutation_prob == "1/n" else float(mutation_prob)

    # Generate weight vector
    node_weights = f_lines[2:number_of_nodes+2]
    weight_vec = GenerateWeightVector(node_weights=node_weights)

    #Build 2D Adjacency matrix
    edges = f_lines[number_of_nodes+2:]
    adj_matrix = BuildAdjacencyMatrix(number_of_nodes=number_of_nodes, edges=edges)
    eff_mat = pd.DataFrame(np.array(np.unique(np.nonzero(adj_matrix)[0], return_counts=True)).T, columns=["NodeID", "Efficiency"], dtype=object)
    eff_mat["NodeID"] = eff_mat["NodeID"].astype(np.int)
    eff_mat["Efficiency"] = (eff_mat["Efficiency"]/eff_mat["Efficiency"].sum())/((weight_vec[eff_mat["NodeID"]]/weight_vec.sum())*2)
    return number_of_nodes, number_of_edges, weight_vec, adj_matrix, eff_mat.sort_values(by="Efficiency", ascending=False)

num_of_nodes, num_of_edges, weight_vec, adj_matrix, eff_mat = ReadInputFile(input_file=input_file)

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
        _temp_adj_matrix = adj_matrix.copy()
        nodes_employed_idx = np.nonzero(cur_sample)
        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=nodes_employed_idx)
    
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        if(len(not_covered_edges_idx[0]) != 0):
            ExecuteRepair(cur_sample=cur_sample, _temp_adj_matrix=_temp_adj_matrix, not_covered_edges_idx=not_covered_edges_idx)
    
    weights_for_each_sample = np.apply_along_axis(lambda x: weight_vec[np.nonzero(x)].sum(), 1, generation)
    sample_sort_idx = np.argsort(weights_for_each_sample)[::-1]
    generation = generation[sample_sort_idx]
    return generation

def ExecuteRepair(cur_sample, _temp_adj_matrix, not_covered_edges_idx):
    is_repaired = False
    _temp_eff_mat = eff_mat.copy()
    while(is_repaired == False):  

        # _temp_eff_mat = _temp_eff_mat[_temp_eff_mat.NodeID.isin(np.unique(not_covered_edges_idx[:][0]))]

        # num_employ_node = np.random.randint(low=2, high=20)
        # # new_employed_idx = _temp_eff_mat[:num_employ_node]["NodeID"].values
        
        new_employed_idx = np.random.choice(not_covered_edges_idx[:][0], size=np.random.randint(low=2, high=4))
        cur_sample[new_employed_idx] = 1

        ModifyAdjMatrixForSample(_temp_adj_matrix=_temp_adj_matrix, nodes_employed_idx=None, new_employed_idx=new_employed_idx)
        not_covered_edges_idx = CheckVertexCover(_temp_adj_matrix=_temp_adj_matrix)
        is_repaired = len(not_covered_edges_idx[0]) == 0
    

def TournementSelectMatingPool(generation):

    random_parents_idx = np.empty((0,2), int)
    for ind in range(4):
        rand_num = np.random.rand()
        if(rand_num> 0.6):
            random_parents_idx = np.append(random_parents_idx, np.random.randint(low=0, high=int(population_size*0.25), size=(int(population_size*0.25), 2)), axis=0)
        elif(rand_num > 0.3):
            random_parents_idx = np.append(random_parents_idx,np.random.randint(low=int(population_size*0.25), high=int(population_size*0.5), size=(int(population_size*0.25), 2)), axis=0)
        elif(rand_num > 0.11):
            random_parents_idx = np.append(random_parents_idx,np.random.randint(low=int(population_size*0.5), high=int(population_size*0.75), size=(int(population_size*0.25), 2)), axis=0)
        else:
            sec_rand_num = np.random.rand()
            if(sec_rand_num > 0.8):
                random_parents_idx = np.append(random_parents_idx, np.random.randint(low=0, high=int(population_size*0.4), size=(int(population_size*0.25), 2)), axis=0)
            elif(sec_rand_num > 0.4):
                random_parents_idx = np.append(random_parents_idx,np.random.randint(low=int(population_size*0.4), high=int(population_size*0.8), size=(int(population_size*0.25), 2)), axis=0)
            else:
                random_parents_idx = np.append(random_parents_idx,np.random.randint(low=int(population_size*0.8), high=int(population_size), size=(int(population_size*0.25), 2)), axis=0)

    is_first_col_costlier = weight_vec[random_parents_idx[:, 0]] > weight_vec[random_parents_idx[:, 1]]
    selected_idx = [row[1] if is_first_col_costlier[ind] else row[0] for ind, row in zip(range(population_size),random_parents_idx)]
    return selected_idx

def ApplyCrossover(generation):
    random_crossover_pairs = np.random.choice(population_size, size=(population_size, 2))

    for pair in random_crossover_pairs:
        if(np.random.rand() < crossover_prob):
            crossover_point = np.random.randint(1, num_of_nodes - 1)
            #! Optimization issue...
            generation[pair[0]][crossover_point:], generation[pair[1]][crossover_point:] = generation[pair[1]][crossover_point:], generation[pair[0]][crossover_point:].copy()

def ApplyMutation(generation):
    tf = np.array([True, False])
    tf_mask = np.random.choice(tf, size=(population_size, len(generation[0])), p=[mutation_prob, 1-mutation_prob])
    generation[:][tf_mask] = np.logical_not(generation[:][tf_mask]).astype(np.int)

def MeanCost(generation):
    return np.sum(weight_vec[np.nonzero(generation)[1]])/len(generation)

generation = GetRandomGeneration(population_size=population_size)
# MeanCost(generation)
first_repair_start=time.time()
CheckRepair(generation=generation)
first_repair_end=time.time()
print(f"First Repair Time: {first_repair_end - first_repair_start}")
print()
for i in range(generation_num):  
    generation_start = time.time()
    print(f"{i}th Generation:")
    tournement_start = time.time()
    generation = generation[TournementSelectMatingPool(generation=generation)]
    tournement_end = time.time()

    crossover_start = time.time()
    ApplyCrossover(generation=generation)
    crossover_end = time.time()

    mutation_start = time.time()
    ApplyMutation(generation=generation)
    mutation_end = time.time()

    second_repair_start = time.time()
    CheckRepair(generation=generation)
    second_repair_end = time.time()

    generation_end = time.time()

    print(f"\tTournement Time: {tournement_end - tournement_start}")
    print(f"\tCrossover Time: {crossover_end - crossover_start}")
    print(f"\tMutation Time: {mutation_end - mutation_start}")
    print(f"\tSecond Repair Time: {second_repair_end - second_repair_start}")
    print(f"\tTotal Generation Time: {generation_end - generation_start}")
    print(f"\tMean(Between Samples) Cost Time: {MeanCost(generation=generation)}")
sys.stdout.close()