import numpy as np
import sys

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

def FindFlipNodes(cur_sample, covered_nodes):
    #Create the temporary vector depending on the nodes/weight
    num_not_included_nodes = len(cur_sample) - cur_sample.sum()
    efficiency_matrix = np.concatenate([np.argwhere(cur_sample == 0), np.zeros(shape=(num_not_included_nodes, 1))], axis=1)
    for cur_node, idx in zip(efficiency_matrix, range(0, len(efficiency_matrix))):
        cur_node_idx = int(cur_node[0])
        cur_neighbours = CheckNeighbours(cur_node_idx)
        unexplored_neighbours = len(cur_neighbours) - np.in1d(covered_nodes, cur_neighbours).sum() 
        node_efficiency = unexplored_neighbours / weight_vec[cur_node_idx] 
        efficiency_matrix[idx, 1] = node_efficiency
    return sorted(efficiency_matrix, key=lambda a_entry: a_entry[1], reverse=True)

def CheckNeighbours(node_idx):
    node_idx = int(node_idx)
    cur_neighbours = np.argwhere(adj_matrix[node_idx] == 1)
    return cur_neighbours

def CheckVertexCover(nodes_employed_idx):
    covered_nodes = set()
    for node_idx in nodes_employed_idx:
        cur_neighbours = CheckNeighbours(node_idx)
        covered_nodes.update(cur_neighbours.flatten())
    if(len(covered_nodes) != len(adj_matrix)):
        return covered_nodes
    return None

def CheckRepair(generation):

    for cur_sample, sample_idx in zip(generation, range(0, len(generation))):
        nodes_employed_idx = np.argwhere(cur_sample==1)
        covered_nodes = CheckVertexCover(nodes_employed_idx=nodes_employed_idx)
        if(covered_nodes is not None):
            cur_sample = ExecuteRepair(cur_sample=cur_sample, covered_nodes=covered_nodes)
    return generation

def ExecuteRepair(cur_sample, covered_nodes):
    is_repaired = False
    while(is_repaired == False):         
        flip_eff_mat = FindFlipNodes(cur_sample, covered_nodes)
        if(np.random.randn() > 0.8):
            rand_flip_idx = np.random.randint(len(flip_eff_mat))
            cur_sample[int(flip_eff_mat[rand_flip_idx][0])] = 1
        else:
            cur_sample[int(flip_eff_mat[0][0])] = 1

        nodes_employed_idx = np.argwhere(cur_sample==1)
        #WE don't have to check every employed node, we only need to check the last added node's neighbours.
        covered_nodes = CheckVertexCover(nodes_employed_idx=nodes_employed_idx)
        is_repaired = covered_nodes is None
    return cur_sample
    
generation = GetRandomGeneration(population_size=population_size)
generation[0] = np.zeros(shape=(len(generation[0])))
generation = CheckRepair(generation=generation)
while(len(broken_samples) > 0):
    broken_sample = ExecuteRepair(broken_samples)


ExecuteRepair(generation=generation)