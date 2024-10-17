import random
from Reads import read_file
import math
import statistics
import time

class GraphPartitioning:
    def __init__(self, nodes, edges, L, U, p):
        self.nodes = nodes  # Nodes with weights
        self.edges = edges  # Edges with weights, assumes (i, j): weight for all i, j pairs in a complete graph
        self.L = L          # Lower bound on cluster weights
        self.U = U          # Upper bound on cluster weights
        self.p = p          # Number of clusters
        self.clusters = [[] for _ in range(p)]  # Cluster assignments

    def greedy_heuristic(self):
        """Improved Greedy Heuristic with consideration for node connectivity and edge weights."""
        available_nodes = set(self.nodes.keys()) # Set of nodes that have not been assigned to a cluster
        sorted_nodes = sorted(available_nodes, key=lambda n: self.nodes[n], reverse=True)  # Sort by node weights (heaviest first)
        
        cluster_index = 0 # Start with the first cluster
        cluster_weights = [0] * self.p  # Initialize the weight of each cluster as 0

        # Assign the first node to the first cluster
        first_node = sorted_nodes.pop(0) # Remove the first node from the list
        self.clusters[cluster_index].append(first_node) # Assign the first node to the first cluster
        cluster_weights[cluster_index] += self.nodes[first_node] # Update the weight of the first cluster
        
        # Start with the next cluster
        cluster_index = (cluster_index + 1) % self.p

        # Assign the rest of the nodes
        for node in sorted_nodes:
            best_cluster = None # Track the best cluster for the current node
            best_cluster_score = -float('inf') # Initialize the best cluster score as negative infinity

            # Try each cluster and calculate the potential score improvement
            for i in range(self.p):
                if cluster_weights[i] + self.nodes[node] <= self.U: # Check if the node can be added to the cluster without exceeding the upper bound
                    potential_score = 0
                    # Calculate potential score based on edge weights to the cluster's nodes
                    for other_node in self.clusters[i]:
                        edge_weight = self.edges.get((int(node), int(other_node)), 0) # Get the edge weight from the graph (or 0 if missing)
                        potential_score += edge_weight

                    # Choose the cluster that maximizes the edge weight (i.e., node connectivity)
                    if potential_score > best_cluster_score:
                        best_cluster_score = potential_score
                        best_cluster = i
            
            # If no valid cluster was found, force the node into the cluster with the lowest total weight
            if best_cluster is None:
                best_cluster = min(range(self.p), key=lambda i: cluster_weights[i])

            # Add the node to the chosen cluster
            self.clusters[best_cluster].append(node)
            cluster_weights[best_cluster] += self.nodes[node]

        # After initial assignment, redistribute nodes from clusters that have excess weight
        underweight_clusters = [i for i, weight in enumerate(cluster_weights) if weight < self.L]

        for uw_cluster in underweight_clusters:
            while cluster_weights[uw_cluster] < self.L:
                # Find a cluster with more than the lower bound (but not necessarily exceeding U)
                donor_clusters = [i for i, weight in enumerate(cluster_weights) if weight > self.L]

                if not donor_clusters:
                    print(f"Could not satisfy the lower bound for cluster {uw_cluster + 1}. No valid donor clusters.")
                    break

                transfer_happened = False  # Track whether a valid transfer occurred

                for donor_cluster in donor_clusters:
                    # Try to find a node to transfer from the donor cluster
                    for node in self.clusters[donor_cluster]:
                        potential_new_weight_uw = cluster_weights[uw_cluster] + self.nodes[node]
                        potential_new_weight_donor = cluster_weights[donor_cluster] - self.nodes[node]

                        # Ensure the donor cluster doesn't fall below the lower bound after the transfer
                        if potential_new_weight_donor >= self.L and potential_new_weight_uw <= self.U:
                            # Move the node from donor to underweight cluster
                            self.clusters[donor_cluster].remove(node)
                            self.clusters[uw_cluster].append(node)
                            cluster_weights[donor_cluster] -= self.nodes[node]
                            cluster_weights[uw_cluster] += self.nodes[node]
                            transfer_happened = True
                            break  # Move to the next underweight cluster

                    # Stop searching this donor cluster if a valid transfer was made
                    if transfer_happened:
                        break

                # If no valid transfer can be made, break the loop to avoid infinite looping
                if not transfer_happened:
                    print(f"Unable to balance cluster {uw_cluster + 1}. No more valid transfers.")
                    break

        # Debugging output: Check the cluster weights and constraints
        for i, cluster in enumerate(self.clusters):
            cluster_weight = sum(self.nodes[node] for node in cluster)
            print(f"Cluster {i+1}: Nodes = {cluster}, Total Weight = {cluster_weight}")
            if self.L <= cluster_weight <= self.U:
                print(f"Cluster {i+1} satisfies the constraints: {self.L} <= {cluster_weight} <= {self.U}")
            else:
                print(f"Cluster {i+1} does NOT satisfy the constraints: {self.L} <= {cluster_weight} <= {self.U}")

        return self.clusters

    def calculate_cluster_score(self, clusters):
        """Calculate the score of a given cluster configuration for a complete graph, with debugging output."""
        total_score = 0
        for cluster in clusters:
            # Debugging: Print the current cluster being processed
            # print(f"Processing Cluster: {cluster}")

            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):  # Only count pairs once
                    node1, node2 = cluster[i], cluster[j]
                    
                    # Get the edge weight from the graph (or 0 if missing)
                    edge_weight = self.edges.get((int(node1), int(node2)), 0)
                    
                    # Debugging: Print the edge being processed and its weight
                    # print(f"Nodes ({node1}, {node2}): Edge Weight in Data = {edge_weight}")
                    
                    # Add the edge weight to the total score
                    total_score += edge_weight

        # Debugging: Print the final score for this configuration
        # print(f"Total Cluster Score: {total_score}")
        return total_score

    def hill_climbing(self, iterations=2000, allow_non_improving_moves=False):
        """Improved Hill Climbing function to maximize the score."""
        best_clusters = [cluster[:] for cluster in self.clusters]  # Deep copy of initial clusters
        best_score = self.calculate_cluster_score(best_clusters) # Initial score for the best clusters

        stagnation_counter = 0  # Counter for tracking stagnation (no improvement)
        max_stagnation = 200  # Number of iterations before re-evaluating

        for _ in range(iterations):
            new_clusters = [cluster[:] for cluster in best_clusters]  # Deep copy for this iteration

            # Randomly select two clusters by index
            cluster_indices = random.sample(range(len(new_clusters)), 2)
            cluster_a_index = cluster_indices[0]
            cluster_b_index = cluster_indices[1]

            cluster_a = new_clusters[cluster_a_index]
            cluster_b = new_clusters[cluster_b_index]

            if cluster_a and cluster_b:
                # Randomly select one node from each cluster to swap
                node_a = random.choice(cluster_a)
                node_b = random.choice(cluster_b)

                # Calculate new weights after the proposed swap
                new_weight_a = sum(self.nodes[node] for node in cluster_a) - self.nodes[node_a] + self.nodes[node_b]
                new_weight_b = sum(self.nodes[node] for node in cluster_b) - self.nodes[node_b] + self.nodes[node_a]

                # Check if the move is valid based on the constraints
                if self.L <= new_weight_a <= self.U and self.L <= new_weight_b <= self.U:
                    # Perform the swap
                    cluster_a.remove(node_a)
                    cluster_b.remove(node_b)
                    cluster_a.append(node_b)
                    cluster_b.append(node_a)

                    # Calculate new score
                    new_score = self.calculate_cluster_score(new_clusters)

                    # Update best score and clusters if new score is better
                    if new_score > best_score:
                        best_clusters = [cluster[:] for cluster in new_clusters]  # Update best clusters
                        best_score = new_score
                        stagnation_counter = 0  # Reset stagnation counter
                    else:
                        stagnation_counter += 1  # Increment stagnation counter

                    # If we reach a certain number of iterations without improvement
                    if stagnation_counter >= max_stagnation:
                        # Diversify clusters slightly by shuffling
                        for cluster in new_clusters:
                            random.shuffle(cluster)  # Shuffle nodes in the clusters
                        stagnation_counter = 0  # Reset stagnation counter after diversification

                # Allow for occasional non-improving moves to escape local optima
                if allow_non_improving_moves and random.random() < 0.1:
                    new_clusters = [cluster[:] for cluster in best_clusters]  # Deep copy to shuffle
                    random.shuffle(new_clusters)  # Shuffle clusters for a different arrangement
                    new_score = self.calculate_cluster_score(new_clusters)
                    if new_score > best_score:
                        best_clusters = new_clusters
                        best_score = new_score

        return best_clusters, best_score
    
s = '\\'
try:
    # Prompt the user to choose a file
    print("Choose a file to read:")
    print("1. Sparse82.txt")
    print("2. RanReal480.txt")
    choice = input("Enter the number of your choice: ")

    # Set the file path based on the user's choice
    # loop if choice is not 1 or 2
    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Enter the number of your choice: ")
    if choice == '1':
        file_path = r'.\prob_instances\Sparse82.txt'
    elif choice == '2':
        file_path = r'.\prob_instances\RanReal480.txt'
            
    print(f"Reading file: {file_path.split(s)[-1]}")

    M, C, bounds, node_weights, edges = read_file(file_path)


    # Define lower bound L, upper bound U, and number of clusters p
    L = bounds[0][0]
    U = bounds[0][1]
    p = C

    better_scores_of_HC = []
    computation_time_HC = []
    # Initialize the graph partitioning problem
    graph_partitioning = GraphPartitioning(node_weights, edges, L, U, p)

    # Step 1: Generate an initial solution using Greedy Heuristic (GH)
    initial_solution = graph_partitioning.greedy_heuristic()
    print(f"Initial Solution: {initial_solution}")
    initial_score = graph_partitioning.calculate_cluster_score(initial_solution)
    print(f"Initial Score: {round(initial_score, 2)}")

    for i in range(10):
        # Get the GraphPartitioning instance with only the initial solution
        graph_partitioning_initial = graph_partitioning

        start_time = time.time()
        # Step 2: Apply Hill Climbing to improve the solution, allowing occasional non-improving moves
        better_solution, better_score = graph_partitioning_initial.hill_climbing(iterations=10000, allow_non_improving_moves=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        computation_time_HC.append(round(elapsed_time,4))
        print(f'Time = {round(elapsed_time, 4)}')
        print(f"better Solution after Hill Climbing: {better_solution[:2]}...")
        print(f"better Score: {round(better_score, 2)}")
        better_scores_of_HC.append(round(better_score,2))



    # Calculate statistics for better scores and computation time
    max_score = max(better_scores_of_HC)
    average_score = sum(better_scores_of_HC) / len(better_scores_of_HC)
    std_dev_score = statistics.stdev(better_scores_of_HC)

    average_time = sum(computation_time_HC) / len(computation_time_HC)

    print()
    print(f"Initial Solution: {initial_solution}")
    print(f"Initial Score: {initial_score}")
    # ===========================================================

    print('==================')
    print(f'better_scores_of_HC = {better_scores_of_HC}')
    print(f'MAX better_scores_of_HC = {max_score}')
    print(f'AVG of better_scores_of_HC = {round(average_score,2)}')
    print(f'Standard Deviation of better_scores_of_HC = {round(std_dev_score,2)}')
    print(f'Average Computation Time = {round(average_time, 4)} seconds')
    print(f'The time computations for each iteration {computation_time_HC}')

except KeyboardInterrupt:
    print("Keyboard Interrupt. Exiting")