# Function to read the Sparse82 file and extract data
def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines into a list

    # First line contains metadata
    first_line = lines[0].strip().split()

    # Extract M and C
    M = int(first_line[0])  # Number of elements
    C = int(first_line[1])  # Number of clusters

    # Extract cluster limits (lower and upper bounds)
    bounds = []
    for i in range(2, 2 + 2 * C, 2):
        lower_bound = int(first_line[i])
        upper_bound = int(first_line[i + 1])
        bounds.append((lower_bound, upper_bound))  # Append as a tuple

    # Extract node weights (after 'W')
    index_of_w = first_line.index('W')
    node_weights = {str(i): int(w) for i, w in enumerate(first_line[index_of_w + 1:])}  # Save as a dictionary

    # Store edges in a dictionary
    edges = {}
    for line in lines[1:]:
        elementA, elementB, edge_weight = map(float, line.strip().split())
        edges[(int(elementA), int(elementB))] = edge_weight  # Store as a tuple key

    return M, C, bounds, node_weights, edges

# Main function
def main():
    # Path to the Sparse82 file
    file_path = r'.\prob_instances\Sparse82.txt'

    # Read data from the Sparse82 file
    M, C, bounds, node_weights, edges = read_file(file_path)

    # Print the extracted values
    print(f"Number of elements (M): {M}")
    print(f"Number of clusters (C): {C}")
    print("Cluster bounds (Lower, Upper):")
    for i, (lower, upper) in enumerate(bounds):
        print(f"Cluster {i + 1}: Lower = {lower}, Upper = {upper}")

    print("Node weights:")
    for node, weight in node_weights.items():
        print(f"Node {node}: Weight {weight}")

    print("Edges (as dictionary):")
    for edge, weight in edges.items():
        print(f"Edge {edge} has weight {weight}")

# Run the main function
if __name__ == "__main__":
    main()
