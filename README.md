# Graph-based Optimization Problem

This project implements a solution for a graph-based optimization problem using greedy heuristic (GH) and hill-climbing algorithms to partition nodes into clusters while maximizing internal connectivity. Developed as part of the CS361 Artificial Intelligence course at Imam Mohammad Ibn Saud Islamic University.

## Problem Description

The algorithm aims to:

- Partition graph nodes into P clusters
- Maximize the sum of edge weights within clusters
- Maintain cluster weights between specified lower (L) and upper (U) bounds
- Process data from text files (Sparse82.txt & RanReal480.txt)

## Features

- **Graph Representation**

  - Nodes with weights
  - Edges with weights
  - Cluster assignments
  - Weight constraints enforcement

- **Algorithms**
  - Greedy Heuristic (GH) for initial solution
  - Hill Climbing for solution optimization
  - Score calculation for cluster configurations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NAWR9/AI_optimization_project.git
cd AI_optimization_project
```

2. Ensure you have Python 3.x installed
3. Place your input files (if you have) in the `prob_instances` directory
4. Enusre it has the same structure of the other files described in `Instance_format.txt` file

## Usage

Run the main script:

```bash
python test.py
```

The program will:

1. Prompt you to choose an input file:
   - Sparse82.txt
   - RanReal480.txt
2. Execute the optimization algorithms
3. Display results including:
   - Initial solution and score
   - Optimized solution and score
   - Performance metrics

## Input File Format

The input files should follow this format:

```
M C L1 U1 L2 U2 ... LC UC W w1 w2 ... wM
node1 node2 edge_weight
...
```

Where:

- M: Number of nodes
- C: Number of clusters
- Li, Ui: Lower and upper bounds for cluster i
- W: Node weights identifier
- wi: Weight of node i
- Subsequent lines: Edge definitions with weights

## Algorithm Details

### Greedy Heuristic (GH)

1. Sorts nodes by weight (descending)
2. Assigns nodes to clusters while:
   - Maximizing edge weight connections
   - Maintaining weight constraints
3. Balances cluster weights through node redistribution

### Hill Climbing

1. Starts with GH solution
2. Iteratively improves through random node swaps while maintaining the constraints
3. Accepts improvements and occasional non-improving moves
4. Continues for specified iterations or until stagnation

## Performance Metrics

The implementation tracks:

- Best, average, and standard deviation of objective function
- Computation time for each algorithm
- Solution quality across multiple runs

## Results

For Sparse82.txt (8 clusters):

- Lower Bound: 25
- Upper Bound: 75
- Best Score Achieved: 842.11
- Average Computation Time: ~0.90 seconds

For RanReal480.txt (20 clusters):

- Lower Bound: 100
- Upper Bound: 150
- Best Score Achieved: 326010.23
- Average Computation Time: ~8.21 seconds

## Contributing

This is an academic project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## Authors

- Osamah Sadeq Shubaita
- Abdulaziz Mohammed Alkathiri
- Omar Khalid Alamoudi
- Sami Mohammed Ahmmad

## Acknowledgments

- Dr. Husein Perez - Course Instructor
- Imam Mohammad Ibn Saud Islamic University
- College of Computer and Information Sciences
