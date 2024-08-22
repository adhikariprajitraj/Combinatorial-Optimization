# Network Flow Optimization

## Overview
This project focuses on solving various network flow problems using different optimization algorithms. Network flow problems are a central theme in operations research, where the goal is to optimize the movement or flow through a network in a way that minimizes or maximizes some quantity (like cost or flow amount).

## Algorithms

### Ford-Fulkerson Method
The Ford-Fulkerson algorithm is utilized to find the maximum flow in a network. It iteratively searches for augmenting paths with available capacity in the residual graph and augments the flow until no more augmenting paths can be found.

#### Mathematical Representation
The flow value `f` in the network must satisfy the following constraints:
- Capacity Constraint: $f(u, v) \leq c(u, v)$
- Flow Conservation: $$\displaystyle{\sum_{v \in V}} f(u, v) = 0 \),\forall u \neq s, t$$


### Edmonds-Karp Algorithm
An implementation of the Ford-Fulkerson method that uses BFS to find the shortest path in terms of the number of edges. The use of BFS ensures that the shortest path is found, preventing the creation of paths that might lead to local optima.

### Dinic's Algorithm
This algorithm is effective for computing the maximum flow in a network using a level graph, which is constructed using BFS. The flow is then increased by finding blocking flows in the level graph using DFS.

### Connection with Operations Research
Operations research focuses on optimizing complex operations and systems, which is directly applicable to network flow problems where decisions about routing and resource allocation must be optimized to improve system efficiency.

### Applications
- Traffic management
- Telecommunications
- Supply chain logistics
- Resource allocation in cloud computing


- **Python**: Primary programming language used for developing algorithms.
- **NetworkX**: Utilized for managing and operating on graph data structures.
- **Matplotlib**: For visualizing the network and flows within it.