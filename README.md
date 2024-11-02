# Network Flow and Energy Optimization

## Overview
This project focuses on solving various optimization problems including network flow and energy arbitrage. It demonstrates the application of different optimization algorithms to real-world scenarios in operations research.

## Network Flow Optimization
Network flow problems are a central theme in operations research, where the goal is to optimize the movement or flow through a network in a way that minimizes or maximizes some quantity (like cost or flow amount).

### Network Algorithms
- **Ford-Fulkerson Method**: Iteratively searches for augmenting paths to find maximum flow
- **Edmonds-Karp Algorithm**: Implements Ford-Fulkerson using BFS for shortest paths
- **Dinic's Algorithm**: Uses level graphs and blocking flows for efficient maximum flow computation

### Mathematical Representation
The flow value `f` in the network must satisfy:
- Capacity Constraint: $f(u, v) \leq c(u, v)$
- Flow Conservation: $$\displaystyle{\sum_{v \in V}} f(u, v) = 0 \),\forall u \neq s, t$$

## Energy Arbitrage Optimization
The project includes an energy arbitrage optimization model that maximizes profit from battery storage operations while considering various constraints and costs.

### Key Features
- Battery charge/discharge optimization
- Time-of-use electricity pricing
- Battery degradation costs
- Round-trip efficiency considerations
- State of charge management

### Optimization Results
The model achieves significant profit optimization while maintaining battery health:
- Net Profit: $53.35
- Energy Charged: 174.3 kWh
- Energy Discharged: 179.5 kWh
- Round-trip Efficiency: 103.0%
- Equivalent Cycles: 1.80

![Energy Optimization Results](optimization_results.png)

## Technologies Used
- **Python**: Primary programming language
- **NetworkX**: Graph operations
- **Matplotlib**: Visualization
- **Pyomo**: Mathematical optimization
- **GLPK**: Linear programming solver

## Applications
- Traffic management
- Telecommunications
- Supply chain logistics
- Energy storage systems
- Resource allocation in cloud computing