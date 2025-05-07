# Replicating Rapid Learning in Mice Using Reinforcement Learning

This was a group project as part of the MSc in Computational Neuroscience at the University of Nottingham. It aims to replicate the findings of Rosenberg et al. (2021) on rapid learning and spatial navigation in mice. The original study observed that mice navigating a labyrinth exhibited sudden insight and discontinuous learning, rapidly improving their navigation efficiency after initial exploration.

## Project Goal

To model and replicate the discontinuous learning behaviour observed in mice using Reinforcement Learning (RL), particularly focusing on how mice develop cognitive maps of their environment.

## Key Concepts

* **Spatial Navigation:** Simulating how mice learn optimal paths in a labyrinth, transitioning from random exploration to efficient route selection.
* **Discontinuous Learning:** Modeling the sudden improvement in path efficiency once the optimal route is discovered.
* **Reinforcement Learning Algorithms:**

  * **Q-Learning:** Captures the decision-making process through state-action pair optimization.
  * **Action Selection Strategies:**

    * **ϵ-greedy:** Balances exploration and exploitation.
    * **Softmax:** Provides a probabilistic approach, favoring higher reward actions.

## Results

The RL model successfully demonstrated rapid learning and spatial insight akin to the original animal study:

* After initial random exploration, the agent exhibited a sudden improvement in navigation, mirroring cognitive map formation.
* The Softmax strategy resulted in more consistent learning compared to the ϵ-greedy strategy.

## Code Structure

```
📁 Rosenberg-2021-Replication
├── README.md                  # Project documentation
├── 📁 Code
│   ├── RLDiscontinuousGraph.py # Q-Learning with graph-based maze representation
│   ├── RLexploration_exploitation.py # RL with matrix-based maze representation
│   ├── RLHeatmaps.py # Heatmap generation for visualising agent trajectories
│   └── PriorProbabilityModel.ipynb # Initial probability model for RL agent
```

## Usage

Train the RL agent using the graph-based maze:

```bash
python RLDiscontinuousGraph.py
```

Generate heatmaps to visualise spatial learning:

```bash
python RLHeatmaps.py
```

## Relevance to Spatial Navigation Research

This project provides insights into how spatial navigation strategies evolve through reinforcement learning, reflecting cognitive processes similar to those in biological systems. The model’s ability to capture discontinuous learning in spatial tasks aligns with research on neural mechanisms of navigation and decision-making.

## Contributors

* Jinhao Fang
* Salaar Mir
* Gautam Sathe
* Zakaria Taghi
