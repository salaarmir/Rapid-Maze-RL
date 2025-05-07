import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import random
import matplotlib.patches as mpatches

# Initialise starting and terminal node for maze graph structure -> this depends on the maze structure set in createNetworkGraph() function below
STARTING_NODE = 1
TERMINAL_NODE = 127
RANDOM_START_NODE = random.randint(1, TERMINAL_NODE)

# Set number of episodes to train
TOTAL_EPISODES = 100
CONTROL_NODES = [72, 114, 93]

# Initialise the optimal number of nodes travarsed to reach the reward
OPTIMAL_PATH = 6


class Environment(object):
    def __init__(self, graphMaze, rat=STARTING_NODE):

        # Initialise maze and reset rat
        self._maze = graphMaze
        self.target = TERMINAL_NODE
        self.reset(rat)

    def reset(self, rat):

        # Reset rat properties
        self.rat = rat
        self.state = (rat)
        self.total_reward = 0
        self.visited = []
        self.visited.append(rat)

        return self.state

    def act(self, state, action):  # Apply the action, update state, get reward

        # Update state
        next_state = self.update_state(state, action)
        self.state = next_state

        # Update list of visited nodes
        self.visited.append(next_state)

        # Return reward and game status
        reward = self.get_reward()
        status = self.game_status()

        return (reward, self.state, status)

    def update_state(self, state, action):

        # Check if action is valid
        if action in self.valid_actions(self._maze, state):
            return action
        else:
            print("invalid move")
            return state

    def valid_actions(self, graphMaze, state):

        # Return list of valid actions
        return list(graphMaze.neighbors(state))

    def get_reward(self):  # get reward after action, can change the value of reward

        node = self.state

        # Check if rat is at reward
        if node == TERMINAL_NODE:  #  Water port position
            return 1.0
            # return 5.0

        # Reduce repeat visits
        if (node) in self.visited:
            return -0.1
        else:
            return -1.0

    def game_status(self):

        node = self.state

        if node == TERMINAL_NODE:  # check if rat is in terminal state
            return True
        else:
            return False


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=1, qtable_path=False):

        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Check if a q-table path has been provided otherwise initialise a new empty q-table
        if qtable_path == False:
            self.q_table = {}
        else:
            with open(qtable_path, 'rb') as f:
                loaded_Q_table = pickle.load(f)

            print("Loaded Q table:", loaded_Q_table)
            new_q_table = {}
            for state_str, actions in loaded_Q_table.items():
                state_tuple = eval(state_str)
                new_q_table[state_tuple] = actions
            self.q_table = new_q_table

    def choose_action_egreedy(self, state, epsilon=0.1):  # epsilon-greedy

        self.epsilon = epsilon

        # Check whether to explore
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.valid_actions(self.env._maze, state))

        # Check and update q-table
        else:
            if state not in self.q_table:
                return np.random.choice(self.env.valid_actions(self.env._maze, state))
            else:
                # print('got max value')
                return max(self.q_table[state], key=self.q_table[state].get)

    def choose_action_softmax(self, state, temperature=1):  # softmax

        self.temperature = temperature
        state_str = state

        if state_str not in self.q_table:
            return np.random.choice(self.env.valid_actions(self.env._maze, state))

        # Return action based on temperature
        q_values = np.array(
            [self.q_table[state_str].get(action, 0) for action in self.env.valid_actions(self.env._maze, state)])
        softmax_probs = np.exp((q_values - np.max(q_values)) / self.temperature) / np.sum(
            np.exp((q_values - np.max(q_values)) / self.temperature))
        action = np.random.choice(self.env.valid_actions(self.env._maze, state), p=softmax_probs)

        return action

    """
    update value of the action the current state in the qtable.
    if the action or state isn't in the table, it won't update the value, instead,
    it will add the state or action to the qtable
    """

    def update_q_table(self, state, action, reward, next_state):

        state_str = state

        # Check if state is in the q-table
        if state_str not in self.q_table:
            self.q_table[state_str] = {action: reward}
        else:
            if action not in self.q_table[state_str]:
                self.q_table[state_str][action] = reward
            else:
                next_state_str = next_state
                next_max = max(self.q_table[next_state_str].values()) if next_state_str in self.q_table else 0
                new_value = self.q_table[state_str][action] + self.learning_rate * (
                            reward + self.discount_factor * next_max - self.q_table[state_str][action])
                self.q_table[state_str][action] = new_value

    # Main RL loop
    def train(self, total_episodes):

        #  Initialise arrays to store count (number of nodes visited to reach reward), episodes and rewards
        paths = []
        episodes = []
        rewards = []
        optimalpaths = []
        controlpaths = []
        rewardpaths = []

        # Number of times rat took optimalpath
        optimalpathcount = 0

        # Number of times rat reached one of the control nodes
        control_count = 0

        # Number of times rat reached reward unoptimally
        unoptimalrewardcount = 0

        # Average path length of rat
        averagepaths = []
        averagepath = 0
        for episode in range(1, total_episodes + 1):

            # Reinitialise environment
            env.__init__(self.env._maze)
            state = self.env.reset(STARTING_NODE)

            # Append episodes
            episodes.append(episode)

            # Reitinitalise episode variables
            done = False
            total_reward = 0
            path = 0

            while not done:
                action = self.choose_action_softmax(state, temperature=0.1)

                # Perform action on current state
                reward, next_state, done = self.env.act(state, action)

                # Update q-table
                self.update_q_table(state, action, reward, next_state)

                # Update state and episode variables
                total_reward += reward
                state = next_state
                path += 1
                averagepath += 1

                # Check if rat has visited the control nodes in the maze
                if state in CONTROL_NODES:
                    control_count += 1


            # Check if optimal path was taken 
            if path <= OPTIMAL_PATH:
                optimalpathcount += 1
            else:
                unoptimalrewardcount += 1

            # Cumulative count of optimal paths as episodes increase
            optimalpaths.append(optimalpathcount)

            # Cumulative count of unoptimal reward paths
            rewardpaths.append(unoptimalrewardcount)

            # Cumulative count of number of times the rat visited the control nodes
            controlpaths.append(control_count/3)

            # Append path and reward arrays
            paths.append(path)
            rewards.append(total_reward)

            # Calculate average path length after each episode
            averagepaths.append(averagepath / episode)
            print(averagepaths)

            # Print Episode information and path taken
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Done:{done}, Path Count:{path}")
            print("Path taken: ", self.env.visited)

            # Show visual of path taken for each episode 
            # show(self.env,episode, path, paths)

        perfectpaths = [num / total_episodes for num in optimalpaths]

        # Plot figure to show number of times optimal path taken as episodes increased
        plt.figure()
        plt.plot(episodes, optimalpaths, color='red', label='Optimal Path to Water')
        plt.plot(episodes, controlpaths, color='blue', label="Control Node Visited")
        plt.plot(episodes, rewardpaths, color='green', label='Nonoptimal Path to Water')
        plt.legend()
        plt.xlabel("Episode Number")
        plt.ylabel("Optimal Path Count")
        plt.show()

        fig, ax1 = plt.subplots()

        # Plot the "Fraction Perfect" data
        color = 'tab:orange'
        ax1.set_xlabel('Runs from entrance to water port')
        ax1.set_ylabel('Fraction Perfect', color=color)
        ax1.plot(episodes, perfectpaths, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for the "Duration" data
        #averagedpaths = [num / total_episodes for num in paths]
        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Duration (choices)', color=color)
        ax2.plot(episodes, averagepaths, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Show the plot
        fig.tight_layout()  # To ensure the labels don't get cut off
        plt.show()


def show(qmaze, counts):  # show the process

    # Plot the path in the maze
    ax = plt.subplot(1, 2, 1)
    plt.grid('on')

    # Use spring_layout with adjusted 'k' for more space between nodes
    pos = nx.spring_layout(qmaze._maze, k=100, seed=42)  # Increase 'k' to spread nodes further apart
    nx.draw(qmaze._maze, pos=pos, ax=ax, with_labels=True, node_size=700,
            node_color=["lightgreen" if node == TERMINAL_NODE or node in qmaze.visited else "lightblue" for node in
                        qmaze._maze.nodes()], font_size=10, font_weight="bold")

    # Plot reward stats per each episode
    plt.subplot(1, 2, 2)
    best_reward = 0.66
    episodes = range(1, len(counts) + 1)
    plt.plot(episodes, counts, color='black', linestyle='-', label=r'$Total_rewards$')
    average_reward = np.mean(counts)
    plt.axhline(y=best_reward, color='green', linestyle='-.', linewidth=2, label=r'$Mean$')
    plt.axhline(y=average_reward, color='red', linestyle='-.', linewidth=2, label=r'$Mean$')
    plt.title('Episode vs Path Count')
    plt.xlabel('Episode')
    plt.ylabel('Path Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def createNetworkGraph():
    # Intialise graph object
    G = nx.Graph()

    # Initialisation of nodes and edges of tree

    edges_simple = [(1, 2), (1, 3),  # First binary choice
                    (2, 4), (2, 5), (3, 6), (3, 7),  # Second binary choice
                    (4, 8), (4, 9), (5, 10), (5, 11), (6, 12), (6, 13), (7, 14), (7, 15),
                    # Third and fourth binary choices
                    (8, 16), (8, 17), (9, 18), (9, 19), (10, 20), (10, 21), (11, 22), (11, 23),
                    (12, 24), (12, 25), (13, 26), (13, 27), (14, 28), (14, 29), (15, 30),
                    (15, 31)]  # Fifth and sixth binary choices

    edges_complex = [
        # Level 1
        (1, 2), (1, 3),
        # Level 2
        (2, 4), (2, 5), (3, 6), (3, 7),
        # Level 3
        (4, 8), (4, 9), (5, 10), (5, 11), (6, 12), (6, 13), (7, 14), (7, 15),
        # Level 4 - more complexity
        (8, 16), (8, 17), (9, 18), (9, 19), (10, 20), (10, 21), (11, 22), (11, 23),
        (12, 24), (12, 25), (13, 26), (13, 27), (14, 28), (14, 29), (15, 30), (15, 31),
        # Level 5 - even more complexity
        (16, 32), (16, 33), (17, 34), (17, 35), (18, 36), (18, 37), (19, 38), (19, 39),
        (20, 40), (20, 41), (21, 42), (21, 43), (22, 44), (22, 45), (23, 46), (23, 47),
        (24, 48), (24, 49), (25, 50), (25, 51), (26, 52), (26, 53), (27, 54), (27, 55),
        (28, 56), (28, 57), (29, 58), (29, 59), (30, 60), (30, 61), (31, 62), (31, 63),
        # Level 6 -
        (32, 64), (32, 65), (33, 66), (33, 67), (34, 68), (34, 69), (35, 70), (35, 71),
        (36, 72), (36, 73), (37, 73), (37, 75), (38, 76), (38, 77), (39, 78), (39, 79),
        (40, 80), (40, 81), (41, 82), (41, 83), (42, 84), (42, 85), (43, 86), (43, 87),
        (44, 88), (44, 89), (45, 90), (45, 91), (46, 92), (46, 93), (47, 94), (47, 95),
        (48, 96), (48, 97), (49, 98), (49, 99), (50, 100), (50, 101), (51, 102), (51, 103),
        (52, 104), (52, 105), (53, 42), (53, 43), (54, 44), (54, 45), (55, 110), (55, 111),
        (56, 112), (56, 113), (57, 114), (57, 115), (58, 116), (58, 117), (59, 118), (59, 119),
        (60, 120), (60, 121), (61, 122), (61, 123), (62, 124), (62, 125), (63, 126), (63, 127),
    ]

    edges_8_level_binary = []

    # Generate edges for a 8-level binary tree
    for parent_node in range(1, 2 ** 8):  #
        left_child = 2 * parent_node
        right_child = 2 * parent_node + 1
        edges_8_level_binary.append((parent_node, left_child))
        edges_8_level_binary.append((parent_node, right_child))

    depth = 5  # Depth of the binary tree

    # Add edges to the graph (nodes are added automatically)
    G.add_edges_from(edges_complex)

    # Draw the graph
    fig, ax = plt.subplots()

    # Change layout of binary tree so its more clear
    pos = nx.spring_layout(G, k=0.2, iterations=100)  # Alternative layout

    node_colors = []
    for node in G.nodes():
        if node == TERMINAL_NODE:
            node_colors.append("lightblue")
        elif node == STARTING_NODE:
            node_colors.append("lightgreen")
        elif node in CONTROL_NODES:
            node_colors.append("orange")
        else:
            node_colors.append("lightyellow")

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=400,
            node_color=node_colors, font_size=10,
            font_weight="bold")

    # Create legend for each node color
    start_patch = mpatches.Patch(color='lightblue', label='Reward Node')
    control_patch = mpatches.Patch(color='orange', label='Control Node')
    end_patch = mpatches.Patch(color='lightgreen', label='Start Node')
    intermediate_patch = mpatches.Patch(color='lightyellow', label='Intermediate Node')

    # Add legend to the plot
    plt.legend(handles=[start_patch, control_patch, end_patch, intermediate_patch])

    plt.title("Graph Maze Representation")
    plt.show()

    return G


# Initialise maze
mazeGraph = createNetworkGraph()

# Intialise agent and environment
env = Environment(mazeGraph)
agent = QLearningAgent(env)

# Train agent and plot results
agent.train(TOTAL_EPISODES)
