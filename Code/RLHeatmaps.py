import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx

# Initialise starting and terminal node for maze graph structure -> this depends on the maze structure set in createNetworkGraph() function below
STARTING_NODE = 1
TERMINAL_NODE = 127
TERMINAL_NODES = np.linspace(64, 127, num=64)
# Set number of episodes to train
TOTAL_EPISODES = 210

# Initialise the optimal number of nodes travarsed to reach the reward
# -> I just adjusted this variable by running the simulation once or twice to see what path count it would converge to 
# -> need to create a function or something to check optimal path based on graph maze
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

        '''
        self._maze.nodes[rat]['rat'] = True
    
        # Reset 'rat' attribute for all other nodes
        for node in self._maze.nodes:
            if node != rat:
                self._maze.nodes[node]['rat'] = False
        '''

        return self.state
    
    def act(self, state, action): # Apply the action, update state, get reward

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
    
    def get_reward(self): # get reward after action, can change the value of reward
        
        node = self.state

        # Check if rat is at reward
        '''
        if node == TERMINAL_NODE: # Water port position
            return 1.0
            #return 5.0
        '''
        # Reduce repeat visits
        if (node) in self.visited:
            return -0.1
        else:
            return -1.0
        
    def game_status(self):
        
        node = self.state
        
        if node in TERMINAL_NODES: # check if rat is in terminal state
            return True
        else:
            return False

class QLearningAgent:
    def __init__(self, env,learning_rate=0.1, discount_factor=1.0,qtable_path=False):

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
            
    def choose_action_egreedy(self, state, epsilon): #epsilon-greedy

        self.epsilon = epsilon

        # Check whether to explore
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.valid_actions(self.env._maze, state))
        
        # Check and update q-table
        else:
            if state not in self.q_table:
                return np.random.choice(self.env.valid_actions(self.env._maze, state))
            else:
                #print('got max value')
                return max(self.q_table[state], key=self.q_table[state].get)
                

    def choose_action_softmax(self, state, temperature=0.1): #softmax
    
        self.temperature=temperature
        state_str = state

        if state_str not in self.q_table:
            return np.random.choice(self.env.valid_actions(self.env._maze, state))

        # Return action based on temperature
        q_values = np.array([self.q_table[state_str].get(action, 0) for action in self.env.valid_actions(self.env._maze, state)])
        softmax_probs = np.exp((q_values - np.max(q_values)) / self.temperature) / np.sum(np.exp((q_values - np.max(q_values)) / self.temperature))
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
                new_value = self.q_table[state_str][action] + self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[state_str][action])
                self.q_table[state_str][action] = new_value


    # Main RL loop
    def train(self, total_episodes):

        # Initialise arrays to store count (number of nodes visited to reach reward), episodes and rewards
        paths = []
        episodes = []
        rewards = []
        optimalpaths = []
        all_paths = []
        
        # Number of times rat took optimalpath
        optimalpathcount = 0
        terminal_nodes = []
        terminal_indexes = [[1,1],[1,3],[5,1],[5,3],[1,7],[1,9],[5,7],[5,9],[1,13],[1,15],[5,13],[5,15],[1,19],[1,21],[5,19],[5,21],[9,1],[9,3],[13,1],[13,3],[9,7],[9,9],[13,7],[13,9],[9,13],[9,15],[13,13],[13,15],[9,19],[9,21],[13,19],[13,21],[17,1],[17,3],[21,1],[21,3],[17,7],[17,9],[21,7],[21,9],[17,13],[17,15],[21,13],[21,15],[17,19],[17,21],[21,19],[21,21],[25,1],[25,3],[29,1],[29,3],[25,7],[25,9],[29,7],[29,9],[25,13],[25,15],[29,13],[29,15],[25,19],[25,21],[29,19],[29,21]]
        other_nodes = [[15,11],[7,11],[23,11],[7,5],[7,17],[23,5],[23,17],[3,5],[11,5],[3,17],[11,17],[19,5],[27,5],[19,17],[27,17],[3,2],[3,8],[11,2],[11,8],[3,14],[3,20],[11,14],[11,20],[17,2],[17,8],[27,2],[27,8],[17,14],[17,20],[27,14],[27,20],[1,2],[5,2],[1,8],[5,8],[9,2],[13,2],[9,8],[13,8],[1,14],[5,14],[1,20],[5,20],[9,14],[13,14],[9,20],[13,20],[19,2],[21,2],[19,8],[21,8],[25,2],[29,2],[25,8],[29,8],[19,14],[21,14],[19,20],[21,20],[25,14],[29,14],[25,20],[29,20]]
        ordered_nodes = [[15,11],[7,11],[23,11],[7,5],[7,17],[23,5],[23,17],[3,5],[11,5],[3,17],[11,17],[19,5],[27,5],[19,17],[27,17],[3,2],[3,8],[11,2],[11,8],[3,14],[3,20],[11,14],[11,20],[17,2],[17,8],[27,2],[27,8],[17,14],[17,20],[27,14],[27,20],[1,2],[5,2],[1,8],[5,8],[9,2],[13,2],[9,8],[13,8],[1,14],[5,14],[1,20],[5,20],[9,14],[13,14],[9,20],[13,20],[19,2],[21,2],[19,8],[21,8],[25,2],[29,2],[25,8],[29,8],[19,14],[21,14],[19,20],[21,20],[25,14],[29,14],[25,20],[29,20],[1,1],[1,3],[5,1],[5,3],[1,7],[1,9],[5,7],[5,9],[1,13],[1,15],[5,13],[5,15],[1,19],[1,21],[5,19],[5,21],[9,1],[9,3],[13,1],[13,3],[9,7],[9,9],[13,7],[13,9],[9,13],[9,15],[13,13],[13,15],[9,19],[9,21],[13,19],[13,21],[17,1],[17,3],[21,1],[21,3],[17,7],[17,9],[21,7],[21,9],[17,13],[17,15],[21,13],[21,15],[17,19],[17,21],[21,19],[21,21],[25,1],[25,3],[29,1],[29,3],[25,7],[25,9],[29,7],[29,9],[25,13],[25,15],[29,13],[29,15],[25,19],[25,21],[29,19],[29,21]]
        heatmap_arrays = []
        maxcount = 0
        interval = 70
        for episode in range(total_episodes):

            # Reinitialise environment
            env.__init__(self.env._maze)
            state = self.env.reset(STARTING_NODE)

            # Append episodes
            episodes.append(episode)

            # Reitinitalise episode variables
            done = False
            total_reward = 0
            path = 0
            epsilon = 0.25
            while not done:
                action = self.choose_action_softmax(state)
                # Perform action on current state
                reward, next_state, done = self.env.act(state, action)
                if next_state in TERMINAL_NODES:
                    terminal_nodes.append(next_state)
                # Update q-table
                self.update_q_table(state, action, reward, next_state)

                # Update state and episode variables
                total_reward += reward
                state = next_state
                path += 1

            # Check if optimal path was taken 
            if path <= OPTIMAL_PATH and state == TERMINAL_NODE:
                optimalpathcount += 1
            
            # Cumulative count of optimal paths as episodes increase
            optimalpaths.append(optimalpathcount)

            # Append path and reward arrays
            paths.append(path)
            rewards.append(total_reward)

            # Print Episode information and path taken
            print(f"Episode {episode+1}, Total Reward: {total_reward}, Done:{done}, Path Count:{path}")
            print("Path taken: ", self.env.visited)    
            all_paths = all_paths + self.env.visited
            # Show visual of path taken for each episode 
            #show(self.env,episode, count, counts)
            count = 0
            if (episode+1) % interval == 0:
                heatmap_array = np.array([
                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.],
                    [0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.],
                    [0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.],
                    [0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.75,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.],
                    [0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    [0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.],
                    [0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.75,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.],
                    [0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.75,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.75,0.,0.,0.75,0.,0.],
                    [0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.,0.,0.,0.,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.,0.],
                    [0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.,0.,0.,0.,0.75,0.,0.],
                    [0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.,0.,0.,0.75,0.75,0.75,0.],
                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    ])
                # Plot figure to show number of times optimal path taken as episodes increased
                g1 = plt.figure()
                plt.plot(episodes, optimalpaths)
                plt.xlabel("Episode Num")
                plt.ylabel("Num of Times Optimal Path Taken")
                plt.title("Num of Times Optimal Path Taken vs Num of Episodes")
                plt.figure()
                g1.show()
                
                for i in terminal_indexes:
                    count = all_paths.count(terminal_indexes.index(i)+64)
                    if count:
                        heatmap_array[i[0]][i[1]] = count
                    if count > maxcount:
                        maxcount = count
                '''
                for i in ordered_nodes:
                    count = all_paths.count(ordered_nodes.index(i)+1)
                    if count:
                        heatmap_array[i[0]][i[1]] = count
                '''
                all_paths = []
                heatmap_arrays.append(heatmap_array)
        counter = 0
        for i in heatmap_arrays:
            g2 = plt.figure()
            cmap = plt.cm.get_cmap('Blues')
            cmap.set_under('black')
            plt.imshow(i, cmap=cmap, interpolation='nearest', vmin = 0.5)
            plt.colorbar()
            plt.title("Episodes: " + str(counter*interval+1) + " - " + str(counter*interval+interval))
            plt.tick_params(left = False, right = False , labelleft = False , 
            labelbottom = False, bottom = False)
            counter = counter + 1
            g2.show()

def show(qmaze,i,count,counts):#show the process

    # Plot the path in the maze
    ax = plt.subplot(1, 2, 1)
    plt.grid('on')

    # Use spring_layout with adjusted 'k' for more space between nodes
    pos = nx.spring_layout(qmaze._maze, k=100, seed=42)  # Increase 'k' to spread nodes further apart
    nx.draw(qmaze._maze, pos=pos, ax=ax, with_labels=True, node_size=700, node_color=["lightgreen" if node == TERMINAL_NODE or node in qmaze.visited else "lightblue" for node in qmaze._maze.nodes()], font_size=10, font_weight="bold")
    
    # Plot reward stats per each episode
    plt.subplot(1, 2, 2)
    best_reward = 0.66
    episodes = range(1, len(counts) + 1)
    plt.plot(episodes, counts,color='black', linestyle='-',label=r'$Total_rewards$')
    average_reward=np.mean(counts)
    plt.axhline(y=best_reward, color='green', linestyle='-.', linewidth=2,label=r'$Mean$')
    plt.axhline(y=average_reward, color='red', linestyle='-.', linewidth=2,label=r'$Mean$')
    plt.title('Episode vs Path Count')
    plt.xlabel('Episode')
    plt.ylabel('Path Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def createNetworkGraph(): 

    # Intialise graph object
    G = nx.Graph()
    
    # This is the initialisation of nodes and edges
    # -> Can create different arrays here to change graph complexity
    # -> Play around with this
    # These were mostly generated by Chatgpt I cba to initialise my own arrays lmao
    edges_simple = [(1, 2), (1, 3),  # First binary choice
               (2, 4), (2, 5), (3, 6), (3, 7),  # Second binary choice
               (4, 8), (4, 9), (5, 10), (5, 11), (6, 12), (6, 13), (7, 14), (7, 15),  # Third and fourth binary choices
               (8, 16), (8, 17), (9, 18), (9, 19), (10, 20), (10, 21), (11, 22), (11, 23),
               (12, 24), (12, 25), (13, 26), (13, 27), (14, 28), (14, 29), (15, 30), (15, 31)]  # Fifth and sixth binary choices
    
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
    
    # Add edges to the graph (nodes are added automatically)
    G.add_edges_from(edges_complex)

    # Draw the graph
    fig, ax = plt.subplots()
    pos = nx.balanced_tree(2, 6)  # Use balanced_tree to get positions
    pos = nx.spring_layout(G)  # Alternative layout
    nx.draw(G, pos,ax=ax, with_labels=True, node_size=700, node_color=["lightgreen" if node == TERMINAL_NODE else "lightblue" for node in G.nodes()], font_size=10, font_weight="bold")
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
