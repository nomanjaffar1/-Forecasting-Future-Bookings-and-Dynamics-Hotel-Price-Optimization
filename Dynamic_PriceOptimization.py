import numpy as np
class DynamicPricingEnvironment:
    def __init__(self, max_price, num_bookings, days_before_stay, max_steps, default_price):
        self.max_price = max_price
        self.num_bookings = num_bookings
        self.days_before_stay = days_before_stay
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        self.price = default_price
    def reset(self):
        # Reset environment to initial state
        self.state = np.random.choice(np.arange(0,300))
        self.current_step = 0
        return self.state

    def step(self, action):
        # Simulate pricing decision and update environment
        if action == 0:
            self.price = min(self.price + 0.10 * self.price, self.max_price)
        elif action == 1:
            self.price = self.price - 0.10 * self.price
        else:
            self.price = self.price
        reward = self.calculate_reward()
        self.state = self.stateRepresentation()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.state, reward, done

    def stateRepresentation(self):
        range_A = np.arange(0, self.num_bookings) # this could be retrieved from real data
        range_B = np.arange(0, self.days_before_stay) # this might be retreived as well
        new_a = np.random.choice(range_A)
        new_b = np.random.choice(range_B)
        index = new_a * range_B.size + new_b
        return index
    def calculate_reward(self):
        # Simple reward function based on price and demand
        demand = np.random.randint(1, 11)  # Simulated demand
        revenue = self.price * demand
        #return revenue_weight * revenue_generated + (1 - revenue_weight) * occupancy_rate
        return revenue

class QLearningAgent:
    def __init__(self, num_actions, num_states, alpha=0.1, gamma=0.9, initial_epsilon=0.9, final_epsilon=0.1, epsilon_decay_steps=100):
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.initial_epsilon = initial_epsilon  # Initial epsilon value
        self.final_epsilon = final_epsilon  # Final epsilon value
        self.epsilon = initial_epsilon  # Epsilon-greedy exploration parameter
        self.epsilon_decay_steps = epsilon_decay_steps  # Number of steps to decay epsilon
        self.epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps  # Decay rate per step
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_epsilon(self, episode):
        # Update epsilon value based on episode count
        if episode <= self.epsilon_decay_steps:
            self.epsilon = self.initial_epsilon - episode * self.epsilon_decay_rate
        else:
            self.epsilon = self.final_epsilon
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

# Parameters
MAX_PRICE = 200
NUM_BOOKINGS = 10
DAYS_BEFORE_STAY = 30
MAX_STEPS = 100
NUM_ACTIONS = 3 #MAX_PRICE + 1  # Discrete price levels
NUM_STATES = NUM_BOOKINGS * DAYS_BEFORE_STAY
DEFAULT_PRICE = 500 #We can also set this at room level but this is just a demo program
# Initialize environment and agent
env = DynamicPricingEnvironment(MAX_PRICE, NUM_BOOKINGS, DAYS_BEFORE_STAY, MAX_STEPS, DEFAULT_PRICE)
agent = QLearningAgent(NUM_ACTIONS, NUM_STATES)

# Train agent
NUM_EPISODES = 10
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    agent.update_epsilon(episode)
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")