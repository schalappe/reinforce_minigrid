# Proximal Policy Optimization (PPO) for Maze Navigation

## Table of Contents
1. [Conceptual Overview of PPO](#conceptual-overview-of-ppo)
   - [Key Advantages](#key-advantages)
2. [PPO Algorithm Breakdown](#ppo-algorithm-breakdown)
   - [Data Collection](#data-collection)
   - [Advantage Estimation](#advantage-estimation)
   - [Clipped Surrogate Objective](#clipped-surrogate-objective)
   - [Policy (Actor) Updates](#policy-actor-updates)
   - [Value Function (Critic) Updates](#value-function-critic-updates)
   - [Key Hyperparameters](#key-hyperparameters)
3. [Applying PPO to the Maze Environment](#applying-ppo-to-the-maze-environment)
   - [Environment Characteristics](#environment-characteristics)
   - [State and Action Space Considerations](#state-and-action-space-considerations)
   - [Reward Structure Adaptation](#reward-structure-adaptation)
   - [Exploration Strategy](#exploration-strategy)
   - [Network Architecture](#network-architecture)
   - [Handling Sparse Rewards](#handling-sparse-rewards)
   - [Implementation Guidelines](#implementation-guidelines)

## Conceptual Overview of PPO

Proximal Policy Optimization (PPO) is a policy gradient algorithm for reinforcement learning introduced by OpenAI in 2017. It belongs to the family of actor-critic methods, combining the benefits of both policy gradient and value-based methods. PPO addresses key challenges in reinforcement learning by providing a more stable, sample-efficient approach to policy optimization.

The core principle behind PPO is to improve the policy in a conservative manner by limiting how much the policy can change in a single update. This is achieved by introducing a "clipped" surrogate objective function that constrains policy updates to stay within a small region around the current policy, preventing excessively large policy changes that could lead to performance collapse.

### Key Advantages

1. **Stability**: PPO offers significant stability improvements over older policy gradient methods by constraining the size of policy updates.
2. **Sample Efficiency**: Compared to methods like REINFORCE or vanilla policy gradients, PPO makes better use of collected experience by performing multiple optimization steps on minibatches of experience.
3. **Simplicity**: Despite its effectiveness, PPO is relatively straightforward to implement, requiring fewer hyperparameters to tune compared to algorithms like Trust Region Policy Optimization (TRPO).
4. **Continuous and Discrete Action Spaces**: PPO can handle both discrete action spaces (like those in the maze environment) and continuous action spaces.
5. **Balance Between Exploration and Exploitation**: The algorithm naturally balances between exploring new strategies and exploiting known good strategies.

## PPO Algorithm Breakdown

### Data Collection

The first step in PPO is to collect experience by having the agent interact with the environment using the current policy:

1. **Trajectory Collection**: The agent collects a set of trajectories (sequences of states, actions, rewards, and next states) by executing the current policy in the environment for a fixed number of timesteps.
2. **Storage**: These trajectories are stored in a buffer along with the probabilities assigned to each action by the current policy (old action probabilities).

```python
# Pseudocode for data collection
def collect_trajectories(policy, env, timesteps_per_batch):
    # Storage for trajectories
    states, actions, rewards, next_states, dones, action_probs = [], [], [], [], [], []
    
    # Reset the environment
    state, _ = env.reset()
    done = False
    
    for t in range(timesteps_per_batch):
        # Get action and its probability from current policy
        action_distribution = policy(state)
        action = action_distribution.sample()
        action_prob = action_distribution.log_prob(action)
        
        # Execute action in environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Store the transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        action_probs.append(action_prob)
        
        # Update state
        state = next_state
        
        # Reset if episode is done
        if done or truncated:
            state, _ = env.reset()
    
    return states, actions, rewards, next_states, dones, action_probs
```

### Advantage Estimation

After collecting data, PPO calculates the advantage for each state-action pair, which measures how much better taking an action is compared to the average action in that state:

1. **Return Calculation**: Compute the return (discounted sum of rewards) for each timestep.
2. **Value Estimation**: Use the value function (critic) to estimate the expected return from each state.
3. **Advantage Calculation**: Calculate the advantage, typically using Generalized Advantage Estimation (GAE).

GAE calculates the advantage as:

$$A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + ... + (\gamma \lambda)^{T-t-1} \delta_{T-1}$$

Where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (the TD error)
- $\gamma$ is the discount factor
- $\lambda$ is the GAE parameter that controls the trade-off between bias and variance

```python
# Pseudocode for GAE calculation
def compute_gae(rewards, values, next_values, dones, gamma, lam):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Terminal state
            delta = rewards[t] - values[t]
            gae = delta
        else:
            # Non-terminal state
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
        
        advantages.append(gae)
    
    # Reverse the list since we computed it backwards
    advantages = advantages[::-1]
    
    # Calculate returns as advantage + value
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return advantages, returns
```

### Clipped Surrogate Objective

The heart of PPO is its novel objective function that prevents excessive policy updates:

$$L^{CLIP}(\theta) = \hat{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies
- $\hat{A}_t$ is the estimated advantage
- $\epsilon$ is the clipping parameter (typically 0.1 or 0.2)

This objective function:
1. Clips the probability ratio to stay within $[1-\epsilon, 1+\epsilon]$
2. Takes the minimum of the clipped and unclipped objectives
3. Ensures the policy doesn't change too drastically in a single update

```python
# Pseudocode for PPO objective
def ppo_loss(states, actions, old_action_probs, advantages, policy, clip_param=0.2):
    # Get current action probabilities
    current_action_distribution = policy(states)
    current_action_probs = current_action_distribution.log_prob(actions)
    
    # Probability ratio (π_θ / π_θ_old)
    ratio = torch.exp(current_action_probs - old_action_probs)
    
    # Clipped objective
    clip_adv = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages
    
    # Take the minimum
    loss = -torch.min(ratio * advantages, clip_adv).mean()
    
    return loss
```

### Policy (Actor) Updates

The policy network (actor) is updated by optimizing the clipped surrogate objective:

1. **Gradient Computation**: Calculate gradients of the PPO loss with respect to the policy parameters.
2. **Parameter Updates**: Update the policy parameters using an optimizer like Adam.
3. **Multiple Epochs**: Perform multiple optimization steps on randomly sampled minibatches of the collected data.

```python
# Pseudocode for policy updates
def update_policy(policy, optimizer, states, actions, old_action_probs, advantages, epochs, batch_size, clip_param):
    # Convert to tensors if needed
    dataset = TensorDataset(states, actions, old_action_probs, advantages)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for state_batch, action_batch, old_prob_batch, advantage_batch in dataloader:
            # Calculate loss
            loss = ppo_loss(state_batch, action_batch, old_prob_batch, advantage_batch, policy, clip_param)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Value Function (Critic) Updates

The value function (critic) is updated to better predict expected returns:

1. **Loss Calculation**: Compute the mean squared error between predicted values and target returns.
2. **Parameter Updates**: Update the value function parameters to minimize this error.

```python
# Pseudocode for value function updates
def update_value_function(value_function, optimizer, states, returns, epochs, batch_size):
    # Convert to tensors if needed
    dataset = TensorDataset(states, returns)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for state_batch, return_batch in dataloader:
            # Predict values
            value_pred = value_function(state_batch)
            
            # Calculate MSE loss
            loss = F.mse_loss(value_pred, return_batch)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Key Hyperparameters

PPO involves several important hyperparameters that significantly affect performance:

1. **Clipping Parameter (ε)**: Typically set to 0.1 or 0.2. Controls how much the policy can change in one update.
2. **GAE Parameter (λ)**: Usually between 0.9 and 1.0. Controls the trade-off between bias and variance in advantage estimation.
3. **Discount Factor (γ)**: Typically 0.99. Determines the importance of future rewards.
4. **Learning Rate**: The step size for the optimizer. Often requires tuning for specific tasks.
5. **Number of Epochs**: How many passes through the data per update. Usually 3-10.
6. **Batch Size**: The size of minibatches used for updates. Smaller batches provide more updates but with higher variance.
7. **Timesteps per Batch**: How many environment steps to collect before an update. Affects sample efficiency.
8. **Value Function Coefficient**: Weight for the value function loss in the combined objective.
9. **Entropy Coefficient**: Weight for the entropy bonus that encourages exploration.

## Applying PPO to the Maze Environment

### Environment Characteristics

The maze environment described in the documentation presents specific challenges and characteristics:

- **3x3 Grid of Rooms**: Each room is 8x8 cells, creating a complex navigation task.
- **Random Starting Position**: The agent starts in a random position each episode.
- **Random Goal Position**: The goal (green ball) is placed randomly in one of the rooms.
- **Door Connectivity**: Rooms are connected by doors that may be open or closed.
- **Structured Reward**: A combination of distance-based progress, room transitions, exploration penalties, and terminal rewards.

### State and Action Space Considerations

**State Space:**
- The maze environment provides observations that include the agent's view of the grid.
- For PPO, we need to ensure the neural network can effectively process this grid-based input.
- Options:
  1. **CNN-Based Approach**: If using raw image observations, a CNN can extract spatial features.
  2. **Feature-Based Approach**: We could extract key features like relative position to goal, current room coordinates, door locations, etc.

```python
# Example state preprocessing
def preprocess_state(observation):
    # For image-based observation
    if isinstance(observation, dict) and 'image' in observation:
        # Convert to tensor and normalize
        image = torch.tensor(observation['image'], dtype=torch.float32) / 255.0
        return image
    # For other observation types, customize as needed
    return torch.tensor(observation, dtype=torch.float32)
```

**Action Space:**
- The maze has a discrete action space (e.g., move left, right, forward).
- PPO implementation should use a categorical policy distribution for action selection.

```python
# Example action distribution
def get_action_distribution(policy_logits):
    return torch.distributions.Categorical(logits=policy_logits)
```

### Reward Structure Adaptation

The maze environment has a sophisticated reward structure that PPO can leverage:

1. **Terminal Reward** (+10.0): PPO's advantage estimation will properly propagate this large reward backward through time.

2. **Distance Progress Reward**: This provides dense feedback, helping PPO learn more efficiently than with sparse rewards alone.

3. **Room Transition Reward**: This coarser reward signal helps guide exploration at the room level.

4. **Exploration Penalty**: Discourages revisiting states, pushing the agent to find new paths.

5. **Step Penalty**: Encourages finding efficient paths.

PPO's advantage estimation through GAE is well-suited to handle this multi-component reward structure:

```python
# Example of handling the maze reward structure in PPO
def compute_returns_and_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    # Standard GAE computation works well with the maze's reward structure
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Terminal state - no bootstrapping
            delta = rewards[t] - values[t]
            gae = delta
        else:
            # Non-terminal state
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
        
        advantages.append(gae)
    
    # Reverse and compute returns
    advantages = advantages[::-1]
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return returns, advantages
```

### Exploration Strategy

The maze environment requires effective exploration to find the goal. PPO can facilitate exploration through:

1. **Entropy Bonus**: Adding an entropy term to the objective encourages policy randomness:

```python
# Adding entropy bonus to PPO loss
def ppo_loss_with_entropy(policy_distribution, states, actions, old_action_probs, advantages, entropy_coef=0.01):
    # Standard PPO loss
    standard_loss = ppo_loss(states, actions, old_action_probs, advantages)
    
    # Calculate entropy
    entropy = policy_distribution.entropy().mean()
    
    # Combined loss with entropy bonus
    total_loss = standard_loss - entropy_coef * entropy
    
    return total_loss
```

2. **Adaptive Exploration**: Gradually reducing exploration as training progresses:

```python
# Example of adaptive entropy coefficient
initial_entropy_coef = 0.01
min_entropy_coef = 0.001
decay_factor = 0.999

# During training
entropy_coef = max(initial_entropy_coef * (decay_factor ** episode), min_entropy_coef)
```

3. **Curiosity-Driven Exploration**: For particularly challenging maze configurations, an intrinsic curiosity module can be added:

```python
# Simplified intrinsic curiosity module
def compute_intrinsic_reward(state, next_state, forward_model):
    # Predict next state
    predicted_next_state = forward_model(state)
    
    # Prediction error as intrinsic reward
    intrinsic_reward = F.mse_loss(predicted_next_state, next_state)
    
    return intrinsic_reward
```

### Network Architecture

Designing effective networks for the maze environment:

**Policy Network (Actor):**
```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(PolicyNetwork, self).__init__()
        
        # For image-based observations (assuming 3D input)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        conv_output_size = self._get_conv_output(input_dim)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def _get_conv_output(self, shape):
        # Helper function to calculate output size of conv layers
        bs = 1
        x = torch.rand(bs, *shape)
        output = self.conv(x)
        return output.view(bs, -1).size(1)
    
    def forward(self, x):
        # Forward pass through the network
        features = self.conv(x)
        logits = self.fc(features)
        return logits
```

**Value Network (Critic):**
```python
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        # Similar convolutional structure as policy network
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        conv_output_size = self._get_conv_output(input_dim)
        
        # Fully connected layers for value prediction
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output
        )
    
    def _get_conv_output(self, shape):
        bs = 1
        x = torch.rand(bs, *shape)
        output = self.conv(x)
        return output.view(bs, -1).size(1)
    
    def forward(self, x):
        features = self.conv(x)
        value = self.fc(features)
        return value
```

### Handling Sparse Rewards

Despite the reward shaping in the maze environment, rewards can still be relatively sparse in large mazes. PPO can address this through:

1. **Generalized Advantage Estimation (GAE)**: By using an appropriate λ value (e.g., 0.95), PPO effectively propagates rewards backward through time.

2. **Curriculum Learning**: Start with simpler maze configurations (e.g., open doors, smaller rooms) and gradually increase complexity:

```python
# Example of curriculum learning implementation
def get_environment_difficulty(progress_percentage):
    # Start with all doors open
    doors_open = progress_percentage < 0.3
    
    # Adjust room size (could require environment modifications)
    room_size = max(4, min(8, int(4 + progress_percentage * 4)))
    
    return {
        'doors_open': doors_open,
        'room_size': room_size
    }

# During training
training_progress = current_episode / total_episodes
env_params = get_environment_difficulty(training_progress)
env = Maze(**env_params)
```

3. **Intermediate Goals**: Adding auxiliary tasks or rewards for discovering new rooms or approaching the goal room.

### Implementation Guidelines

1. **Data Collection**:
   - Collect trajectory data using the current policy
   - Store states, actions, rewards, next states, dones, and old action probabilities

2. **Advantage Calculation**:
   - Compute returns and advantages using GAE
   - Consider the multi-component reward structure of the maze

3. **Policy Updates**:
   - Use the clipped surrogate objective
   - Include entropy bonus for exploration
   - Perform multiple epochs of minibatch updates

4. **Value Function Updates**:
   - Train the value function to predict expected returns
   - This helps reduce variance in policy gradient estimates

5. **Hyperparameter Tuning**:
   - Discount factor (γ): 0.99 is a good starting point
   - GAE parameter (λ): 0.95 balances bias and variance
   - Clipping parameter (ε): 0.2 is commonly used
   - Learning rate: Start with 3e-4 for Adam optimizer
   - Number of epochs: 4-10 epochs per update
   - Batch size: 64-256 for minibatch updates
   - Entropy coefficient: 0.01, possibly with decay

6. **Evaluation and Monitoring**:
   - Track average episode reward
   - Monitor policy entropy to ensure appropriate exploration
   - Visualize agent trajectories to identify navigation patterns

By following these guidelines and adapting PPO to the specific characteristics of the maze environment, it becomes possible to learn effective navigation policies that balance exploration and exploitation, handle the structured reward system, and efficiently find paths to the goal across various maze configurations.