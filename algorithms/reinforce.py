from policy_net import PolicyNet

# TODO 2.1c : implement REINFORCE

class PolicyGradient:
    def __init__(
      self,
      policy_net : PolicyNet,
      reward_to_go: bool = False
    ):
      """Policy gradient algorithm based on the REINFORCE algorithm.

      Args:
          policy_net (PolicyNet): Policy network
          reward_to_go (bool): True if using reward_to_go, False if not (False in part 1, True in part 2)
      """
      self.policy_net = policy_net
      self.reward_to_go = reward_to_go

    @property
    def device(self):
      return next(iter(self.policy_net.parameters())).device

    def to(self, device : str | torch.device):
      self.policy_net.to(device)
      return self

    def compute_action(self, state : np.ndarray) -> np.ndarray:
      """Select an action based on the policy network

      Args:
          state (np.ndarray): State of the environment

      Returns:
          action (np.ndarray): Action to take
      """
      with torch.no_grad():
        # TODO: Implement the action selection here based on the policy network output probabilities
        # You also need to convert between numpy arrays and torch tensors
        # HINT: Use `torch.no_grad()`
        normal_tensor = self.policy_net(torch.from_numpy(state))
        action = normal_tensor.sample()

        return action.numpy()

    def compute_loss(
      self,
      episode : list[tuple[
          np.ndarray, np.ndarray, float
      ]],
      gamma : float
    ) -> torch.Tensor:
      """Compute the loss function J for the REINFORCE algorithm

      Args:
          episode (list): List of tuples (state, action, reward)
          gamma (float): Discount factor

      Returns:
          loss (torch.Tensor): The value of the loss function
      """
      states, actions, rewards = zip(*episode)

      states = torch.from_numpy(np.array(states)).float()
      actions = torch.from_numpy(np.array(actions)).float()

      normal = self.policy_net(states)
      log_probs = normal.log_prob(actions).sum(dim=-1)

      # TODO: Extract states, actions and rewards from the episode, and maybe convert them to torch
      if not self.reward_to_go:
        total_reward = 0
        for timestep, reward in enumerate(rewards):
          total_reward += (gamma ** timestep) * reward

        loss = -(log_probs.sum() * total_reward)

      else:
        T = len(episode)

        dp = np.zeros(T + 1)

        for timestep in range(T - 1, -1, -1):
          current_reward = rewards[timestep] + gamma * dp[timestep + 1]
          dp[timestep] = current_reward
        rewards = torch.from_numpy(dp[:-1]).float()

        loss = -(log_probs * rewards).sum()

      return loss

    def update_policy(self, episodes, optimizer, gamma):
      """Update the policy network using the batch of episodes

      Args:
          episodes (list): List of episodes
          optimizer (torch.optim): Optimizer
          gamma (float): Discount factor
      Returns:
          loss (float): The value of the loss function
      """
      losses = []
      for episode in episodes:
        losses.append(self.compute_loss(episode, gamma))

      loss = torch.stack(losses).mean()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      return loss.item()