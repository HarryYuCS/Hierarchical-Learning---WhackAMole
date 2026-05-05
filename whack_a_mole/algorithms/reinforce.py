
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as D
import torch
import numpy as np
from tqdm import trange

from whack_a_mole.algorithms.base import EvalResult, RLAlgorithm, TrainConfig, TrainResult
from whack_a_mole.algorithms.utils import flatten_observation

# TODO 2.1c : implement REINFORCE

class StochasticPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Policy network for the REINFORCE algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(StochasticPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim * 2)

        

    def forward(self, state: torch.Tensor) -> D.Normal:
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): State of the environment. Shape (N, state_dim)

        Returns:
            action_dist (D.Normal): Normal distribution representing \pi(a_t | s_t)
        """
        x = f.relu(self.fc1(state.float()))
        x = self.fc2(x)
        mu, ln_sigma = torch.split(x, x.shape[-1]//2, -1)

        return D.Normal(mu, torch.exp(ln_sigma))


class Reinforce(RLAlgorithm):
    def __init__(self, policy_net : StochasticPolicyNet):
        """Policy gradient algorithm based on the REINFORCE algorithm, with REWARD-TO-GO

        Args:
            policy_net (PolicyNet): Policy networks
            reward_to_go (bool): True if using reward_to_go, False if not (False in part 1, True in part 2)
        """
        self.policy_net = policy_net

    def compute_action(self, state : np.ndarray) -> np.ndarray:
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (np.ndarray): Action to take
        """
        return self.predict(state, deterministic=False)

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        state = flatten_observation(obs)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(self.device)
            normal_tensor = self.policy_net(state_tensor)
            action = normal_tensor.mean if deterministic else normal_tensor.sample()
            return action.cpu().numpy()

    def compute_loss(
      self,
      episode : list[tuple[np.ndarray, np.ndarray, float]],
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

        # Reward to go with DP
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

    def train(self, env, config: TrainConfig) -> TrainResult:
        """
        Wraps training loop given config
        
        for step in episodes:
            gather rollout

            update policy:
                compute loss for rollout
                step optimizer
        
        """
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        history_rewards = []
        history_losses = []
        timesteps = 0

        episode_iter = trange(config.episodes, desc="REINFORCE", disable=not config.show_progress)
        for episode_idx in episode_iter:
            obs, _ = env.reset(seed=config.seed)
            episode = []
            episode_reward = 0.0

            for _ in range(config.max_steps_per_episode):
                state = flatten_observation(obs)
                action = self.predict(state, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode.append((state, action, float(reward)))
                episode_reward += float(reward)
                timesteps += 1
                obs = next_obs
                if terminated or truncated:
                    break

            loss = self.update_policy([episode], optimizer, config.gamma)
            history_rewards.append(episode_reward)
            history_losses.append(loss)

            if config.show_progress:
                episode_iter.set_postfix(
                    loss=f"{loss:.4f}",
                    reward=f"{episode_reward:.2f}",
                    steps=timesteps,
                )
            if config.log_every > 0 and ((episode_idx + 1) % config.log_every == 0 or episode_idx == 0):
                recent_n = min(config.log_every, len(history_losses))
                avg_loss = float(np.mean(history_losses[-recent_n:]))
                avg_reward = float(np.mean(history_rewards[-recent_n:]))
                print(
                    f"[REINFORCE] episode={episode_idx + 1}/{config.episodes} "
                    f"avg_loss={avg_loss:.4f} avg_reward={avg_reward:.2f} timesteps={timesteps}"
                )

        return TrainResult(
            episode_rewards=history_rewards,
            losses=history_losses,
            timesteps=timesteps,
            metadata={"algorithm": "reinforce"},
        )

    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        """
        Evaluate the reward attainment across a number of episodes
        """
        rewards = []
        successes = []

        for _ in range(episodes):
            obs, _ = env.reset()
            total = 0.0
            final_success = 0.0
            for _ in range(500):
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                final_success = float(info.get("is_success", 0.0))
                if terminated or truncated:
                    break
            rewards.append(total)
            successes.append(final_success)

        return EvalResult(
            mean_reward=float(np.mean(rewards) if rewards else 0.0),
            std_reward=float(np.std(rewards) if rewards else 0.0),
            success_rate=float(np.mean(successes) if successes else 0.0),
            episode_rewards=rewards,
        )

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    @classmethod
    def load(cls, path: str, env=None, policy_net: StochasticPolicyNet | None = None):
        if policy_net is None:
            if env is None:
                raise ValueError("env must be provided when policy_net is not supplied")
            obs, _ = env.reset()
            state_dim = flatten_observation(obs).shape[0]
            action_dim = int(np.prod(env.action_space.shape))
            policy_net = StochasticPolicyNet(state_dim, action_dim, hidden_dim=128)
        policy_net.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(policy_net)
    
    @property
    def device(self):
        return next(iter(self.policy_net.parameters())).device

    def to(self, device : str | torch.device):
        self.policy_net.to(device)
        return self

    def select_action(self, obs):
        return self.predict(obs, deterministic=True)
