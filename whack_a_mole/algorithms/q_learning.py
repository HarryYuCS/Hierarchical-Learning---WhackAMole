import torch
import torch.nn as nn
import numpy as np

from whack_a_mole.algorithms.base import EvalResult, RLAlgorithm, TrainConfig, TrainResult
from whack_a_mole.algorithms.utils import flatten_observation

class QNet(nn.Module):
    """
    A neural net to learn the Q function
    """
    def __init__(self, reward_dim : int, state_dim : int):
        """
        Initializes a neural net to approximate the q function given state and action
        
        Args:
            reward_dim : int
            state_dim : int
        """
        super().__init__()
        hidden_dim = 128
        input_dim = state_dim + reward_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state.float(), action.float()], dim=-1)
        return self.model(x).squeeze(-1)

class QLearningActor(RLAlgorithm):
    """
    An actor using Q learning with sampling since we have a continuous action space
    """
    def __init__(self, q_net):
        self.q_net = q_net
        self.action_low = None
        self.action_high = None
        self.action_dim = None

    def configure_action_space(self, action_space):
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)
        self.action_dim = int(np.prod(action_space.shape))

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        state = flatten_observation(obs)
        if self.action_dim is None:
            self.action_dim = 4
            self.action_low = -np.ones(self.action_dim, dtype=np.float32)
            self.action_high = np.ones(self.action_dim, dtype=np.float32)

        candidates = 256 if deterministic else 64
        actions = np.random.uniform(self.action_low, self.action_high, size=(candidates, self.action_dim)).astype(np.float32)
        state_batch = np.repeat(state[None, :], candidates, axis=0)

        with torch.no_grad():
            values = self.q_net(
                torch.from_numpy(state_batch).to(self.device),
                torch.from_numpy(actions).to(self.device),
            )
            best_idx = int(torch.argmax(values).item())
        return actions[best_idx]

    def compute_loss(
        self,
        transitions : list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
        gamma : float
    ):
        """
        Given rollouts, compute the MSE loss between the objective R + q_next and q_current.
        """
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.asarray(dones, dtype=np.float32)).to(self.device)

        q_values = self.q_net(states, actions)

        next_actions = []
        next_states_np = next_states.cpu().numpy()
        for ns in next_states_np:
            next_actions.append(self.predict(ns, deterministic=True))
        next_actions = torch.from_numpy(np.asarray(next_actions, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            next_q = self.q_net(next_states, next_actions)
            targets = rewards + gamma * (1.0 - dones) * next_q

        return nn.functional.mse_loss(q_values, targets)
    
    def update_policy(self, episodes, optimizer, gamma):
        """
        Given episodes, update the policy a single time.
        """
        all_transitions = []
        for episode in episodes:
            all_transitions.extend(episode)
        loss = self.compute_loss(all_transitions, gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def train(self, env, config: TrainConfig) -> TrainResult:
        """
        Wraps entire training loop given a single config

        for step in episodes:
            gather rollout

            update policy:
                compute loss for rollout
                step optimizer

        """
        self.configure_action_space(env.action_space)
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        history_rewards = []
        history_losses = []
        timesteps = 0

        for _ in range(config.episodes):
            # gather rollout
            obs, _ = env.reset(seed=config.seed)
            episode_transitions = []
            total_reward = 0.0
            for _ in range(config.max_steps_per_episode):
                state = flatten_observation(obs)
                action = self.predict(state, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = flatten_observation(next_obs)
                done = bool(terminated or truncated)
                episode_transitions.append((state, action, float(reward), next_state, done))
                total_reward += float(reward)
                timesteps += 1
                obs = next_obs
                if done:
                    break

            loss = self.update_policy([episode_transitions], optimizer, config.gamma)
            history_rewards.append(total_reward)
            history_losses.append(loss)

        return TrainResult(
            episode_rewards=history_rewards,
            losses=history_losses,
            timesteps=timesteps,
            metadata={"algorithm": "q_learning"},
        )

    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        """
        Evaluate the reward attainment across a number of episodes
        """
        self.configure_action_space(env.action_space)
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
        torch.save(self.q_net.state_dict(), path)

    @classmethod
    def load(cls, path: str, env=None, q_net: QNet | None = None):
        if q_net is None:
            if env is None:
                raise ValueError("env must be provided when q_net is not supplied")
            obs, _ = env.reset()
            state_dim = flatten_observation(obs).shape[0]
            action_dim = int(np.prod(env.action_space.shape))
            q_net = QNet(action_dim, state_dim)
        q_net.load_state_dict(torch.load(path, map_location="cpu"))
        actor = cls(q_net)
        if env is not None:
            actor.configure_action_space(env.action_space)
        return actor

    @property
    def device(self):
        return next(iter(self.q_net.parameters())).device

    def to(self, device : str | torch.device):
        self.q_net.to(device)
        return self

    def select_action(self, obs):
        return self.predict(obs, deterministic=True)

