import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from whack_a_mole.algorithms.base import EvalResult, RLAlgorithm, TrainConfig, TrainResult
from whack_a_mole.algorithms.utils import flatten_observation

class QNet(nn.Module):
    """
    A neural net to learn the Q function
    """
    def __init__(self, action_dim: int, state_dim: int):
        """
        Initializes a neural net to approximate the q function given state and action
        
        Args:
            action_dim : int
            state_dim : int
        """
        super().__init__()
        hidden_dim = 128
        input_dim = state_dim + action_dim
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

    def _ensure_qnet_dimensions(self, state_dim: int, action_dim: int) -> None:
        expected = state_dim + action_dim
        current = self.q_net.model[0].in_features
        if current == expected:
            return
        device = self.device
        self.q_net = QNet(action_dim=action_dim, state_dim=state_dim).to(device)

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        state = flatten_observation(obs)
        if self.action_dim is None:
            self.action_dim = 4
            self.action_low = -np.ones(self.action_dim, dtype=np.float32)
            self.action_high = np.ones(self.action_dim, dtype=np.float32)
        self._ensure_qnet_dimensions(state_dim=state.shape[0], action_dim=self.action_dim)

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
        self.to(config.device)
        init_obs, _ = env.reset(seed=config.seed)
        self._ensure_qnet_dimensions(
            state_dim=flatten_observation(init_obs).shape[0],
            action_dim=self.action_dim,
        )
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        history_rewards = []
        history_losses = []
        timesteps = 0

        episode_iter = trange(config.episodes, desc="Q-Learning", disable=not config.show_progress)
        for episode_idx in episode_iter:
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

            if config.show_progress:
                episode_iter.set_postfix(
                    loss=f"{loss:.4f}",
                    reward=f"{total_reward:.2f}",
                    steps=timesteps,
                )
            if config.log_every > 0 and ((episode_idx + 1) % config.log_every == 0 or episode_idx == 0):
                recent_n = min(config.log_every, len(history_losses))
                avg_loss = float(np.mean(history_losses[-recent_n:]))
                avg_reward = float(np.mean(history_rewards[-recent_n:]))
                print(
                    f"[Q-Learning] episode={episode_idx + 1}/{config.episodes} "
                    f"avg_loss={avg_loss:.4f} avg_reward={avg_reward:.2f} timesteps={timesteps}"
                )

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
        init_obs, _ = env.reset()
        self._ensure_qnet_dimensions(
            state_dim=flatten_observation(init_obs).shape[0],
            action_dim=self.action_dim,
        )
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
