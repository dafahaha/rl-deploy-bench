"""Calibration data generator for RL model quantization.

This is a key RL-specific feature: instead of using random noise for
quantization calibration (which generic ML tools do), we generate
realistic observations by running a policy in a simulation environment.

This produces much better quantization results because the calibration
data matches the actual observation distribution the model will see
during deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationConfig:
    """Configuration for calibration data generation."""

    num_samples: int = 500
    collection_strategy: str = "policy"  # 'random', 'policy', 'mixed'
    mixed_ratio: float = 0.3  # ratio of random samples in 'mixed' mode
    max_steps_per_episode: int = 1000
    num_episodes: Optional[int] = None
    seed: int = 42
    include_initial_observations: bool = True
    observation_filter: Optional[Callable[[np.ndarray], bool]] = None


@dataclass
class CalibrationDataset:
    """Dataset of calibration observations."""

    observations: np.ndarray
    config: CalibrationConfig
    env_name: str = ""
    collection_stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.observations[idx]

    def save(self, path: str) -> str:
        """Save calibration dataset to .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            env_name=self.env_name,
            num_samples=len(self.observations),
            config_num_samples=self.config.num_samples,
            config_strategy=self.config.collection_strategy,
            config_seed=self.config.seed,
        )
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "CalibrationDataset":
        """Load calibration dataset from .npz file."""
        data = np.load(path, allow_pickle=True)
        config = CalibrationConfig(
            num_samples=int(data.get("config_num_samples", len(data["observations"]))),
            collection_strategy=str(data.get("config_strategy", "policy")),
            seed=int(data.get("config_seed", 42)),
        )
        return cls(
            observations=data["observations"],
            config=config,
            env_name=str(data.get("env_name", "")),
        )

    def get_statistics(self) -> dict:
        """Get statistics about the calibration dataset."""
        obs = self.observations
        return {
            "num_samples": len(obs),
            "observation_shape": list(obs.shape[1:]),
            "mean": float(np.mean(obs)),
            "std": float(np.std(obs)),
            "min": float(np.min(obs)),
            "max": float(np.max(obs)),
            "per_dimension_mean": [float(m) for m in np.mean(obs, axis=0)],
            "per_dimension_std": [float(s) for s in np.std(obs, axis=0)],
        }


class EnvironmentCalibrationGenerator:
    """Generate calibration data from a Gymnasium environment.

    Uses a trained policy (or random actions) to collect realistic
    observations from the environment for quantization calibration.
    """

    def __init__(
        self,
        env_name: str,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        config: Optional[CalibrationConfig] = None,
    ):
        """Initialize the calibration data generator.

        Args:
            env_name: Name of the Gymnasium environment (e.g., 'Pendulum-v1').
            policy: Optional policy function that takes an observation and
                returns an action. If None, random actions are used.
            config: Calibration configuration.
        """
        self.env_name = env_name
        self.policy = policy
        self.config = config or CalibrationConfig()

    def generate(self) -> CalibrationDataset:
        """Generate calibration dataset.

        Returns:
            CalibrationDataset with collected observations.
        """
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "gymnasium is required for environment-based calibration. "
                "Install with: pip install gymnasium"
            )

        config = self.config
        rng = np.random.RandomState(config.seed)
        observations = []
        episodes_completed = 0
        total_steps = 0

        env = gym.make(self.env_name)

        try:
            while len(observations) < config.num_samples:
                obs, _ = env.reset(seed=config.seed + episodes_completed)

                if config.include_initial_observations:
                    if self._passes_filter(obs):
                        observations.append(obs.copy())

                for step in range(config.max_steps_per_episode):
                    if len(observations) >= config.num_samples:
                        break

                    # Choose action based on strategy
                    action = self._choose_action(obs, rng, env)

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_steps += 1

                    if self._passes_filter(obs):
                        observations.append(obs.copy())

                    if terminated or truncated:
                        break

                episodes_completed += 1

                if config.num_episodes is not None and episodes_completed >= config.num_episodes:
                    break

        finally:
            env.close()

        # Trim to exact number of samples
        observations = np.array(observations[: config.num_samples], dtype=np.float32)

        stats = {
            "episodes_completed": episodes_completed,
            "total_steps": total_steps,
            "collection_strategy": config.collection_strategy,
        }

        return CalibrationDataset(
            observations=observations,
            config=config,
            env_name=self.env_name,
            collection_stats=stats,
        )

    def _choose_action(self, obs: np.ndarray, rng: np.random.RandomState, env) -> np.ndarray:
        """Choose an action based on the collection strategy."""
        strategy = self.config.collection_strategy

        if strategy == "random":
            return env.action_space.sample()

        elif strategy == "policy":
            if self.policy is not None:
                action = self.policy(obs)
                if isinstance(action, tuple):
                    action = action[0]
                return np.asarray(action)
            return env.action_space.sample()

        elif strategy == "mixed":
            if rng.random() < self.config.mixed_ratio:
                return env.action_space.sample()
            elif self.policy is not None:
                action = self.policy(obs)
                if isinstance(action, tuple):
                    action = action[0]
                return np.asarray(action)
            return env.action_space.sample()

        else:
            raise ValueError(f"Unknown collection strategy: {strategy}")

    def _passes_filter(self, obs: np.ndarray) -> bool:
        """Check if observation passes the optional filter."""
        if self.config.observation_filter is None:
            return True
        try:
            return bool(self.config.observation_filter(obs))
        except Exception:
            return True


class SB3PolicyCalibrationGenerator(EnvironmentCalibrationGenerator):
    """Calibration generator using a Stable Baselines3 trained policy.

    Convenience class that wraps an SB3 model as the policy.
    """

    def __init__(
        self,
        env_name: str,
        sb3_model,
        config: Optional[CalibrationConfig] = None,
        deterministic: bool = True,
    ):
        """Initialize with an SB3 model.

        Args:
            env_name: Gymnasium environment name.
            sb3_model: Trained Stable Baselines3 model.
            config: Calibration configuration.
            deterministic: Whether to use deterministic actions.
        """
        self.sb3_model = sb3_model
        self.deterministic = deterministic

        def policy(obs: np.ndarray) -> np.ndarray:
            action, _ = sb3_model.predict(obs, deterministic=deterministic)
            return action

        super().__init__(env_name, policy=policy, config=config)


def create_calibration_data_reader(
    dataset: CalibrationDataset,
    input_name: str = "observation",
):
    """Create an ONNX Runtime CalibrationDataReader from a CalibrationDataset.

    This bridges the RL-specific calibration data generation with ONNX
    Runtime's static quantization API.

    Args:
        dataset: CalibrationDataset to use.
        input_name: Name of the model input.

    Returns:
        A CalibrationDataReader instance compatible with onnxruntime.quantization.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class DatasetCalibrationReader(CalibrationDataReader):
        def __init__(self, data: CalibrationDataset, name: str):
            super().__init__()
            self.data = data
            self.input_name = name
            self.current = 0

        def get_next(self):
            if self.current >= len(self.data):
                return None
            obs = self.data[self.current]
            self.current += 1
            # Add batch dimension
            return {self.input_name: obs[np.newaxis].astype(np.float32)}

        def rewind(self):
            self.current = 0

    return DatasetCalibrationReader(dataset, input_name)


def generate_calibration_from_env(
    env_name: str,
    output_path: Optional[str] = None,
    num_samples: int = 500,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sb3_model=None,
    deterministic: bool = True,
    seed: int = 42,
) -> Tuple[CalibrationDataset, str]:
    """Convenience function to generate and optionally save calibration data.

    Args:
        env_name: Gymnasium environment name.
        output_path: Optional path to save the .npz dataset.
        num_samples: Number of calibration samples to collect.
        policy: Optional policy function.
        sb3_model: Optional SB3 trained model (used if policy is None).
        deterministic: Whether to use deterministic actions for SB3 policy.
        seed: Random seed.

    Returns:
        Tuple of (CalibrationDataset, saved_path or empty string).
    """
    config = CalibrationConfig(num_samples=num_samples, seed=seed)

    if sb3_model is not None:
        generator = SB3PolicyCalibrationGenerator(
            env_name, sb3_model, config=config, deterministic=deterministic
        )
    else:
        generator = EnvironmentCalibrationGenerator(env_name, policy=policy, config=config)

    dataset = generator.generate()

    saved_path = ""
    if output_path:
        saved_path = dataset.save(output_path)

    return dataset, saved_path
