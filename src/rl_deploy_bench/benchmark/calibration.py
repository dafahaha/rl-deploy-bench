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
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

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
        # allow_pickle=False: datasets are saved with np.savez_compressed of plain
        # numpy arrays / scalars, so pickle is unnecessary and disabling it avoids
        # deserializing arbitrary objects from an untrusted .npz file.
        data = np.load(path, allow_pickle=False)
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
        consecutive_dry_episodes = 0
        max_consecutive_dry_episodes = 3

        env = gym.make(self.env_name)

        try:
            while len(observations) < config.num_samples:
                collected_before = len(observations)
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

                # Progress guard: if a whole episode accepted zero observations and
                # we still need samples, the observation_filter is rejecting
                # everything.
                #
                # When num_episodes is set, the outer loop terminates on its own
                # once episodes_completed >= num_episodes, so we must NOT raise
                # here: the user explicitly chose to bound the run and accept a
                # shorter dataset. Raising before that bound would also make the
                # old error message ("set num_episodes to bound the run") a lie,
                # since num_episodes did not actually suppress the raise.
                #
                # When num_episodes is None the outer while would spin forever
                # resetting the env. We only raise after several *consecutive*
                # dry episodes, so a single unlucky first episode (e.g. a
                # threshold filter that rejects an early CartPole episode whose
                # pole starts on the "wrong" side) does not abort the run.
                if len(observations) == collected_before:
                    consecutive_dry_episodes += 1
                else:
                    consecutive_dry_episodes = 0

                if (
                    config.num_episodes is None
                    and consecutive_dry_episodes >= max_consecutive_dry_episodes
                    and len(observations) < config.num_samples
                ):
                    raise RuntimeError(
                        "Calibration observation_filter rejected every observation "
                        f"for {max_consecutive_dry_episodes} consecutive episodes "
                        f"(collected {len(observations)}/{config.num_samples}). "
                        "Relax the filter or lower num_samples. If you want to "
                        "bound the run without an error, set num_episodes: the "
                        "generator will then stop after that many episodes even if "
                        "num_samples is not reached."
                    )

                episodes_completed += 1

                if config.num_episodes is not None and episodes_completed >= config.num_episodes:
                    break

        finally:
            env.close()

        # Degenerate bounds: num_episodes=0 stops the run before any useful
        # episode, and a filter that rejects everything (or a too-tight bound)
        # yields an empty dataset, which leaves static quantization behavior
        # undefined. Warn loudly instead of silently delivering either.
        if config.num_episodes == 0:
            warnings.warn(
                "num_episodes=0 bounds calibration to no completed episodes; "
                "the resulting dataset may be empty or truncated. Raise "
                "num_episodes to collect observations."
            )
        if len(observations) == 0:
            warnings.warn(
                f"Calibration produced 0 observations "
                f"(num_episodes={config.num_episodes}, "
                f"num_samples={config.num_samples}); an empty dataset makes "
                "static quantization behavior undefined. Raise num_episodes / "
                "num_samples or relax observation_filter."
            )

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
        """Check if observation passes the optional filter.

        A filter that raises is treated as a user bug: the exception propagates
        immediately (fail-closed) rather than being silently swallowed and the
        observation accepted. This surfaces filter bugs fast during calibration
        instead of silently collecting unfiltered data as if the filter had
        returned ``True``.
        """
        if self.config.observation_filter is None:
            return True
        return bool(self.config.observation_filter(obs))


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
