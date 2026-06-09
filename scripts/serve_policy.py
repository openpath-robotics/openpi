import dataclasses
import enum
import logging
import socket
import time

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _warmup_policy(
    policy: _policy.Policy,
    *,
    action_dim: int = 16,
    action_horizon: int = 50,
    wrench_dim: int = 0,
) -> None:
    """JAX JIT warmup for both sample_actions and sample_actions_rtc.

    sample_actions (naive) and sample_actions_rtc (RTC) are compiled separately.
    Without warmup each first call takes 10–15 s, causing d_naive/d_actual
    to overflow the action horizon and corrupt the RTC soft-mask parameter d.

    Sends all possible camera keys so warmup works for 2/3/4-cam configs.
    action_dim/action_horizon/wrench_dim must match the actual model config.
    """
    IMG_H, IMG_W = 480, 640
    dummy_base = {
        "observation/state":             np.zeros(action_dim, dtype=np.float32),
        "observation/image":             np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "observation/left_wrist_image":  np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "observation/center_image":      np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "prompt": "warmup",
    }
    if wrench_dim > 0:
        dummy_base["observation/wrench"] = np.zeros(wrench_dim, dtype=np.float32)

    # ── Naive warmup (compiles sample_actions) ──────────────────────────────
    logging.info("[warmup] Compiling sample_actions (naive)  action_dim=%d wrench_dim=%d ...", action_dim, wrench_dim)
    t0 = time.monotonic()
    try:
        policy.infer(dummy_base)
    except Exception as e:
        logging.warning("[warmup] naive infer failed (non-fatal): %s", e)
    logging.info("[warmup] sample_actions done in %.1f s", time.monotonic() - t0)

    # ── RTC warmup (compiles sample_actions_rtc) ────────────────────────────
    dummy_rtc = dict(dummy_base)
    dummy_rtc["_rtc_prev_chunk"] = np.zeros((action_horizon, action_dim), dtype=np.float32)
    dummy_rtc["_rtc_d"]          = np.int32(5)
    dummy_rtc["_rtc_s"]          = np.int32(10)
    dummy_rtc["_rtc_beta"]       = np.float32(5.0)

    logging.info("[warmup] Compiling sample_actions_rtc (RTC)  prev_chunk=(%d,%d) ...", action_horizon, action_dim)
    t0 = time.monotonic()
    try:
        policy.infer(dummy_rtc)
    except Exception as e:
        logging.warning("[warmup] RTC infer failed (non-fatal): %s", e)
    logging.info("[warmup] sample_actions_rtc done in %.1f s", time.monotonic() - t0)

    logging.info("[warmup] Both JIT functions compiled — server ready.")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Resolve action_dim/action_horizon/wrench_dim from train config for correct warmup shapes.
    action_dim, action_horizon, wrench_dim = 16, 50, 0
    if isinstance(args.policy, Checkpoint):
        try:
            train_cfg = _config.get_config(args.policy.config)
            action_dim    = train_cfg.model.action_dim
            action_horizon = train_cfg.model.action_horizon
            wrench_dim    = getattr(train_cfg.model, "wrench_dim", 0)
        except Exception as e:
            logging.warning("[warmup] Could not read train config dims (using defaults): %s", e)

    # Warm up both JAX JIT functions before accepting client connections.
    # This prevents the first real inference from being 10-15x slower (JIT compile).
    _warmup_policy(policy, action_dim=action_dim, action_horizon=action_horizon, wrench_dim=wrench_dim)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
