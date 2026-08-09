import gymnasium as gym
import ale_py

gym.register_envs(ale_py)   # explicit registration; also stops linters flagging ale_py as unused

env = gym.make(
    "ALE/Pong-v5",
    obs_type="grayscale",           # (210, 160) uint8 frames
    frameskip=4,                    # one stored frame = 4 emulator ticks (default, keep it)
    repeat_action_probability=0.0,  # CRITICAL — sticky actions off, so logged action == executed action
)

obs, info = env.reset()          # starts an episode; obs = (210,160) uint8 grayscale frame

obs, _, terminated, truncated, info = env.step(action)