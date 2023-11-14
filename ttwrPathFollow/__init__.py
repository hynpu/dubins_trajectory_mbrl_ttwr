from gymnasium.envs.registration import register

register(
    id="ttwrPathFollow/ttwrPathFollow-v0",
    entry_point="ttwrPathFollow.envs:PathFollowEvn",
     max_episode_steps=1000,
)