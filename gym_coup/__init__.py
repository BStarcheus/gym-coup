from gym.envs.registration import register

register(
    id='coup-v0',
    entry_point='gym_coup.envs:CoupEnv',
)