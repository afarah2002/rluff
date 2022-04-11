from gym.envs.registration import register

register(
	id='Flapper-v0',
	entry_point='flapper_env.envs:FlapperEnv'
)