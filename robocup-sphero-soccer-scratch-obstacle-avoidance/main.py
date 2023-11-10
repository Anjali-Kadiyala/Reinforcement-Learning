import os
import Environment

# Environments: [1A_0D_0K,]
os.environ['SOCCER_ENV'] = '1A_0D_0K'

# Field size: [XS, S, M, L]
os.environ['SOCCER_DIMS'] = 'S'

MAX_EPISODES = 100000

env = Environment.make()
for episode_count in range(1, MAX_EPISODES + 1):
	env.run(episode_count)