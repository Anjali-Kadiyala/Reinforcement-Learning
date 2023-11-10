from Configurations import Environment as p
from Environment.Player import Player
from Environment.Ball import Ball
from Utilities.Logger import Logger
from gymnasium import spaces
import numpy as np
import mujoco as mj
import os

class Environment_1A_0D_0K():
		def __init__(self):
				self.SIZE = os.environ['SOCCER_DIMS']

				self.environment = '1A_0D_0K/{0}.xml'.format(self.SIZE)
				self.set_env_path(self.environment)

				# Action Space
				self.action_space = spaces.Box(
						low  = np.array([p.ENV_BOLT_MIN_SPEED, p.ENV_BOLT_MIN_ROTATION], dtype=np.float32),
						high = np.array([p.ENV_BOLT_MAX_SPEED, p.ENV_BOLT_MAX_ROTATION], dtype=np.float32),
				)

				# Observation Space
				FIELD_LENGTH, FIELD_WIDTH = p.OBSERVATION_SPACE[self.SIZE]['FIELD_DIMENSIONS']

				observation_space_low = np.array([
					-FIELD_LENGTH, 											# agent_pos_x
					-FIELD_WIDTH, 											# agent_pos_y
					p.ENV_BOLT_MIN_SPEED, 							# agent_vel_x
					p.ENV_BOLT_MIN_SPEED, 							# agent_vel_y
					p.ENV_BOLT_MIN_ROTATION,						# agent_heading
					-FIELD_LENGTH, 											# ball_pos_x
					-FIELD_WIDTH, 											# ball_pos_y
					p.ENV_BALL_MIN_SPEED,  							# ball_vel_x
					p.ENV_BALL_MIN_SPEED,								# ball_vel_y
					-FIELD_LENGTH, -FIELD_WIDTH,				# away_player_1_pos_x, away_player_1_pos_y
					-FIELD_LENGTH, -FIELD_WIDTH,				# away_player_2_pos_x, away_player_2_pos_y
					-FIELD_LENGTH, -FIELD_WIDTH,				# away_player_3_pos_x, away_player_3_pos_y
					-abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][0]), 	# goal_top_x
					-abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][1]), 	# goal_top_y
					-abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][0]), 	# goal_bottom_x
					-abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][1]), 	# goal_bottom_y
				], dtype=np.float32,)

				observation_space_high = np.array([
					FIELD_LENGTH, 										# agent_pos_x
					FIELD_WIDTH, 											# agent_pos_y
					p.ENV_BOLT_MAX_SPEED, 						# agent_vel_x
					p.ENV_BOLT_MAX_SPEED, 						# agent_vel_y
					p.ENV_BOLT_MAX_ROTATION,					# agent_heading
					FIELD_LENGTH, 										# ball_pos_x
					FIELD_WIDTH, 											# ball_pos_y
					p.ENV_BALL_MAX_SPEED,  						# ball_vel_x
					p.ENV_BALL_MAX_SPEED,							# ball_vel_y
					FIELD_LENGTH, FIELD_WIDTH,				# away_player_1_pos_x, away_player_1_pos_y
					FIELD_LENGTH, FIELD_WIDTH,				# away_player_2_pos_x, away_player_2_pos_y
					FIELD_LENGTH, FIELD_WIDTH,				# away_player_3_pos_x, away_player_3_pos_y
					abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][0]), 	# goal_top_x
					abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][1]), 	# goal_top_y
					abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][0]), 	# goal_bottom_x
					abs(p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'][1]), 	# goal_bottom_y
				], dtype=np.float32,)

				self.observation_space = spaces.Box(
						low = observation_space_low,
						high= observation_space_high,
						dtype=np.float32,
				)

				self.logger = Logger()
				self.logger.write('Observation space: {0}'.format(self.observation_space))
				self.logger.write('Action space: {0}'.format(self.action_space))

		def set_env_path(self, environment):
				dirname = os.path.dirname(__file__)
				abspath = os.path.join(dirname + "/../Mujoco/" + environment)
				self.env_path = abspath

		def get_env_path(self):
				return self.env_path

		def init_characters(self, model):
				# Players - Single Player
				self.player = Player(
						model,
						p.TEAM_HOME,
						"home_player_1",
						self.observation_space,
						self.action_space,
				)

				# Ball
				self.ball = Ball(model, p.BALL)

				# Away players
				self.away_player_1 = Player(model, p.TEAM_AWAY, "away_player_1", self.observation_space, self.action_space,)
				self.away_player_2 = Player(model, p.TEAM_AWAY, "away_player_2", self.observation_space, self.action_space,)
				self.away_player_3 = Player(model, p.TEAM_AWAY, "away_player_3", self.observation_space, self.action_space,)
			
		def get_speed_from_velocity(self, velocity):
				x_vel, y_vel = velocity[0], velocity[1]
				speed = np.sqrt(x_vel**2 + y_vel**2)
				return [speed]

		def get_heading_from_velocity(self, velocity):
				x_vel, y_vel = velocity[0], velocity[1]
				rotation = ((np.arctan2(y_vel, x_vel)) + 360) % 360
				return [rotation]

		def get_observation_space(self, data):
				# agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y, agent_heading, ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, goal_top_x, goal_top_y, goal_bottom_x, goal_bottom_y
				player_position = self.player.get_position(data)
				player_velocity = self.player.get_velocity(data)[:2]
				player_rotation = [self.player.heading]

				ball_position = self.ball.get_position(data)
				ball_velocity = self.ball.get_velocity(data)[:2]

				away_player_1_position = self.away_player_1.get_position(data)
				away_player_2_position = self.away_player_2.get_position(data)
				away_player_3_position = self.away_player_3.get_position(data)

				state_player = np.concatenate((player_position[:2], player_velocity, player_rotation))
				state_ball = np.concatenate((ball_position[:2], ball_velocity))
				state_obstacle = np.concatenate((away_player_1_position[:2], away_player_2_position[:2], away_player_3_position[:2]))
				state_goal = np.concatenate((p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_TOP'], p.OBSERVATION_SPACE[self.SIZE]['GOAL_HOME_BOTTOM']))

				state_space = np.concatenate((state_player, state_ball, state_obstacle, state_goal))
				return state_space

		def scale_linear(self, x, min, max):
			# Linear scaling: f(x) = 0.5 * (max - min) * x + 0.5 * (max + min)
			return (0.5 * (max - min) * x) + (0.5 * (max + min))

		def preprocess_tanh_actions(self, action):
				# Actions are in the range [-1, 1]
				action_speed, action_rotation = action
				player_speed = self.scale_linear(action_speed, p.ENV_BOLT_MIN_SPEED, p.ENV_BOLT_MAX_SPEED)
				player_rotation = self.scale_linear(action_rotation, p.ENV_BOLT_MIN_ROTATION, p.ENV_BOLT_MAX_ROTATION)
				return player_speed, player_rotation

		def step(self, data, action):
				# Apply action on player and get reward
				speed, rotation = self.preprocess_tanh_actions(action) # ([0-20], [0-359])
				self.player.set_heading_and_velocity(data, rotation, speed)
				reward, done = self.player.get_reward(data, self.ball)

				# Get new observation
				new_observation = self.get_observation_space(data)

				# if agent collides with away players or agent touches ball
				done = done

				info = {}

				return new_observation, reward, done, info
