from Configurations import Environment as envProps
from Utilities.Logger import Logger

from mujoco.glfw import glfw
import mujoco as mj

import os
import numpy as np
import time

class Simulation:
		def __init__(self, env, sim_length=45, frames_per_second=60):
				self.env = env
				self.logger = Logger()

				self.sim_length = sim_length
				self.frames_per_second = frames_per_second

				self.button_left = False
				self.button_middle = False
				self.button_right = False
				self.lastx = 0
				self.lasty = 0

				self.model = mj.MjModel.from_xml_path(self.env.get_env_path())
				self.data = mj.MjData(self.model)
				self.cam = mj.MjvCamera()
				self.opt = mj.MjvOption()

				self.env.init_characters(self.model)

				glfw.init()
				self.window = glfw.create_window(1200, 900, "RoboCup Soccer - SSL", None, None)
				glfw.make_context_current(self.window)
				glfw.swap_interval(1)

				mj.mjv_defaultCamera(self.cam)
				mj.mjv_defaultOption(self.opt)

				glfw.set_key_callback(self.window, self.keyboard)
				glfw.set_cursor_pos_callback(self.window, self.mouse_move)
				glfw.set_mouse_button_callback(self.window, self.mouse_button)
				glfw.set_scroll_callback(self.window, self.mouse_scroll)

				self.scene = mj.MjvScene(self.model, maxgeom=10000)
				self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

		def keyboard(self, window, key, scancode, act, mods):
				if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
						mj.mj_resetData(self.model, self.data)
						mj.mj_forward(self.model, self.data)

		def mouse_button(self, window, button, act, mods):
				self.button_left = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
				)
				self.button_middle = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
				)
				self.button_right = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
				)

				glfw.get_cursor_pos(window)

		def mouse_move(self, window, xpos, ypos):
				dx = xpos - self.lastx
				dy = ypos - self.lasty
				self.lastx = xpos
				self.lasty = ypos

				if (
						(not self.button_left)
						and (not self.button_middle)
						and (not self.button_right)
				):
						return

				width, height = glfw.get_window_size(window)

				PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
				PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
				mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

				if self.button_right:
						if mod_shift:
								action = mj.mjtMouse.mjMOUSE_MOVE_H
						else:
								action = mj.mjtMouse.mjMOUSE_MOVE_V
				elif self.button_left:
						if mod_shift:
								action = mj.mjtMouse.mjMOUSE_ROTATE_H
						else:
								action = mj.mjtMouse.mjMOUSE_ROTATE_V
				else:
						action = mj.mjtMouse.mjMOUSE_ZOOM

				mj.mjv_moveCamera(
						self.model, action, dx / height, dy / height, self.scene, self.cam
				)

		def mouse_scroll(self, window, xoffset, yoffset):
				action = mj.mjtMouse.mjMOUSE_ZOOM
				mj.mjv_moveCamera(
						self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam
				)

		def reset(self):
				mj.mj_resetData(self.model, self.data)
				mj.mj_kinematics(self.model, self.data)
				mj.mj_forward(self.model, self.data)

				self.init_controller(self.model, self.data)

		def init_controller(self, model, data):
				FIELD_LENGTH, FIELD_WIDTH = envProps.OBSERVATION_SPACE[os.environ['SOCCER_DIMS']]['FIELD_DIMENSIONS']
				FIELD_LENGTH, FIELD_WIDTH = FIELD_LENGTH, FIELD_WIDTH
				spawn_x_limit = FIELD_LENGTH - 3
				spawn_y_limit = FIELD_WIDTH - 3

				if envProps.RANDOMIZE_INITIAL_POSITIONS_PLAYERS == True:
						random_position = (
								np.random.uniform(-FIELD_LENGTH, FIELD_LENGTH),
								np.random.uniform(-FIELD_WIDTH, FIELD_WIDTH),
								envProps.RADIUS_PLAYER,
						)
						self.env.player.set_position(data, random_position)

				if envProps.RANDOMIZE_INITIAL_POSITIONS_BALL == True:
						random_position = (
								np.random.uniform(-spawn_x_limit, spawn_x_limit),
								np.random.uniform(-spawn_y_limit, spawn_y_limit),
								envProps.RADIUS_BALL,
						)
						self.env.ball.set_position(data, random_position)

				# Spawn obstacles in between the player and the ball in a line perpendicular to the line connecting the player and the ball
				player_position = self.env.player.get_position(data)
				ball_position = self.env.ball.get_position(data)

				midpoint = ((player_position[0] + ball_position[0]) / 2, (player_position[1] + ball_position[1]) / 2)
				angle = np.arctan2(ball_position[1] - player_position[1], ball_position[0] - player_position[0]) + np.pi / 2

				gap_inbetween = envProps.RADIUS_PLAYER * 15
				self.env.away_player_1.set_position(data, (midpoint[0], midpoint[1], envProps.RADIUS_PLAYER))
				self.env.away_player_2.set_position(data, (midpoint[0] - gap_inbetween * np.cos(angle), midpoint[1] - gap_inbetween * np.sin(angle), envProps.RADIUS_PLAYER))
				self.env.away_player_3.set_position(data, (midpoint[0] + gap_inbetween * np.cos(angle), midpoint[1] + gap_inbetween * np.sin(angle), envProps.RADIUS_PLAYER))

				self.observation = self.env.get_observation_space(data)
				self.nsteps = 0
				self.done = False
				self.score  = 0

				self.logger.write("Initial player position: {0}, ball position: {1}".format(self.env.player.get_position(data), self.env.ball.get_position(data)))

		def step(self, data):
			action = self.env.player.ai.choose_action(self.observation)
			new_observation, reward, self.done, info = self.env.step(data, action)
			self.env.player.ai.remember(self.observation, action, reward, new_observation, self.done)
			self.score += reward
			self.observation = new_observation

		def run(self, episode_count):
				self.logger.write("START: Episode count: {0}".format(episode_count))
				
				self.reset()

				start_time = time.time()
				episode_actor_loss = 0
				episode_critic_loss = 0
				n_trains = 0

				# The simulation runs in 60 fps
				while not glfw.window_should_close(self.window):
						time_prev = self.data.time

						while self.data.time - time_prev < 1 / self.frames_per_second:
							mj.mj_step(self.model, self.data)
						
						self.step(self.data)

						if (self.env.player.ai.memory.mem_cntr % self.env.player.ai.batch_size == 0 \
							and self.env.player.ai.memory.mem_cntr > 1000):
							n_trains += 1
							actor_loss, critic_loss = self.env.player.ai.learn()
							episode_actor_loss += actor_loss
							episode_critic_loss += critic_loss

						if self.done or (self.data.time > self.sim_length):
							break

						mj.mjv_updateScene(
								self.model,
								self.data,
								self.opt,
								None,
								self.cam,
								mj.mjtCatBit.mjCAT_ALL.value,
								self.scene,
						)

						viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
						viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

						mj.mjr_render(viewport, self.scene, self.context)

						glfw.swap_buffers(self.window)
						glfw.poll_events()
				
				self.env.player.ai.save_models()

				self.logger.write("Episode count {0}, Actor loss: {1}, Critic loss: {2}".format(episode_count, episode_actor_loss, episode_critic_loss))
				self.logger.write("Final player position: {0}, ball position: {1}".format(self.env.player.get_position(self.data), self.env.ball.get_position(self.data)))
				self.logger.write("END: Episode count {0}, Episode length: {1}, Score: {2}".format(episode_count, self.data.time ,self.score))

		def stop(self):
				glfw.terminate()
