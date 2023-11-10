import mujoco as mj
from Utilities.Logger import Logger

class Ball:
	def __init__(self, model, name):
		self.name = name

		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)

		logger = Logger()
		logger.write("Ball {0} initialized. id_body {1} id_geom {2} id_joint {3}".format(self.name, self.id_body, self.id_geom, self.id_joint))

	def get_position(self, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def set_position(self, data, position):
		data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position

	def get_velocity(self, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]

	def stop(self, data):
		data.qvel[self.id_joint * 6: self.id_joint * 6 + 6] = [0, 0, 0, 0, 0, 0]