import os
from datetime import datetime

class Logger:
	def __new__(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Logger, cls).__new__(cls)
		return cls.instance

	def __init__(self):
		# Path to Logs folder
		dirname = os.path.dirname(__file__)
		self.path_logs = os.path.join(dirname + "/Logs/")

		# Read file names from folder
		file_names = os.listdir(self.path_logs)

		if len(file_names) == 0:
			self.current_log_number = -1
			self.create_new_log()
		else:
			self.current_log = self.path_logs + file_names[-1]
			self.current_log_name = file_names[-1]
			self.current_log_number = int(self.current_log_name.split('_')[1].split('.')[0])

	def write(self, message):
		with open(self.current_log, 'a') as file:
			file.write(str(datetime.now()) + ': ' + str(message) + '\n')

		with open(self.current_log, 'r') as file:
			if (len(file.readlines()) > 1000):
				self.create_new_log()

	def create_new_log(self):
		self.current_log_number += 1
		self.current_log_name = 'log_' + str(self.current_log_number) + '.txt'
		self.current_log = self.path_logs + self.current_log_name

		self.write('Created new log file')
		