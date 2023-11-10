import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class Graph:
	def __init__(self, source):
		style.use('fivethirtyeight')
		self.source = source

		self.fig = plt.figure()
		self.ax1 = self.fig.add_subplot(1,1,1)

	def animate(self):
		data = open(self.source, 'r').read()