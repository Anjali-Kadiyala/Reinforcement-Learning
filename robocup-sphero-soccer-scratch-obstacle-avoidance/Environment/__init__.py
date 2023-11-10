import os

from Environment.Simulation import Simulation
from Environment.Environment_1A_0D_0K import Environment_1A_0D_0K

def make():
#     match os.environ['SOCCER_ENV']:
#         case '1A_0D_0K':
#             env = Environment_1A_0D_0K()
#         case _:
#             raise NotImplementedError

    env = Environment_1A_0D_0K()

    simulation = Simulation(env)

    return simulation
