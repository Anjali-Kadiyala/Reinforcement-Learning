import numpy as np

# Observation Space
OBSERVATION_SPACE = {
    'XS': {
        'FIELD_DIMENSIONS': [6, 4.125],
        'GOAL_HOME_TOP': [-5.7375, 0.625],
        'GOAL_AWAY_TOP': [5.7375, 0.625],
        'GOAL_HOME_BOTTOM': [-5.7375, -0.625],
        'GOAL_AWAY_BOTTOM': [5.7375, -0.625],
    },
    'S': {
        'FIELD_DIMENSIONS': [12, 8.25],
        'GOAL_HOME_TOP': [-11.475, 1.25],
        'GOAL_AWAY_TOP': [11.475, 1.25],
        'GOAL_HOME_BOTTOM': [-11.475, -1.25],
        'GOAL_AWAY_BOTTOM': [11.475, -1.25],
    },
    'M': {
        'FIELD_DIMENSIONS': [24, 16.5],
        'GOAL_HOME_TOP': [-22.95, 2.5],
        'GOAL_AWAY_TOP': [22.95, 2.5],
        'GOAL_HOME_BOTTOM': [-22.95, -2.5],
        'GOAL_AWAY_BOTTOM': [22.95, -2.5],
    },
    'L': {
        'FIELD_DIMENSIONS': [48, 33],
        'GOAL_HOME_TOP': [-45.9, 5],
        'GOAL_AWAY_TOP': [45.9, 5],
        'GOAL_HOME_BOTTOM': [-45.9, -5],
        'GOAL_AWAY_BOTTOM': [45.9, -5],
    },
}

RANDOMIZE_INITIAL_POSITIONS_PLAYERS = True
RANDOMIZE_INITIAL_POSITIONS_BALL = True

RADIUS_PLAYER = 0.365
RADIUS_BALL = 0.215

SPEED_MULTIPLIER = 5

# Sphero Bolt Specs - MuJoCo
ENV_BOLT_MASS = 200
ENV_BOLT_MIN_ROTATION = -np.pi
ENV_BOLT_MAX_ROTATION = np.pi
ENV_BOLT_MIN_SPEED = 0 * SPEED_MULTIPLIER
ENV_BOLT_MAX_SPEED = 1 * SPEED_MULTIPLIER

# Soccer Ball Specs - MuJoCo
ENV_BALL_MASS = 2.77
ENV_BALL_MIN_SPEED = 0
ENV_BALL_MAX_SPEED = (ENV_BOLT_MASS * ENV_BOLT_MAX_SPEED * SPEED_MULTIPLIER) / (ENV_BALL_MASS) # assuming elastic collision 


TEAM_HOME = "HOME_TEAM"  # blue players
TEAM_AWAY = "AWAY_TEAM"  # red players
BALL = "ball"

ENV_1A_0D_0K = "Environment_1A_0D_0K"

