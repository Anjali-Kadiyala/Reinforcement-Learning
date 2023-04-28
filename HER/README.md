## Hindsight Experience Replay (HER)

HDF is an RL technique to improve learning efficiency of the agents. The idea behind HER is to use failed experiences to train the agent. Unlike in traditional RL, where if an agent fails to achieve its goals, whatever experience/ actions taken so far are discarded, in HER these failed experiences are replayed with different goals (which is more similar to how humans learn). This improves the models performance and efficiency. 

#### Here are the general steps for how HER works:

1. The agent takes actions in the environment to try to achieve a specific goal.
2. If the agent fails to achieve the goal, HER stores the experience in a replay buffer, along with the original goal that the agent was trying to achieve.
3. HER then replays the experience with a different goal, which is selected randomly from the previous experiences stored in the replay buffer.
4. The agent then learns from the replayed experience and updates its policy to improve its performance in achieving the new goal.
5. The process is repeated multiple times with different goals until the agent has learned to achieve a wide range of goals, even those it did not initially encounter during training.

The key benefit of HER is that it allows the agent to learn from failed experiences, which would otherwise be discarded. By reusing these experiences with different goals, the agent can better understand the dynamics of the environment and improve its decision-making process. This makes the agent more robust and adaptive to different situations and goals.

#### Example

Say an agent that is trying to learn how to navigate a maze. If the agent fails to reach the goal, HER would take that failed experience and replay it with a different goal, such as reaching a different point in the maze. By doing this, the agent learns from the failed experience and can better adapt to similar situations in the future.


Reference repo for the reimplementation: https://github.com/alirezakazemipour/DDPG-HER

In this reimplementation, I have used HER along with Deep Deterministic Policy Gradient (DDPG). 
