
import numpy as np
import torch
from collections import deque,namedtuple
import matplotlib.pyplot as plt
from Agent import DDPGAgent
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

brain_name = env.brain_names[0]
brain=env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))


agent_ddpg = DDPGAgent(state_size, action_size)

def ddpg(agent, n_episodes=300, max_t=1000, update_every=3, batch_size=50, policy_noise=0.2):
    step = 0
    scores = []
    scores_window = deque(maxlen=100)

    for i in range(n_episodes):
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        for t in range(max_t):
            action = agent.act(state, policy_noise)
            env_info = env.step(action)[brain_name]
            state_next = env_info.vector_observations
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.cache(state, action, reward, state_next, done)
            step += 1
            if (len(agent.memory) >= batch_size) and (step % update_every) == 0:
                experiences = agent.recall(batch_size)
                agent.learn(experiences)
                agent.sync_target()
            state = state_next
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
        # if i % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

    #torch.save(agent.actor.state_dict(),'checkpoint_actor.pth')
    return scores

scores = ddpg(agent_ddpg)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()