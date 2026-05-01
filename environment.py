import gymnasium as gym


#boilerplate code from Open AI Gymnasium official website, modified for pendulum env
env = gym.make("Pendulum-v1", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]

episode_over = False
total_reward = 0


while not episode_over:
  action = env.action_space.sample()
#   action = [0]
  print(f"Action: {action}")
#   print(f"Action Type: {type(action)}")

  observation,reward,terminated,truncated,info = env.step(action)

  total_reward+=reward
  episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")

env.close()