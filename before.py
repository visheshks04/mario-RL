import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

print(SIMPLE_MOVEMENT)

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print(env.action_space)

done = True

#Looping through each frame in the game
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

    print(f"state shape: {state.shape}")
    print(f"reward: {reward}")
    print(f"info: {info}")

    env.render()
    # time.sleep(1)

env.close()