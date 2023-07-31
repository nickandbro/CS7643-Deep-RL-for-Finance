from stable_baselines3 import A2C
import numpy as np

def train_a2c(env, timesteps):
    print("Training A2C model...")
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    #env.render()
    print("Training completed.")
    return model

def test_a2c(model, env):
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _  = env.step(action)
        env.render()
        if done:
            print("info", info)
            break
    print("Done testing")