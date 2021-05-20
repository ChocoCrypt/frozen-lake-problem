import gym
import numpy as np

env = gym.make("FrozenLake-v0")
env.render()


Q = np.zeros([env.observation_space.n , env.action_space.n])
alfa = 0.5
gamma = 0.99
for i in range(0,10000):
    estado = env.reset()
    finalizado = False
    while(not finalizado):
        action = env.action_space.sample()
        nuevo_estado ,  reward , finalizado , info = env.step(action)
        anterior  = Q[estado , action]
        nuevo_max = np.max(Q[nuevo_estado,  :])
        nuevo_valor = (1-alfa) * anterior  + alfa*(reward + gamma*nuevo_max)
        Q[estado, action] = nuevo_valor
        print(Q)
        print("\n")
        estado = nuevo_estado

wins = 0
for i in range(0,1000):
    estado = env.reset()
    finalizado = False
    while(not finalizado):
        action = np.argmax(Q[estado,:])
        nuevo_estado ,  reward , finalizado , info = env.step(action)
        if(reward == 1):
            wins += 1
        anterior  = Q[estado , action]
        nuevo_max = np.max(Q[nuevo_estado,  :])
        nuevo_valor = (1-alfa) * anterior  + alfa*(reward + gamma*nuevo_max)
        Q[estado, action] = nuevo_valor
        estado = nuevo_estado
        env.render()

print(f"winning rate: {wins/1000} ")





