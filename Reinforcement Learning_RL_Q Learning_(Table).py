# -*- coding: utf-8 -*-
"""
RL: Q-Learning (Table)

@author: Stefano Gioia
"""
                   
import gym    #gym environments

# Initialize environment
env = gym.make("MountainCar-v0") #OpenAI's gym, "MountainCar-v0" environment; objective: get the cart to the flag

#Starting observation state
#state0 = env.reset()  #state: position (along horizontal axis) and velocity
            

## Discretization (states) ##
# Ranges of state values: env.observation_space.high, env.observation_space.low
state1_slots = 10
state2_slots = 10
states_slots = [state1_slots, state2_slots]
slot_size = (env.observation_space.high - env.observation_space.low)/states_slots             


## Q-Learning parameters ##
# Update after action:
#q_new = (1 - learn_rate) * q + learn_rate * (reward + discount_fctr * max_new_state_q)
learn_rate = 0.2  # (0,1]
discount_fctr = 0.95 #for future rewards, [0,1]

episodes = 416 
#episodes: "n of games" : times of max fixed number of steps tried (not said obj. reached): learning to reach objective from different conditions and back-propagation

## Exploration-exploitation settings 
# Gradual decrease of exploration (and increase of exploitation) with comparison betw. a decreasing "eps" and rand number over episodes
eps = 1 
# Range of episodes for eps variation
start_eps_decrease = 0 #episode starts from 0, so by setting 0: decrease from the beginning
end_eps_decrease = episodes//2  #eps to zero (negative is also ok) in about half of episodes
eps_decr_val = eps/(end_eps_decrease - start_eps_decrease) 


## Initialize q table ("rewards table") randomly.Here non positive rewards ("punishments"), top reward for objective is 0
import numpy as np
q_table = np.random.uniform(low=-2, high=0, size=(states_slots + [env.action_space.n]))

#To visually check some episodes during run
episode_check = [413]

## Function to get discrete states ##      
def discr_state(STATE):
    discrete_state = (STATE - env.observation_space.low)/slot_size
    return tuple(discrete_state.astype(np.int))  #return tuple of integers

## Run ##
for episode in range(episodes):
    
    state_d = discr_state(env.reset())
    ended = False   #flag for the achievement of the objective 
     
  
    while not ended:
        
        if np.random.random() > eps:
            
            action = np.argmax(q_table[state_d]) #exploitation
            
        else:
            
            action = np.random.randint(0, env.action_space.n) #exploration
            

        new_state, reward, ended, other = env.step(action) 
        
        if episode in episode_check:
           env.render()
        
        new_state_d = discr_state(new_state)
                              
        
        if new_state[0] < env.goal_position: #if not ended: #simpler implementation
            
            # Maximum Q value for new state (actions compared)
            max_new_state_q = np.max(q_table[new_state_d])

            # Current Q value for chosen action
            q = q_table[state_d + (action,)]

            # Q value update for state and performed action 
            q_new = (1 - learn_rate) * q + learn_rate * (reward + discount_fctr  * max_new_state_q)

            # Q table update
            q_table[state_d + (action,)] = q_new
            
        elif new_state[0] >= env.goal_position:  # Ended for any reason, check objective

            q_table[state_d + (action,)] = 0 # If objective reached: q_value is the best possible reward, here 0

            print("Objective reached at episode " + str(episode) )
        
        # State update
        state_d = new_state_d
        

    # eps decrease if episode number is within specified episodes' range
    if  end_eps_decrease >= episode >= start_eps_decrease : 
        eps -= eps_decr_val 

env.close()
