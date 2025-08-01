import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
from game_env import game_env, conf         

# To make the states hashable for the agent
def state_encoder(obs):
    # Buckets for ball
    col_w = 2
    row_h = 2
    ball_col = obs["ball_x"] // col_w
    ball_row = obs["ball_y"] // row_h
    
    # Speed
    ball_vx = obs["ball_vx"] + 2                # -2,-1 ... 2 :--> 0 ...4 
    ball_vy = 0 if obs["ball_vy"]<0 else 1    # 0 = up, 1 = down 
    
    # paddle offset (+6, -6)
    paddle_centre = obs["paddle_x"][2]               
    paddle_offset = paddle_centre // col_w - ball_col
    
    # Brick count buckets: For efficancy, we set 15 as min bricks. this shrinks the size of distinct states for Q
    bricks_left = min(obs["num_bricks"], 15)   
    
    return (ball_col, ball_row, ball_vx, ball_vy, paddle_offset, bricks_left)                                   

class MC_agent:
    def __init__(self, action_space, state_encoder, eps, gamma, alpha):
        self.actions = action_space
        self.state_enc = state_encoder
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        
        # Q(a) and N 
        self.Q = defaultdict(float) # ((state, action) : G)
        self.N = defaultdict(int) # ((state, action): N)
        
        # Episode Buffers
        self.S = list() # states
        self.A = list() # actions       
        self.R = list() # rewards
        
    def start_episode(self, init_obs):
        self.S = list() 
        self.A = list()    
        self.R = list() 
        
        # inital state
        s_0 = self.state_enc(init_obs)
        self.S.append(s_0)
        
        # Inital action - random chosen by epsilon greedy
        a_0 = self.epsilon_greedy(s_0, self.eps)
        self.A.append(a_0)
        
        return a_0
    
    def step(self, obs, reward, done):
        # Record Reward
        self.R.append(reward)
        
        # Finished or new obs
        if done:
            return None
        else: 
            next_s = self.state_enc(obs)
            self.S.append(next_s)
            next_a = self.epsilon_greedy(next_s, self.eps)
            self.A.append(next_a)
            return next_a
            

    def end_episode(self):
        # walk through steps backwards and calc Q
        G = 0
        for t in reversed(range(len(self.S))):
            G = self.gamma * G + self.R[t]
            self.update_Q(self.S[t], self.A[t], G)
        
    
    
    def epsilon_greedy(self, state, eps):
        u = random.uniform(0,1) 
        if u > eps:
            # Exploitation: Greedy Choice
            q_list = [self.Q[(state, a)] for a in self.actions]
            max_q = max(q_list)
            greedy_actions = [a 
                              for a, q in zip(self.actions, q_list) 
                              if q == max_q]
            next_action = random.choice(greedy_actions) # in case there are more then one   
        else:
            # Exploration: Random Choice
            next_action = random.choice(self.actions)
        
        return next_action 
            
     
    def update_Q(self, state, action, G):
        # Chose learning rate or sample avarage
        if self.alpha is None:
            self.N[(state, action)] += 1
            n = self.N[(state, action)]
            alpha_t = 1/ n    # Sample Avrage
        else:
            alpha_t = self.alpha  # Learning rate
                 
        error = G - self.Q[(state, action)]
        self.Q[(state, action)] += alpha_t * error
       

    
    def policy(self, state):
        # Here we return the greedy choice for every state to refelct the optimal policy
        q_list = [self.Q[(state, a)] for a in self.actions]
        max_q = max(q_list)
        greedy_actions = [a 
                        for a, q in zip(self.actions, q_list) 
                        if q == max_q]
        next_action = random.choice(greedy_actions)  
        return next_action     
    
        
        
        


