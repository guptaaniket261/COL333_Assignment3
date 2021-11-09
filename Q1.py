import numpy as np

def goLeft(r,c):
    bannedLeft = [(0,2),(1,2),(3,1),(3,3),(4,1),(4,3)]
    if (r,c) in bannedLeft:
        return False
    return True

def goRight(r,c):
    bannedRight = [(0,1),(1,1),(3,0),(3,2),(4,0),(4,2)]
    if (r,c) in bannedRight:
        return False
    return True


class Taxi_MDP:

  """
    5x5 grid 

  +---------+
  |R: | : :G|
  |T: | : : |
  | : : : : |
  | | : | : |
  |Y| : |B: |
  +---------+

  """
        
  # Taxi Start Position, T = (x,y)
  # Passenger Initial position, P = (a,b)
  # Passenger Destination, D = (p,q)
    
  def __init__(self, T = (1,0), P_pos = (0,0), D = (0,4)):
    
    p_locs = [(0,0),(0,4),(4,0),(4,3)]  # passenger can be at one of these locations
    num_actions = 6
    num_states = 500
    num_rows = 5
    num_cols = 5
    P = dict()
    if P_pos not in p_locs or D not in p_locs:
        raise 0
    self.state = (T[0])*100 + (T[1])*20 + (p_locs.index(P_pos))*4 + p_locs.index(D)
    for state in range(num_states):
        P[state] = {action: [] for action in range(num_actions)}
    
    """
    
    P = {   0: {0:[], 1:[],...,5:[]},
            1: {0:[], 1:[],...,5:[]},
            .
            .
            .
          499: {0:[], 1:[],...,5:[]}
        }
    
    """

    # actions = 0: South
    #           1: North
    #           2: East
    #           3: West
    #           4: Pickup
    #           5: Putdown
    
    for row in range(num_rows):
        for col in range(num_cols):
            for p_loc in range(len(p_locs) + 1):  # including when passenger is in taxi
                for d_loc in range(len(p_locs)):
                    state = row*100 + col*20 + p_loc*4 + d_loc
                    for action in range(num_actions):
                        n_p_loc, reward = p_loc, -1
                        done = False
                        taxi_loc = (row,col)
                        if action < 4:
                            n_row0, n_row1, n_row2, n_row3 = min(row+1,num_rows-1), max(row-1, 0), row, row
                            n_col0, n_col1, n_col2, n_col3 = col, col, col, col
                            if goRight(row,col):
                                n_col2 = min(col+1,num_cols-1)
                            if goLeft(row,col):
                                n_col3 = max(col-1,0)
                            new_state0 = n_row0 * 100 + n_col0 * 20 + n_p_loc * 4 + d_loc
                            new_state1 = n_row1 * 100 + n_col1 * 20 + n_p_loc * 4 + d_loc
                            new_state2 = n_row2 * 100 + n_col2 * 20 + n_p_loc * 4 + d_loc
                            new_state3 = n_row3 * 100 + n_col3 * 20 + n_p_loc * 4 + d_loc
                            same_state = False
                            same_state_prob = 0
                            if action == 0:
                                if new_state0 == state:
                                    same_state = True
                                    same_state_prob += 0.85
                                else:
                                    P[state][action].append((0.85,new_state0,reward,done))
                                if new_state1 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state1,reward,done))
                                if new_state2 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state2,reward,done))
                                if new_state3 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state3,reward,done))
                                if same_state:
                                    P[state][action].append((same_state_prob,state,reward,done))

                            if action == 1:
                                if new_state1 == state:
                                    same_state = True
                                    same_state_prob += 0.85
                                else:
                                    P[state][action].append((0.85,new_state1,reward,done))
                                if new_state2 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state2,reward,done))
                                if new_state3 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state3,reward,done))
                                if new_state0 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state0,reward,done))
                                if same_state:
                                    P[state][action].append((same_state_prob,state,reward,done))

                            if action == 2:
                                if new_state2 == state:
                                    same_state = True
                                    same_state_prob += 0.85
                                else:
                                    P[state][action].append((0.85,new_state2,reward,done))
                                if new_state3 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state3,reward,done))
                                if new_state0 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state0,reward,done))
                                if new_state1 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state1,reward,done))
                                if same_state:
                                    P[state][action].append((same_state_prob,state,reward,done))

                            if action == 3:
                                if new_state3 == state:
                                    same_state = True
                                    same_state_prob += 0.85
                                else:
                                    P[state][action].append((0.85,new_state3,reward,done))
                                if new_state0 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state0,reward,done))
                                if new_state1 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state1,reward,done))
                                if new_state2 == state:
                                    same_state = True
                                    same_state_prob += 0.05
                                else:
                                    P[state][action].append((0.05,new_state2,reward,done))
                                if same_state:
                                    P[state][action].append((same_state_prob,state,reward,done))
                        else:
                            n_row, n_col = row, col 
                            if action == 4:
                                if p_loc < 4 and taxi_loc == p_locs[p_loc]:
                                    n_p_loc = 4
                                else:
                                    reward = -10
                            elif action == 5:
                                if p_loc == 4:
                                    ## passenger actually dropped only if taxi is at one of the depots ##
                                    ##       Two sub cases         ##
                                    ## 1. Dropped at destination 
                                    ## 2. Dropped at some other depot
                                    ## Otherwise location of passenger doesn't change and reward remains -1
                                    if taxi_loc == D:
                                        n_p_loc = p_locs.index(taxi_loc)
                                        reward = 20
                                        done = True
                                    elif taxi_loc in p_locs:
                                        n_p_loc = p_locs.index(taxi_loc)
                                        reward = -1
                                elif p_loc < 4 and taxi_loc == p_locs[p_loc]:
                                    # both at same location, so reward is still -1, but no drop could happen
                                    reward = -1
                                else:
                                    # taxi and passenger at different locations 
                                    reward = -10
                            new_state = n_row * 100 + n_col * 20 + n_p_loc * 4 + d_loc
                            P[state][action].append((1.0,new_state,reward,done))
    #print(P)
    self.env = P

  def step(self,action):
    transitions = self.env[self.state][action]
    n_states = len(transitions)
    probs = [100*i[0] for i in transitions]
    cum_probs = np.cumsum(probs)
    r = np.random.randint(low = 0, high = 100)
    for i in range(n_states):
        if r < cum_probs[i]:
            new_state = transitions[i][1] 
            self.state = transitions[i][1]
            new_state = new_state//20
            col = new_state%5
            row = new_state//5
            print(row,col)
            print(transitions[i])
            return transitions[i]


def main():
    a = Taxi_MDP()
    s = a.state
    s = s//20
    col = s%5
    row = s//5
    print(row,col)
    a.step(0)
    a.step(1)

if __name__ == "__main__":
    main()
                                
                            
                            
                    
 