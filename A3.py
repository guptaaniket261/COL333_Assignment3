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

def encode(row_p, col_p, row_t, col_t, inside_taxi):
	return (10000*row_p + 1000*col_p + 100*row_t + 10*col_t+ inside_taxi)

def decode(state):
	# row_p, col_p, row_t, col_t, inside_taxi
	sol = [0,0,0,0,0]
	for i in range(5):
		sol[4-i] = (state%10)
		state = state//10
	return sol


class State:

	def __init__(self, row_p, col_p, row_t, col_t, inside_taxi):
		self.row_p = row_p
		self.col_p = col_p
		self.row_t = row_t
		self.col_t = col_t
		self.inside_taxi = inside_taxi


class Taxi_MDP:

	# actions = 0: South
    #           1: North
    #           2: East
    #           3: West
    #           4: Pickup
    #           5: Putdown

	def __init__(self, T = (1,0), P_pos = (0,0), D = (0,4)):
		self.p_locs = [(0,0),(0,4),(4,0),(4,3)]
		if P_pos not in self.p_locs or D not in self.p_locs:
			raise 0

		self.num_rows = 5
		self.num_cols = 5
		self.currState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)
		self.startState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)	
		self.destState = encode(D[0], D[1], D[0], D[1], 0)
		self.A = {"S": 0, "N": 1, "E": 2, "W": 3, "Pickup": 4, "Putdown": 5}
		self.states, self.state_index, self.index_state = self.getStates()
		self.P, self.R = self.getModels()

	def getStates(self):
		states = []
		state_index = {}
		index_state = {}
		counter = 0
		for row_p in range(self.num_rows):
			for col_p in range(self.num_cols):
				state = encode(row_p, col_p, row_p, col_p, 1)
				states.append(state)
				state_index[state] = counter
				index_state[counter] = state
				counter += 1
				for row_t in range(self.num_rows):
					for col_t in range(self.num_cols):
						state = encode(row_p, col_p, row_t, col_t, 0)
						states.append(state)
						state_index[state] = counter
						index_state[counter] = state
						counter += 1
		return states, state_index,index_state

	def getModels(self):
		P = dict()
		R = dict()
		states = self.states
		
		num_actions = len(self.A)
		num_rows = self.num_rows
		num_cols = self.num_cols
		for state in states:
			P[state] = {action: [] for action in range(num_actions)}
			R[state] = {action: 0 for action in range(num_actions)}
		for state in states:
			if state == self.destState:
				continue
			row_p, col_p, row_t, col_t, inside_taxi = decode(state)
			n_row0, n_row1, n_row2, n_row3 = min(row_t+1,num_rows-1), max(row_t-1, 0), row_t, row_t
			n_col0, n_col1, n_col2, n_col3 = col_t, col_t, col_t, col_t
			if goRight(row_t, col_t):
				n_col2 = min(col_t+1,num_cols-1)
			if goLeft(row_t,col_t):
				n_col3 = max(col_t-1,0)
			
			new_state0 = encode(n_row0, n_col0, n_row0, n_col0, inside_taxi)
			new_state1 = encode(n_row1, n_col1, n_row1, n_col1, inside_taxi)
			new_state2 = encode(n_row2, n_col2, n_row2, n_col2, inside_taxi)
			new_state3 = encode(n_row3, n_col3, n_row3, n_col3, inside_taxi)
		
			for action in range(num_actions):
				if action < 4:
					same_state = False
					same_state_prob = 0
					R[state][action] = -1
					if action == 0:
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if same_state:
							P[state][action].append((same_state_prob,state))
						
					if action == 1:
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if same_state:
							P[state][action].append((same_state_prob,state))

					if action == 2:
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if same_state:
							P[state][action].append((same_state_prob,state))

					if action == 3:
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if same_state:
							P[state][action].append((same_state_prob,state))

				else:
					if not(row_t == row_p and col_t == col_p):
						R[state][action] = -10
						P[state][action].append((1.0, state))
						continue

					if action == 4:
						n_state = encode(row_p, col_p, row_t, col_t, 1)
					else:
						n_state = encode(row_p, col_p, row_t, col_t, 0)
					P[state][action].append((1.0, n_state))
					if n_state == self.destState:
						R[state][action] = 20
					else:
						R[state][action] = -1
		return P, R

	def step(self, action):
		if self.currState == self.destState:
			print("Destination Reached...")
			return self.currState, 0

		transitions = self.P[self.currState][action]
		n_states = len(transitions)
		probs = [100*i[0] for i in transitions]
		cum_probs = np.cumsum(probs)
		r = np.random.randint(low = 0, high = 100)
		for i in range(n_states):
			if r < cum_probs[i]:
				new_state = transitions[i][1] 
				self.currState = transitions[i][1]
				# print(transitions)
				print(decode(new_state))
				return transitions[i], self.R[self.currState][action]

	def get_rand_start(self):
		r = np.random.randint(low = 0, high = 4)
		row_p, col_p = self.p_locs[r]
		row_t = np.random.randint(low = 0, high = 4)
		col_t = np.random.randint(low = 0, high = 4)
		while row_t == row_p and col_p == col_t:
			row_t = np.random.randint(low = 0, high = 4)
			col_t = np.random.randint(low = 0, high = 4)	
		inside_taxi = 0
		start_state = encode(row_p, col_p, row_t, col_t, inside_taxi)
		return start_state



class Policy:
	def __init__(self):
		return

	def value_iteration(self, MDP, epsilon, discount):
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		delta = 0
		converged = False
		policy = {state : -1 for state in MDP.states}
		while(not converged):
			for state in MDP.states:
				old_utilities[state] = new_utilities[state]
			delta = 0
			for state in MDP.states:
				new_utilities[state] = -np.inf
				for a in MDP.A:
					action = MDP.A[a]
					temp_utility = 0
					for prob, neighbour in MDP.P[state][action]:
						temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
					new_utilities[state] = max(temp_utility, new_utilities[state])
				delta = max(delta, abs(new_utilities[state] - old_utilities[state]))
			if delta < epsilon:
				converged = True
		
		
		for state in MDP.states:
			optimal_utility = -np.inf
			for a in MDP.A:   ## MDP.A = {"S": 0, "N": 1, "E": 2, "W": 3, "Pickup": 4, "Putdown": 5}
				action = MDP.A[a]
				temp_utility = 0
				for prob, neighbour in MDP.P[state][action]:
					temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
				if temp_utility > optimal_utility:
					optimal_utility = temp_utility
					policy[state] = action
		return policy


	def policy_evaluation_linear(self, MDP, policy, discount):
		num_states = len(MDP.states)
		equations_LHS = [ [] for i in range(num_states) ]
		equations_RHS = [0 for i in range(num_states)]
		for state in MDP.states:
			action = policy[state]
			equation = [ 0 for j in range(num_states)]
			equations[state_index[state]] = 1
			for prob, next_state in MDP.P[state][action]:
				equation[state_index[next_state]] = - prob * discount
			equations_LHS[state_index[state]] = equation
			equations_RHS[state_index[state]] = MDP.R[state][action]
		vals = np.linalg.solve(equations_LHS,equations_RHS)
		utilities = {index_state[i]: v for i,v in enumerate(vals)}
		return utilities

	def policy_evaluation_iterative(self, MDP, policy, discount, epsilon = 0.001):
		converged = False
		delta = 0
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		while(not converged):
			delta = 0
			old_utilities = new_utilities
			for state in MDP.states:
				action = policy[state]
				temp_utility = 0
				for prob, neighbour in MDP.P[state][action]:
					temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
				new_utilities[state] = temp_utility
				delta = max(delta, new_utilities-old_utilities)
			if delta < epsilon:
				converged = True
		return new_utilities

	def policy_iteration(self, MDP, policy, discount, epsilon = 0.001, iterative = 0):
		utilities = {state : 0 for state in MDP.states}
		improved_policy = policy
		converged = False
		iteration = 0
		while(not converged):
		
			policy = improved_policy
			iteration += 1
			print("Iteration: " + str(iteration))

			## Policy Evaluation ##
			if iterative:
				utilities = policy_evaluation_iterative(MDP,policy,discount,epsilon)
			else:
				utilities = policy_evaluation_linear(MDP,policy,discount)

			## Policy Improvement ##
			for state in MDP.states:
				optimal_utility = -np.inf
				for a in MDP.A:
					action = MDP.A[a]
					temp_utility = 0
					for prob, neighbour in MDP.P[state][action]:
						temp_utility += prob*(MDP.R[state][action] + discount*utilities[neighbour])
					if temp_utility > optimal_utility:
						optimal_utility = temp_utility
						improved_policy[state] = action
			
			if policy == improved_policy:
				converged = True			
		
		return utilities,improved_policy

		def epsilon_greedy(action,epsilon, iter_num, decaying_epsilon):
			r = np.random.rand()
			if decaying_epsilon:
				epsilon = epsilon/iter_num
			if r < epsilon:
				## Exploration
				action = np.random.randint(low=0,high=6)
			return action

		def q_learning(self,MDP,policy,alpha,discount,epsilon,num_episodes,decaying_epsilon = False):
			q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
			iteration = 1
			for episode in range(num_episodes):
				state = MDP.get_rand_start() 
				done = False
				while not done:
					action = epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
					transition, reward = MDP.step(action)   ## transition = prob, next_state
					next_state = transition[1]
					next_opt_action = max(MDP.P[next_state], key= lambda action: MDP.P[next_state][action])
					td_update_sample = reward + discount * q_table[next_state][next_opt_action] 
					q_table[state][action] = (1-alpha) * q_table[state][action] +  alpha * td_update_sample
					iteration += 1
					if next_state == MDP.destState:
						done = True
					state = next_state

			learned_policy = {state : -1 for state in MDP.states}
			for state in MDP.states:
				learned_policy[state] = max(MDP.q_table[state], key= lambda action: MDP.q_table[state][action])
			return learned_policy


		def SARSA(self,MDP,policy,alpha,discount,epsilon,num_episodes,decaying_epsilon = False):
			q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
			iteration = 1
			for episode in range(num_episodes):
				state = MDP.get_rand_start() 
				done = False
				while not done:
					action = epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
					transition, reward = MDP.step(action)   ## transition = prob, next_state
					next_state = transition[1]
					next_action = policy[next_state]
					td_update_sample = reward + discount * q_table[next_state][next_action] 
					q_table[state][action] = (1-alpha) * q_table[state][action] +  alpha * td_update_sample
					iteration += 1
					if next_state == MDP.destState:
						done = True
					state = next_state

			learned_policy = {state : -1 for state in MDP.states}
			for state in MDP.states:
				learned_policy[state] = max(MDP.q_table[state], key= lambda action: MDP.q_table[state][action])
			return learned_policy
		

def main():
	a = Taxi_MDP()
	a.step(0)
	a.step(1)

if __name__ == "__main__":
    main()
					
					










		


		

