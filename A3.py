import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def goLeft(r,c,bigger_grid):
	bannedLeft = [(0,2),(1,2),(3,1),(3,3),(4,1),(4,3)]
	if bigger_grid:
		bannedLeft = [ (0,3),(1,3),(2,3),(3,3),(0,8),(1,8),(2,8),(3,8),(2,6),(3,6),(4,6),(5,6),(6,1),(7,1),(8,1),(9,1),(6,4),(7,4),(8,4),(9,4),(6,8),(7,8),(8,8),(9,8)  ]
	if (r,c) in bannedLeft:
		return False
	return True

def goRight(r,c,bigger_grid):
	bannedRight = [(0,1),(1,1),(3,0),(3,2),(4,0),(4,2)]
	if bigger_grid:
		bannedRight = [(0,2),(1,2),(2,2),(3,2),(0,7),(1,7),(2,7),(3,7),(2,5),(3,5),(4,5),(5,5),(6,0),(7,0),(8,0),(9,0),(6,3),(7,3),(8,3),(9,3),(6,7),(7,7),(8,7),(9,7) ]
	if (r,c) in bannedRight:
		return False
	return True

def encode(row_p, col_p, row_t, col_t, inside_taxi):
	return (10000*row_p + 1000*col_p + 100*row_t + 10*col_t+ inside_taxi)

def decode(state):
	# row_p, col_p, row_t, col_t, inside_taxi
	sol = [0,0,0,0,0]
	for i in range(5):
		sol[4-i] = (state % 10)
		state = state//10
	return sol

def equal(dict1, dict2):
	for i in dict1:
		if dict1[i] != dict2[i]:
			return False
	return True

def getRandStart(destination,bigger_grid = False):
	n=5
	if bigger_grid:
		n=10
	row_t = np.random.randint(low = 0, high = n)
	col_t = np.random.randint(low = 0, high = n)
	p_locs = [(0,0),(0,4),(4,0),(4,3)]
	if bigger_grid:
		p_locs = [ (0,0),(0,5),(0,8),(3,3),(4,6),(8,0),(9,4),(9,9) ]	
	found = False
	loc = (0, 0)
	while( not found):
		r = np.random.randint(low = 0, high = len(p_locs))
		loc = p_locs[r]
		if loc!=destination and loc!=(row_t, col_t):
			break
	return loc, (row_t, col_t)


def evaluate(q_table, states, batch_size, discount, dest,bigger_grid = False):
	disc_sum_rewards = []
	learned_policy = {state : -1 for state in states}
	for state in states:
		learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
	for i in range(batch_size):
		p_loc, t_loc = getRandStart(dest)
		test_mdp = Taxi_MDP(t_loc, p_loc, D = dest, bigger_grid=bigger_grid) 
		rewards = test_mdp.simulate(learned_policy,verbose = False)
		disc_sum_reward = 0
		factor = 1
		for j in range(len(rewards)):
			disc_sum_reward += factor*rewards[j]
			factor *= discount
		disc_sum_rewards.append(disc_sum_reward)
	# print(disc_sum_reward)
	averaged_sum = sum(disc_sum_rewards) / batch_size
	return averaged_sum

class State:

	def __init__(self, row_p, col_p, row_t, col_t, inside_taxi):
		self.row_p = row_p
		self.col_p = col_p
		self.row_t = row_t
		self.col_t = col_t
		self.inside_taxi = inside_taxi


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
	# (0, 0) is at top left
  # Taxi Start Position, T = (x,y)
  # Passenger Initial position, P_pos = (a,b)
  # Passenger Destination, D = (p,q)

	# actions = 0: South
	#           1: North
	#           2: East
	#           3: West
	#           4: Pickup
	#           5: Putdown

	def __init__(self, T = (1,0), P_pos = (4,0), D = (0,4), bigger_grid = False):
		self.p_locs = [(0,0),(0,4),(4,0),(4,3)]
		self.num_rows = 5
		self.num_cols = 5
		self.bigger_grid = bigger_grid

		if bigger_grid:
			self.p_locs = [ (0,0),(0,5),(0,8),(3,3),(4,6),(8,0),(9,4),(9,9) ]
			self.num_rows = 10
			self.num_cols = 10
			if D == (0,4):
				D = (0,8)
			if P_pos == (4,0):
				P_pos = (0,0)

		if P_pos not in self.p_locs or D not in self.p_locs:
			print("Starting and ending position must be among the given depots")
			raise 0

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
		return states, state_index, index_state

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
			if goRight(row_t, col_t,self.bigger_grid):
				n_col2 = min(col_t+1,num_cols-1)
			if goLeft(row_t,col_t,self.bigger_grid):
				n_col3 = max(col_t-1,0)

			if (inside_taxi == 1):
				new_state0 = encode(n_row0, n_col0, n_row0, n_col0, inside_taxi)
				new_state1 = encode(n_row1, n_col1, n_row1, n_col1, inside_taxi)
				new_state2 = encode(n_row2, n_col2, n_row2, n_col2, inside_taxi)
				new_state3 = encode(n_row3, n_col3, n_row3, n_col3, inside_taxi)

			else:
				new_state0 = encode(row_p, col_p, n_row0, n_col0, inside_taxi)
				new_state1 = encode(row_p, col_p, n_row1, n_col1, inside_taxi)
				new_state2 = encode(row_p, col_p, n_row2, n_col2, inside_taxi)
				new_state3 = encode(row_p, col_p, n_row3, n_col3, inside_taxi)
		
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
						# TODO
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
				reward = self.R[self.currState][action]
				self.currState = transitions[i][1]
				return transitions[i], reward 

	def simulate(self, policy,verbose=True):
		steps = 0
		rewards = []
		A = {0 :"S", 1: "N", 2: "E", 3: "W", 4: "Pickup", 5: "Putdown"}
		self.currState = self.startState
		while(self.currState !=self.destState):
			if steps > 500:
				break
			steps += 1
			prev_state = self.currState
			prev_Px, prev_Py, prev_Tx, prev_Ty, inside_taxi = decode(prev_state)
			prev_P, prev_T = (prev_Px, prev_Py), (prev_Tx, prev_Ty)
			transition, reward = self.step(policy[prev_state])  ### transition = prob, next_state
			prob = transition[0]   ## CHANGE1 
			new_state = self.currState
			new_Px, new_Py, new_Tx, new_Ty, n_inside_taxi = decode(new_state)
			new_P , new_T = (new_Px, new_Py), (new_Tx, new_Ty)
			action_taken = A[policy[prev_state]]
			if verbose:
				print("CurrTaxi: {0}, CurrPass: {1}, Inside:{2}, action: {3}, prob: {4}, reward: {5}, NewTaxi: {6}, NewPass: {7}, Inside:{8}".format(prev_T, prev_P, inside_taxi, action_taken, prob, reward,new_T, new_P, n_inside_taxi ))
				#print("Taxi: {0}, Passenger: {1}, Inside_taxi:{2}, Action: {3}".format(prev_T, prev_P, inside_taxi, action_taken ))
			rewards.append(reward)
		return rewards


	def get_rand_start(self):
		row_pd,col_pd, row_td,col_td,inside_taxi_d = decode(self.destState)
		row_p,col_p = row_pd,col_pd
		while row_p == row_pd and col_p == col_pd:
			r = np.random.randint(low = 0, high = 4)
			row_p, col_p = self.p_locs[r]
		row_t = np.random.randint(low = 0, high = 5)
		col_t = np.random.randint(low = 0, high = 5)	
		inside_taxi = 0
		start_state = encode(row_p, col_p, row_t, col_t, inside_taxi)
		self.startState = start_state
		self.currState = start_state
		return start_state

class Policy:
	def __init__(self):
		return

	def value_iteration(self, MDP, epsilon, discount=0.9):
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		delta = 0
		converged = False
		policy = {state : -1 for state in MDP.states}
		iterations = 0
		iter_array = []
		max_norm_array = []
		while(not converged):
			iterations += 1
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
			max_norm_array.append(delta)
			iter_array.append(iterations)
			if delta < epsilon:
				converged = True
		plt.plot(iter_array, max_norm_array)
		plt.title("Value Iteration, epsilon = " + str(epsilon) + ", discount = "+str(discount) )
		plt.xlabel("Number of iterations")
		plt.ylabel("Max Norm value")
		plt.savefig("Val_iter_d_" + str(int(discount*100)))
		plt.show()
		
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

		
		print("Epsilon: {0}, Discount: {1} => Iterations: {2}".format(epsilon, discount,iterations))
		return policy

	def policy_evaluation_linear(self, MDP, policy, discount):
		num_states = len(MDP.states)
		equations_LHS = [ [] for i in range(num_states) ]
		equations_RHS = [0 for i in range(num_states)]
		for state in MDP.states:
			action = policy[state]
			equation = [ 0 for j in range(num_states)]
			equation[MDP.state_index[state]] = 1
			for prob, next_state in MDP.P[state][action]:
				if next_state == state:
					equation[MDP.state_index[state]] += (-prob*discount)
				else:
					equation[MDP.state_index[next_state]] = - prob * discount
			equations_LHS[MDP.state_index[state]] = equation
			equations_RHS[MDP.state_index[state]] = MDP.R[state][action]
		vals = np.linalg.solve(equations_LHS,equations_RHS)
		utilities = {MDP.index_state[i]: v for i,v in enumerate(vals)}
		return utilities

	def policy_evaluation_iterative(self, MDP, policy, discount, epsilon = 1e-8):
		converged = False
		delta = 0
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		while(not converged):
			delta = 0
			for state in MDP.states:
				old_utilities[state] = new_utilities[state]
			for state in MDP.states:
				action = policy[state]
				temp_utility = 0
				for prob, neighbour in MDP.P[state][action]:
					temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
				new_utilities[state] = temp_utility
				delta = max(delta, abs(new_utilities[state] - old_utilities[state]))
			if delta < epsilon:
				converged = True
		return new_utilities

	def policy_iteration(self, MDP, temp_policy, discount, epsilon = 1e-18, iterative = 0, calc_loss = False, opt_utilities = []):
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		delta = 0		
		improved_policy = temp_policy
		converged = False
		iteration = 0
		iterations, policy_losses = [], []
		print("\n")
		print("Policy Iteration Method with discount {0}".format(discount))
		while(not converged) and iteration < 500:
			policy = {state : improved_policy[state] for state in MDP.states}
			iteration += 1
			for state in MDP.states:
				old_utilities[state] = new_utilities[state]
			delta = 0
			## Policy Evaluation ##
			if iterative == 1:
				new_utilities = self.policy_evaluation_iterative(MDP, policy, discount, epsilon)
			else:
				new_utilities = self.policy_evaluation_linear(MDP, policy, discount)
			for state in MDP.states:
				delta = max(delta,abs(new_utilities[state] - old_utilities[state]))

			if delta < epsilon:
				converged = True

			if calc_loss:
				U = list(new_utilities.values())
				U_opt = list(opt_utilities.values())
				policy_loss = np.linalg.norm(np.subtract(U,U_opt))
				print("Iteration: {0}, Policy Loss: {1}".format(iteration,policy_loss)) 
				iterations.append(iteration)
				policy_losses.append(policy_loss)

			## Policy Improvement ##
			for state in MDP.states:
				optimal_utility = -np.inf
				for a in MDP.A:
					action = MDP.A[a]
					temp_utility = 0
					for prob, neighbour in MDP.P[state][action]:
						temp_utility += prob*(MDP.R[state][action] + discount*new_utilities[neighbour])
					if temp_utility > optimal_utility:
						optimal_utility = temp_utility
						improved_policy[state] = action
			
			if equal(policy , improved_policy) or iteration > 550:
				converged = True			
		if calc_loss:
			if iterative == 1:
				plt.title("Policy Loss vs No. of iterations, discount: {0} using {1}".format(discount, "iterative method"))
			else:
				plt.title("Policy Loss vs No. of iterations, discount: {0} using {1}".format(discount, "linear algebra method"))
			
			plt.xlabel("No. of iterations")
			plt.ylabel("Policy Loss")
			plt.plot(iterations,policy_losses)
			plt.savefig("Pol_Iter_{0}_{1}".format(int(discount*100), iterative))
			plt.show()
		return new_utilities,improved_policy

	def epsilon_greedy(self,action,epsilon, iter_num, decaying_epsilon):
		r = np.random.rand()
		if decaying_epsilon:
			epsilon = epsilon/iter_num
		if r < epsilon:
			## Exploration
			action = np.random.randint(low=0,high=6)
		return action

	
	def getRandStart(self, destination, bigger_grid = False):
		n = 5
		if bigger_grid:
			n=10
		row_t = np.random.randint(low = 0, high = n)
		col_t = np.random.randint(low = 0, high = n)
		p_locs = [(0,0),(0,4),(4,0),(4,3)]
		if bigger_grid:
			p_locs = [ (0,0),(0,5),(0,8),(3,3),(4,6),(8,0),(9,4),(9,9) ]
		found = False
		loc = (0, 0)
		while( not found):
			r = np.random.randint(low = 0, high = len(p_locs))
			loc = p_locs[r]
			if loc!=destination:
				break
		return loc, (row_t, col_t)


	def q_learning(self,dest,policy,alpha,discount,epsilon,num_episodes = 2000, decaying_epsilon = False, max_steps = 500,bigger_grid=False):
		MDP = Taxi_MDP(bigger_grid = bigger_grid)
		q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
		iteration = 1
		rewards, episodes = [], []
		for episode in range(num_episodes): 
			p_loc, t_loc = self.getRandStart(dest)
			MDP = Taxi_MDP(t_loc, p_loc, dest)
			state = MDP.startState
			done = False
			num_steps = 0
			while (not done):
				if num_steps>max_steps:
					# print(episode)
					break
				action = self.epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
				transition, reward = MDP.step(action)   ## transition = prob, next_state
				next_state = transition[1]   ## same as MDP.currState
				next_opt_action = policy[next_state]
				td_update_sample = reward + discount * q_table[next_state][next_opt_action] 
				q_table[state][action] = (1-alpha) * q_table[state][action] +  alpha * td_update_sample
				iteration += 1
				if next_state == MDP.destState:
					done = True
				state = next_state
				num_steps += 1
				policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
				
			
			####   Calculate discounted sum of rewards for this episode, averaged over 10 runs  #####
			curr_score = evaluate(q_table, MDP.states, 10, discount, dest,bigger_grid)
			rewards.append(curr_score)  
			episodes.append(episode)

		plt.plot(episodes, rewards)
		plt.xlabel("No. of training episodes")
		plt.ylabel("Accumulated Reward (averaged over 10 different runs)")
		if decaying_epsilon:
			plt.title("Q-learning with decaying exploration rate")
			plt.savefig("q_learning_decay_{0}_{1}".format(int(alpha*10), int(epsilon*100)))
		else:
			plt.title("Q-learning with fixed exploration rate")
			plt.savefig("q_learning_{0}_{1}".format(int(alpha*10), int(epsilon*100)))
		plt.show()
		
		learned_policy = {state : -1 for state in MDP.states}
		for state in MDP.states:
			learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
		utility = evaluate(q_table, MDP.states, 20, discount, dest,bigger_grid)
		if decaying_epsilon:
			print("Q Learning (decaying exploration rate) Utility: " + str(utility))
		else:
			print("Q Learning Utility: " + str(utility))					
		return learned_policy, utility


	def SARSA(self,dest,policy,alpha,discount,epsilon,num_episodes=1500,decaying_epsilon = False,max_steps=500,bigger_grid = False):
		MDP = Taxi_MDP(bigger_grid=bigger_grid)
		q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
		iteration = 1
		rewards, episodes = [], []
		for episode in range(num_episodes): 
			p_loc, t_loc = self.getRandStart(dest,bigger_grid)
			MDP = Taxi_MDP(t_loc, p_loc, dest,bigger_grid)
			prev_state = MDP.startState
			prev_action = self.epsilon_greedy(policy[prev_state], epsilon, iteration, decaying_epsilon)
			transition, prev_reward = MDP.step(prev_action)
			state = transition[1]

			done = False
			num_steps = 0
			while (not done):
				if num_steps>max_steps:
					# print(episode)
					break
				if state == MDP.destState:
					q_table[prev_state][prev_action] = (1-alpha) * q_table[prev_state][prev_action] +  alpha * prev_reward
					break
				action = self.epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
				#print(action)
				transition, reward = MDP.step(action)   ## transition = prob, next_state
				next_state = transition[1]
				td_update_sample = prev_reward + discount * q_table[state][action]
				q_table[prev_state][prev_action] = (1-alpha) * q_table[prev_state][prev_action] +  alpha * td_update_sample
				iteration += 1
				prev_state = state
				prev_action = action
				prev_reward = reward
				state = next_state
				num_steps += 1
				policy[prev_state] = max(q_table[prev_state], key= lambda action: q_table[prev_state][action])				
			
			####   Calculate discounted sum of rewards for this episode, averaged over 10 runs  #####
			curr_score = evaluate( q_table,MDP.states,10,discount, dest,bigger_grid)
			rewards.append(curr_score)  
			episodes.append(episode)

		
		plt.plot(episodes, rewards)
		plt.xlabel("No. of training episodes")
		plt.ylabel("Accumulated Reward (averaged over 10 different runs)")
		if decaying_epsilon:
			plt.title("SARSA with decaying exploration rate")
			plt.savefig("sarsa_learning_decay")
		else:
			plt.title("SARSA with fixed exploration rate)")
			plt.savefig("sarsa_learning")
		plt.show()
		learned_policy = {state : -1 for state in MDP.states}
		for state in MDP.states:
			learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
		utility = evaluate(q_table,MDP.states,20,discount, dest,bigger_grid)
		if decaying_epsilon:
			print("SARSA (decaying exploration rate) Utility: " + str(utility))
		else:
			print("SARSA Utility: " + str(utility))		
		return learned_policy, utility
	

def main():
	args = sys.argv[1:]
	print("Note: (0, 0) is at top left in our implementation")
	policy_obj = Policy()
	if args[0] == "A":
		tx, ty = [int(i) for i in input("Enter coordinates for Taxi: ").strip().split()]
		px, py = [int(i) for i in input("Enter coordinates for Passenger: ").strip().split()]
		dx, dy = [int(i) for i in input("Enter coordinates for Destination: ").strip().split()]
		if args[1] == "2":
			if args[2] == "a":
				MDP = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				policy = policy_obj.value_iteration(MDP,1e-12,0.9)
				MDP.simulate(policy)

			if args[2] == "b":
				MDP = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
				for discount in discounts:
					policy = policy_obj.value_iteration(MDP, 1e-18, discount)

			if args[2] == "c":
				MDP1 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				print("Discount : 0.1")
				policy1 = policy_obj.value_iteration(MDP1, 1e-18, 0.1)
				MDP1.simulate(policy1)
				print("\n")

				MDP2 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				print("Discount : 0.99")
				policy2 = policy_obj.value_iteration(MDP2, 1e-18, 0.99)
				MDP2.simulate(policy2)
				print("\n")
			
				print("After sampling start states")
				p_loc, t_loc = policy_obj.getRandStart((dx,dy))
				print("Taxi start position: " + str(t_loc))
				print("Passenger start position: " + str(p_loc))
				MDP1 = Taxi_MDP(t_loc, p_loc, (dx, dy))
				print("Discount : 0.1")
				policy1 = policy_obj.value_iteration(MDP1, 1e-18, 0.1)
				MDP1.simulate(policy1)
				print("\n")

				MDP2 = Taxi_MDP(t_loc, p_loc, (dx, dy))
				print("Discount : 0.99")
				policy2 = policy_obj.value_iteration(MDP2, 1e-18, 0.99)
				MDP2.simulate(policy2)
				print("\n")

				p_loc, t_loc = policy_obj.getRandStart((dx,dy))
				print("Taxi start position: " + str(t_loc))
				print("Passenger start position: " + str(p_loc))
				MDP1 = Taxi_MDP(t_loc, p_loc, (dx, dy))
				print("Discount : 0.1")
				policy1 = policy_obj.value_iteration(MDP1, 1e-18, 0.1)
				MDP1.simulate(policy1)
				print("\n")

				MDP2 = Taxi_MDP(t_loc, p_loc, (dx, dy))
				print("Discount : 0.99")
				policy2 = policy_obj.value_iteration(MDP2, 1e-18, 0.99)
				MDP2.simulate(policy2)
				print("\n")				


		if args[1] == "3":
			if args[2] == "a":
				print("Iterative Method: ")
				MDP1 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				temp_policy1 = {state : np.random.randint(low = 0 , high = 6) for state in MDP1.states}
				utility, policy1 = policy_obj.policy_iteration(MDP1, temp_policy1, 0.99, 1e-18, 1)
				MDP1.simulate(policy1)
				print(utility)

				print("\n")
				print("Linear Algebra Method: ")
				MDP2 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
				temp_policy2 = {state : np.random.randint(low = 0 , high = 6) for state in MDP2.states}
				utility, policy2 = policy_obj.policy_iteration(MDP2, temp_policy2, 0.99, 1e-12, 1)
				MDP2.simulate(policy2)
				print(utility)


			if args[2] == "b":
				discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
				for discount in discounts:
					MDP1 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
					policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP1.states}
					utilities, learned_policy = policy_obj.policy_iteration(MDP1, policy, discount, iterative=1)
					
					MDP2 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
					policy_iter = {state : np.random.randint(low = 0 , high = 6) for state in MDP1.states}
					utility_iter, learned_policy_iter = policy_obj.policy_iteration(MDP2, policy_iter, discount, iterative=1, calc_loss = True, opt_utilities = utilities)
					# MDP2.simulate(learned_policy_iter)

					MDP3 = Taxi_MDP((tx, ty), (px, py), (dx, dy))
					policy_linear = {state : np.random.randint(low = 0 , high = 6) for state in MDP1.states}
					utility_linear, learned_policy_linear = policy_obj.policy_iteration(MDP3, policy_linear, discount, iterative=0, calc_loss = True, opt_utilities = utilities)
					# MDP3.simulate(learned_policy_linear)

	if args[0] == "B":
		if args[1] == "2":
			dest = (0, 0)
			MDP = Taxi_MDP()

			temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
			policy, utility = policy_obj.q_learning(dest,temp_policy, 0.25, 0.99, 0.1, decaying_epsilon = False)
			print("Accumulated reward with Q-learning (constant eps): ", utility)
			print("\n")

			temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
			policy, utility = policy_obj.q_learning(dest,temp_policy, 0.25, 0.99, 0.1, decaying_epsilon = True)
			print("Accumulated reward with Q-learning (decaying eps): ", utility)
			print("\n")

			temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
			policy, utility = policy_obj.SARSA(dest, temp_policy, 0.25, 0.99, 0.1, decaying_epsilon = False)
			print("Accumulated reward with SARSA (fixed eps): ", utility)
			print("\n")

			temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
			policy, utility = policy_obj.SARSA(dest, temp_policy, 0.25, 0.99, 0.1, decaying_epsilon = True)
			print("Accumulated reward with SARSA (decaying eps): ", utility)
			print("\n")

		if args[1] == "3":
			dest = (0, 0)
			MDP = Taxi_MDP()
			temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
			policy, utility = policy_obj.SARSA(dest, temp_policy, 0.25, 0.99, 0.1, decaying_epsilon = True)
			for i in range(5):
				p_loc, t_loc = getRandStart(dest)
				print("Passenger: {0}, Taxi: {1}, Destination: {2}".format(p_loc, t_loc, dest))
				instance = Taxi_MDP(t_loc, p_loc, dest)
				instance.simulate(policy)
				print("\n")


		if args[1] == "4":
			dest = (0, 0)
			eps = [0, 0.05, 0.1, 0.5, 0.9]
			alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
			MDP = Taxi_MDP()
			for e in eps:
				temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
				policy, utility = policy_obj.q_learning(dest,temp_policy, 0.1, 0.99, e, decaying_epsilon = False)

			for a in alpha:
				temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in MDP.states}
				policy, utility = policy_obj.q_learning(dest,temp_policy, a, 0.99, 0.1, decaying_epsilon = False)

		if args[1] == "5":
			dest = (0, 0)
			depots = [ (0,0),(0,5),(0,8),(3,3),(4,6),(8,0),(9,4),(9,9) ]
			for i in range(5):
				p_loc,t_loc = policy_obj.getRandStart(dest,True)
				instance = Taxi_MDP(t_loc, p_loc, dest,bigger_grid = True)
				temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in instance.states}	
				policy, utility = policy_obj.q_learning(dest,temp_policy, alpha=0.25, discount=0.99, epsilon=1e-18, num_episodes=2000, decaying_epsilon= False, max_steps= 1500,bigger_grid = True)
				# instance.simulate(policy)
				print(utility)


		




	# a = Taxi_MDP()
	# po = Policy()
	# eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-18]
	# for epsilon in eps:
	# 	policy = po.value_iteration(a, epsilon, 0.9)

	# discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
	# for discount in discounts:
	# 	policy = po.value_iteration(a, 1e-18, discount)
	
	# T , P, D = (4, 3), (4,0), (0, 0)
	# instance = Taxi_MDP(T, P, D)
	# policy_finder = Policy()
	# print(instance.startState)


	# policy = policy_finder.value_iteration(instance, 1e-18, 0.1)
	# # print(policy[instance.startState])
	# # print(instance.P[instance.startState][policy[instance.startState]])
	# instance.simulate(policy)


	# temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in instance.states}
	# utilities, policy = policy_finder.policy_iteration(instance, temp_policy, 0.99, 1e-8, 1)
	# isn = Taxi_MDP(T, P, D)
	# temp_policy1 = {state : np.random.randint(low = 0 , high = 6) for state in instance.states}

	# u, p = policy_finder.policy_iteration(isn, temp_policy1, 0.99, 1e-18, 0)

	# instance.simulate(p)

	## Evaluating the episodes while learning ## (remove) 

	#### Q Learning ####
	# print("Q Learning")
	# start = time.time()
	# temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in instance.states}	
	# # # # destx, desty = input("Enter Destination Coordinates: ")
	
	# policy = policy_finder.q_learning(D,temp_policy,0.25,0.99,0.1, decaying_epsilon = False)
	# instance.simulate(policy[0])

	# print(time.time()-start)

	# #### Q Learning with decaying exploration rate ####
	# print("Q Learning with decaying exploration rate")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.q_learning(instance,temp_policy,0.25,0.99,1e-18,decaying_epsilon=True)
	# print(time.time()-start)
	
	# #### SARSA ####
	# print("SARSA")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.SARSA(instance,temp_policy,0.25,0.99,1e-18)
	# print(time.time()-start)

	# #### SARSA with decaying exploration rate ####
	# print("SARSA with decaying exploration rate")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.SARSA(instance,temp_policy,0.25,0.99,1e-18,decaying_epsilon=True)
	# print(time.time()-start)


	# print(policy)
	# a.step(0)
	# a.step(1)
	
if __name__ == "__main__":
    main()
					

