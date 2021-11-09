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
		p_locs = [(0,0),(0,4),(4,0),(4,3)]
		if P_pos not in p_locs or D not in p_locs:
			raise 0

		self.num_rows = 5
		self.num_cols = 5
		self.currState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)
		self.startState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)	
		self.destState = encode(D[0], D[1], D[0], D[1], 0)
		self.A = {"S": 0, "N": 1, "E": 2, "W": 3, "Pickup": 4, "Putdown": 5}
		self.states = self.getStates()
		self.P, self.R = self.getModels()

	def getStates(self):
		states = []
		for row_p in range(self.num_rows):
			for col_p in range(self.num_cols):
				# states.append(State(row_p, col_p, row_p, col_p, 1))
				states.append(encode(row_p, col_p, row_p, col_p, 1))
				for row_t in range(self.num_rows):
					for col_t in range(self.num_cols):
						states.append(encode(row_p, col_p, row_t, col_t, 0))
						# states.append(State(row_p, col_p, row_t, col_t, 0))
		return states

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
			if inside_taxi == 1:
				new_state0 = encode(n_row0, n_col0, n_row0, n_col0, 1)
				new_state1 = encode(n_row1, n_col1, n_row1, n_col1, 1)
				new_state2 = encode(n_row2, n_col2, n_row2, n_col2, 1)
				new_state3 = encode(n_row3, n_col3, n_row3, n_col3, 1)
			else:
				new_state0 = encode(row_p, col_p, n_row0, n_col0, 0)
				new_state1 = encode(row_p, col_p, n_row1, n_col1, 0)
				new_state2 = encode(row_p, col_p, n_row2, n_col2, 0)
				new_state3 = encode(row_p, col_p, n_row3, n_col3, 0)
			# if state == self.startState:
			# 	print(decode(new_state0))
			# 	print(decode(new_state1))
			# 	print(decode(new_state2))
			# 	print(decode(new_state3))
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
		

def main():
	a = Taxi_MDP()
	a.step(0)
	a.step(1)

if __name__ == "__main__":
    main()
					
					










		


		

