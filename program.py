import pdb
import math
from sys import argv
import random

########################################################################
#find the max difference between utilities
def difference(values, s):
	diff = 0

	for (i,j) in [(i,j) for i in range(s) for j in range(s)]:
		if diff < abs(values[0][i][j] - values[1][i][j]):
			diff = abs(values[0][i][j] - values[1][i][j])

	return diff
########################################################################

########################################################################
#find expected reward of a given state using Bellman-equation
#k is the value representing the action: up(0), down(1), left(2), right(3)
def expectedReward(values, k, i, j):
    #terminal state
	if i == 0 and j == s - 1:
		return 5
	else:
		#get the values of each adjacent cell
		if i != 0:
			up = values[0][i - 1][j]
		else:
			up = values[0][i][j]

		if i != s - 1:
			down = values[0][i + 1][j]
		else:
			down = values[0][i][j]

		if j != 0:
			left = values[0][i][j - 1]
		else:
			left = values[0][i][j]

		if j != s - 1:
			right = values[0][i][j + 1]
		else:
			right = values[0][i][j]

		#return according to selected action
		if k == 0:
			return -0.1 + discount * (0.85 * up + 0.05 * down + 0.05 * left + 0.05 * right)
		if k == 1:
			return -0.1 + discount * (0.05 * up + 0.85 * down + 0.05 * left + 0.05 * right) 
		if k == 2:
			return -0.1 + discount * (0.05 * up + 0.05 * down + 0.85 * left + 0.05 * right) 
		else:
			return -0.1 + discount * (0.05 * up + 0.05 * down + 0.05 * left + 0.85 * right)
########################################################################

########################################################################
#print the values given by vi
def printValues(values, s):
	for i in range(s):
		for j in range(s):
			print "%f &" % values[1][i][j],
		print "\\\\"
########################################################################

########################################################################
#print the policy given by vi
def printPolicy(values, s):
	
	for i in range(s):
		for j in range(s):
			#find max action
			value_maxAction = -9999;

			for k in range(4):
				#up
				if k == 0 and i != 0:
					aux = values[1][i - 1][j]
				#down
				elif k == 1 and i != s - 1:
					aux = values[1][i + 1][j]
				#left
				elif k == 2 and j != 0:
					aux = values[1][i][j - 1]
				#right
				elif k == 3 and j != s - 1:
					aux = values[1][i][j + 1]
				#bump wall
				else:
					aux = values[1][i][j]
			
				if aux > value_maxAction:
					maxAction = k
					value_maxAction = aux

			#print it
			if maxAction == 0:
				print "up    & ",
			elif maxAction == 1:
				print "down  & ",
			elif maxAction == 2:
				print "left  & ",
			else:
				print "right & ",
		print "\\\\"
########################################################################

########################################################################
#value iteration algorithm
def vi(s, epsilon):
	global iterations
	
	#initialize values to zero
	values = [[[0 for i in range(1000)] for j in range(1000)] for k in range(2)]
	
	#loop until convergence
	while True:
		iterations -= 1
		
		#copy values[1] to values[0]
		for (i,j) in [(i,j) for i in range(s) for j in range(s)]:
			values[0][i][j] = values[1][i][j]
		
		#loop through the states
		for (i,j) in [(i,j) for i in range(s) for j in range(s)]:
			
			#find max value
			values[1][i][j] = -9999
			
			for k in range(4):
				aux = expectedReward(values, k, i, j)
				
				if aux > values[1][i][j]:
					values[1][i][j] = aux
		
		#print "***************************************\n"
		#printValues(values, s)
		
		#break conditions
		if difference(values, s) <= epsilon or iterations < 1:
			break

	return values
########################################################################

########################################################################
#this will get a sample of executing action a at state (i, j)
def newSample(i, j, a):
	global s
	
	k = random.uniform(0, 1)
	
	#we did not succeeded doing action a
	if k > 0.85:
		actions = [0, 1, 2, 3]
		
		#eliminate action a
		actions.remove(a)
		
		#select new action
		a = random.choice(actions)
		
	
	#get next state	
	if a == 0:
		if i == 0:
			i2 = i
			j2 = j
		else:
			i2 = i - 1
			j2 = j
	elif a == 1:
		if i == s - 1:
			i2 = i
			j2 = j
		else:
			i2 = i + 1
			j2 = j
	elif a == 2:
		if j == 0:
			i2 = i
			j2 = j
		else:
			i2 = i
			j2 = j - 1
	else:
		if j == s - 1:
			i2 = i
			j2 = j
		else:
			i2 = i
			j2 = j + 1
	
	#return the sampled state
	return (i2, j2) 
########################################################################

########################################################################
#this will simulate we are "running" in the world
def rollout(i, j, depth):
	global s
	global discount
	
	if depth == 0:
		return 0
	if i == 0 and j == s - 1:
		return 5
	else:
		actions = [0, 1, 2, 3]
		
		#remove bump-walls actions
		if i == 0:
			actions.remove(0)
		if i == s - 1:
			actions.remove(1)
		if j == 0:
			actions.remove(2)
		if j == s - 1:
			actions.remove(3)
		
		#select a random action
		a = random.choice(actions)
		
		#get a sample of the action
		(i2, j2) = newSample(i, j, a)
		
		#return expected reward
		return -0.1 + discount * rollout(i2, j2, depth - 1)
########################################################################

########################################################################
#here is where the actual UCT algorithm happens
def simulate(i, j, depth):
	global s
	global Q
	global N
	global T
	global exploration
	global discount
	
	#last node to be explored or terminal state
	if depth == 0:
		return 0
	
	if i == 0  and j == s - 1:
		return 5
		
	#state not in tree	
	elif (i, j) not in T:
		for k in range(4):
			#initialize to zero
			N[(i, j, k)] = 0
			Q[(i, j, k)] = 0
		
		#add state to the tree
		T.add((i, j))
		
		#return the rollout	
		return rollout(i, j, depth)
	
	else:
		#get the value of N(s)
		n = 0
		actions = []
		
		for k in range(4):
			n += N[(i, j, k)]
			
			if N[(i, j, k)] == 0:
				actions.append(k)
			
		maxAction = 0
		
		#remove bump-walls actions
		if i == 0 and 0 in actions:
			actions.remove(0)
		if i == s - 1 and 1 in actions:
			actions.remove(1)
		if j == 0 and 2 in actions:
			actions.remove(2)
		if j == s - 1 and 3 in actions:
			actions.remove(3)
		
		#one or more action was not previously selected
		if len(actions) > 0:
			#select randomly between them
			maxAction = random.choice(actions)
				
		#find the maximum action
		else:
			maxValue = -9999
			for k in range(4):
				if N[(i, j, k)] != 0:
					aux = Q[(i, j, k)] + exploration * math.sqrt(math.log(n) / N[(i, j, k)])
					if aux > maxValue:
						maxAction = k
						maxValue = aux
			
		#get a new sample
		(i2, j2) = newSample(i, j, maxAction)
		
		#compute q with the value of the new sample
		q = -0.1 + discount * simulate(i2, j2, depth - 1)
		
		#update N and Q
		N[(i, j, maxAction)] += 1
		Q[(i, j, maxAction)] += (q - Q[(i, j, maxAction)])/N[(i, j, maxAction)]
		
		#return the value for this node
		return q 
########################################################################

########################################################################
#this function just call the simulate sequence
def uct(depth, simulations):
	global s
	global Q
	global N
	global T
	
	for i in range(simulations):
		q = simulate(s - 1, 0, depth)
	
	#return the action with the biggest Q value
	maxValue = -9999
	maxAction = 0
	for k in [0, 3]:
		aux = Q[(s - 1, 0, k)]
		
		if aux > maxValue:
			maxAction = k
			maxValue = aux

	#return value and best action
	return (maxValue, maxAction)
########################################################################

########################################################################
#here starts the execution of the program
#scan arguments
it = iter(range(len(argv)))

alg = False
discount = 1
epsilon = 0
iterations = 999999
simulations = 10000
exploration = 1
depth = 5
s = 0

#initialize some dicts and sets
Q = {}
N = {}
T = set()

for i in it:
	#scan type of algorithm
	if argv[i] == "-alg":
		i = next(it)

		if argv[i] == "vi":
			alg = False
		elif argv[i] == "uct":
			alg = True

	#scan gamma
	if argv[i] == "-gamma":
		i = next(it)
		discount = float(argv[i])

	#scan epsilon
	elif argv[i] == "-epsilon":
		i = next(it)
		epsilon = float(argv[i])

	#scan max number of iterations of VI
	elif argv[i] == "-iterations":
		i = next(it)
		iterations = int(argv[i])

	#scan number of simulations of UCT
	elif argv[i] == "-simulations":
		i = next(it)
		simulations = int(argv[i])

	#scan exploration constant of UCT
	elif argv[i] == "-exploration":
		i = next(it)
		exploration = int(argv[i])

	#scan max depth of UCT
	elif argv[i] == "-depth":
		i = next(it)
		depth = int(argv[i])

	#scan size of problem
	elif argv[i] == "-size":
		i = next(it)
		s = int(argv[i])
		
#run algorithms
if not alg:
	values = vi(s, epsilon)
	print "Values:"
	printValues(values, s)
	print
	print "Policy:"
	printPolicy(values, s)

else:
	print "Value:"
	(x, a) = uct(depth, simulations)
	print x
	print "Action:"
	print a
########################################################################
