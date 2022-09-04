#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random
import copy
import math

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	# O(n^3) Time complexity
	# O(n) Space Complexity
	def greedy(self, time_allowance=60.0):
		results = {} # hashmap to store all the results
		bssf = None
		cities = self._scenario.getCities()
		foundTour = False # simple boolean value to keep track if found tour or not
		count = 0
		
		start_time = time.time()
		# O(n)
		while not foundTour and time.time()-start_time < time_allowance:
			# O(n)
			for each_city in cities:
				route = self.build_route(each_city, cities)
				if route is not None:
					if bssf is None or route.cost < bssf.cost:
						count += 1
						bssf = route
						foundTour = True
		end_time = time.time()

		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results


	def build_route(self, startCity, cities):
		route = [] # this will keep track of all the routes in order
		visited = [] # keeps track of all the visited cities
		cities_num = len(cities)
		#O (n^2)
		for _ in range(cities_num):
			smallest = math.inf 
			bestCity = None

			for city in cities:
				if city not in visited:
					length = startCity.costTo(city)
					if smallest > length:
						smallest = length
						bestCity = city
			nextCity = bestCity

			if nextCity is None:
				return None

			route.append(nextCity)
			visited.append(nextCity)
			startCity = nextCity

		return TSPSolution(route)
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		results = {} # hashmap to store results to pull from
		maxQueue = 0
		totalStatesCreated = 0
		totalStatesPruned = 0
		count = 0 # number of solutions found

		bssf = self.greedy()['soln'] # using solution from greedy to find initial BSSF instead of using default tour
		cities = self._scenario.getCities()
		pqueue = [] # priority queue
		heapq.heapify(pqueue) # heapify priority queue

		matrix = [[math.inf for _ in range(len(cities))] for _ in range(len(cities))]
		# print(matrix[0][0])
		# print(matrix[-1][-1])

		for i in range(len(cities)):
			for j in range(len(cities)):
				matrix[i][j] = cities[i].costTo(cities[j])

		# print(matrix[0][0])
		# print(matrix[-1][-1])
		index = random.randint(0, len(cities) - 1) # random starting point

		# We need to create an initial state and start reducing it first
		initialState = self.createInitialState(matrix, cities, index)
		#print(initialState)

		heapq.heappush(pqueue, initialState)

		start_time = time.time()
		while len(pqueue) > 0 and time.time() - start_time < time_allowance:
			currentState = heapq.heappop(pqueue) # getting the top current state off the queue

			if len(pqueue) > maxQueue: # updating the siZe of maxQueue
				maxQueue = len(pqueue)

			if currentState.lower_bound < bssf.cost:
				if len(currentState.visited_route) == len(cities): # if we have visited every single city
					last_city = currentState.route[-1]
					start_city = currentState.route[0] # initial city with first initial random index each time arbitrary value

					if last_city.costTo(start_city) is not math.inf:
						solution = TSPSolution(currentState.route) # give current array of routes

						if solution.cost < bssf.cost: # if the solution cost is actually less, substitute
							bssf = solution 

							# Pruning - O(n) operation
							for i in pqueue:
								if currentState.lower_bound >= bssf.cost:
									pqueue.remove(i)
									totalStatesPruned += 1
							count += 1 # increment number of solutions found
					else:
						continue
				else: 
					for i in range(len(cities)):
						if i not in currentState.visited_route:
							newState = States(currentState, i, cities) # generate a new state
							totalStatesCreated += 1 # increment number of new states created

							if newState.lower_bound < bssf.cost and newState.lower_bound != math.inf:
								heapq.heappush(pqueue, newState)
							else:
								totalStatesPruned += 1 # TODO: Double check if this is where I was missing pruning

		end_time = time.time()

		# Results map				
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQueue
		results['total'] = totalStatesCreated
		results['pruned'] = totalStatesPruned

		return results

	# O(n^2)
	def createInitialState(self, matrix, cities, startIndex):

		cost = 0
		depth = 1
		startIndex = startIndex

		# Creating the initial state, reducing it and then returning the result
		initialState = State(None, None, None) # Try it with different values and see what has changed
		initialState.set_first_state(matrix, cities, startIndex, cost, depth) # Try it with different values and see what has changed

		return initialState

	def generateStates(self, currentState): # Is there a better way to handle states?
		pass

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		
class States:

    def __init__(self, current_state, city_index, cities):
        self.current_state = current_state # Needs to keep track in which state we're currently working on
        self.matrix = copy.deepcopy(current_state.matrix) # Copy current State matrix, if I don't copy it doesn't like it
        self.depth = self.current_state.depth + 1 # Keep track of depth
        self.lower_bound = 0 #InitialiZe and keep track of lower bound
        self.parent_state_lower_bound = self.current_state.lower_bound # Keep track of the parent's lower bound, might not need it we'll see

        self.rows = copy.deepcopy(current_state.rows) # copy rows that will be used to be reduced each time 
        self.cols = copy.deepcopy(current_state.cols) # copy cols that will be used to be reduced each time
        self.route = copy.deepcopy(current_state.route) # copy routes
        self.visited_route = copy.deepcopy(current_state.visited_route) # copy visited routes each time 

        self.from_city_index = current_state.to_city_index
        self.to_city_index= city_index

        self.reduce_matrix(cities) # Reduce the matrix

    def __lt__(self, other): # Someone in Slack said this might help?
        return True

    # function to reduce further states
    def reduce_matrix(self, cities):
        cost_of_path = self.matrix[self.from_city_index][self.to_city_index] # Keeping track of the cost from to city

        for i in range(len(self.matrix)): # blocking out row with inf
            self.matrix[self.from_city_index][i] = math.inf

        for i in range(len(self.matrix)): # blocking out cols with inf
            self.matrix[i][self.to_city_index] = math.inf

        self.matrix[self.to_city_index][self.from_city_index] = math.inf # get speficic index from city to city and mark as inf
        self.rows.add(self.from_city_index) # keeping track of row value to not accidentally reduce it
        self.cols.add(self.to_city_index) # keeping track of col value to not accidentally reduce it
        cost = self.reduce_rows() # cost of reducing rows
        cost += self.reduce_cols(self.to_city_index) # cost of reducing add
        self.lower_bound = cost + cost_of_path + self.parent_state_lower_bound # adding costs to lower bound together
        self.route.append(cities[self.to_city_index]) # add current city to the routes array and to visited set
        self.visited_route.add(self.to_city_index)

    def _reduce(self): # this will only reduce the initial state without making a copy just yet
        cost_row = self.reduce_rows()
        cost_col = self.reduce_cols(self.to_city_index)
        self.lower_bound = cost_row + cost_col

    def set_first_state(self, matrix, cities, startIndex, cost, depth):
        self.matrix = copy.deepcopy(matrix)
        self.parent_state_lower_bound = cost
        self.depth = depth
        self.visited_route = set()  # keep track of the cities that we have visited
        self.route = []
        self.cols = set() # keep track of the cols and rows that are marked as inf
        self.rows = set()
        self.to_city_index= startIndex # starting city index (random in this case)
        self.route.append(cities[startIndex]) # add the starting index to visited route
        self.visited_route.add(startIndex) # add the starting index to visited route
        self._reduce() # Initial reduce for the first State - only needed once

    def reduce_rows(self):
        cost= 0
        for row in range(len(self.matrix)):
            if row not in self.rows:
                matrix_row = self.matrix[row] # getting the minimun value present in a row
                smallest_value = math.inf
                for i in range(len(matrix_row)):
                    num = matrix_row[i]
                    if num == 0:
                        smallest_value =  num
                        break
                    elif num < smallest_value:
                        smallest_value = num
                if smallest_value > 0 and smallest_value != math.inf:
                    for col in range(len(self.matrix)):
                        cur_val = self.matrix[row][col]
                        self.matrix[row][col] = cur_val - smallest_value
                    cost += smallest_value
        return cost

    def reduce_cols(self, colTo = 0):
        cost_reduce = 0
        min_bool = True # making sure we haven't already updated the minimun value when updating the row
        for row in range(len(self.matrix)):
            if colTo == 0:
                min_bool = True # only if we reduce the initial state
            elif row in self.cols or row is colTo: # this will guarantee that we haven't actually visited before
                min_bool = False
            if min_bool: # If we haven't updated a column value when updating the rows
                smallest_value = math.inf
                for i in range(len(self.matrix)):
                    num = self.matrix[i][row]
                    if num == 0:
                        smallest_value =  num
                        break
                    elif num < smallest_value:
                        smallest_value = num
                if smallest_value > 0 and smallest_value != math.inf:
                    for col in range(len(self.matrix)):
                        cur_val = self.matrix[col][row]
                        self.matrix[col][row] = cur_val - smallest_value
                    cost_reduce += smallest_value
            min_bool = True
        return cost_reduce