#!/usr/bin/python3


import math
import numpy as np
import random
import time
import copy



class TSPSolution:
    def __init__( self, listOfCities):
        self.route = listOfCities
        self.cost = self._costOfRoute()
        #print( [c._index for c in listOfCities] )

    def _costOfRoute( self ):
        cost = 0
        last = self.route[0]
        for city in self.route[1:]:
            cost += last.costTo(city)
            last = city
        cost += self.route[-1].costTo( self.route[0] )
        return cost

    def enumerateEdges( self ):
        elist = []
        c1 = self.route[0]
        for c2 in self.route[1:]:
            dist = c1.costTo( c2 )
            if dist == np.inf:
                return None
            elist.append( (c1, c2, int(math.ceil(dist))) )
            c1 = c2
        dist = self.route[-1].costTo( self.route[0] )
        if dist == np.inf:
            return None
        elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
        return elist


def nameForInt( num ):
    if num == 0:
        return ''
    elif num <= 26:
        return chr( ord('A')+num-1 )
    else:
        return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)








class Scenario:

    HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

    def __init__( self, city_locations, difficulty, rand_seed ):
        self._difficulty = difficulty

        if difficulty == "Normal" or difficulty == "Hard":
            self._cities = [City( pt.x(), pt.y(), \
                                  random.uniform(0.0,1.0) \
                                ) for pt in city_locations]
        elif difficulty == "Hard (Deterministic)":
            random.seed( rand_seed )
            self._cities = [City( pt.x(), pt.y(), \
                                  random.uniform(0.0,1.0) \
                                ) for pt in city_locations]
        else:
            self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


        num = 0
        for city in self._cities:
            #if difficulty == "Hard":
            city.setScenario(self)
            city.setIndexAndName( num, nameForInt( num+1 ) )
            num += 1

        # Assume all edges exists except self-edges
        ncities = len(self._cities)
        self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

        if difficulty == "Hard":
            self.thinEdges()
        elif difficulty == "Hard (Deterministic)":
            self.thinEdges(deterministic=True)

    def getCities( self ):
        return self._cities


    def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
        perm = np.arange(n)
        for i in range(n):
            randind = random.randint(i,n-1)
            save = perm[i]
            perm[i] = perm[randind]
            perm[randind] = save
        return perm

    def thinEdges( self, deterministic=False ):
        ncities = len(self._cities)
        edge_count = ncities*(ncities-1) # can't have self-edge
        num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

        can_delete	= self._edge_exists.copy()

        # Set aside a route to ensure at least one tour exists
        route_keep = np.random.permutation( ncities )
        if deterministic:
            route_keep = self.randperm( ncities )
        for i in range(ncities):
            can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

        # Now remove edges until 
        while num_to_remove > 0:
            if deterministic:
                src = random.randint(0,ncities-1)
                dst = random.randint(0,ncities-1)
            else:
                src = np.random.randint(ncities)
                dst = np.random.randint(ncities)
            if self._edge_exists[src,dst] and can_delete[src,dst]:
                self._edge_exists[src,dst] = False
                num_to_remove -= 1




class City:
    def __init__( self, x, y, elevation=0.0 ):
        self._x = x
        self._y = y
        self._elevation = elevation
        self._scenario	= None
        self._index = -1
        self._name	= None

    def setIndexAndName( self, index, name ):
        self._index = index
        self._name = name

    def setScenario( self, scenario ):
        self._scenario = scenario

    ''' <summary>
        How much does it cost to get from this city to the destination?
        Note that this is an asymmetric cost function.
         
        In advanced mode, it returns infinity when there is no connection.
        </summary> '''
    MAP_SCALE = 1000.0
    def costTo( self, other_city ):

        assert( type(other_city) == City )

        # In hard mode, remove edges; this slows down the calculation...
        # Use this in all difficulties, it ensures INF for self-edge
        if not self._scenario._edge_exists[self._index, other_city._index]:
            return np.inf

        # Euclidean Distance
        cost = math.sqrt( (other_city._x - self._x)**2 +
                          (other_city._y - self._y)**2 )

        # For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
        if not self._scenario._difficulty == 'Easy':
            cost += (other_city._elevation - self._elevation)
            if cost < 0.0:
                cost = 0.0					# Shouldn't it cost something to go downhill, no matter how steep??????


        return int(math.ceil(cost * self.MAP_SCALE))

# State class to represent all the different states

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

    def reduce_cols(self, colTo=-1):
        cost_reduce = 0
        min_bool = True # making sure we haven't already updated the minimun value when updating the row
        for row in range(len(self.matrix)):
            if colTo == -1:
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

    
