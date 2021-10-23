# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

import time

import search
import util
from game import Actions
from game import Agent
from game import Directions


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic',
                 pacman_energy_level=10, food_energy_level=3):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

        self.pacmanEnergyLevel = int(pacman_energy_level)
        self.foodEnergyLevel = int(food_energy_level)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        if self.searchType == HungerGamesSearchProblem:
            problem = self.searchType(state, self.pacmanEnergyLevel, self.foodEnergyLevel)  # Makes a new search problem
        else:
            problem = self.searchType(state)
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost

"""=============================================START OF MY OWN CODE================================================="""

class HungerGamesSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point (maze exit point) in the maze,
    with the constraint that PacMan must always have a positive energy level.

    Initially, PacMan has a given energy level: pacman_energy_level. Then,
    - PacMan loses one energy level in each step
    - PacMan can get food_energy_level extra energy levels with every food dot that it eats.

    The state space consists of ((x, y), energy_level, food_grid) tuples, where
    - (x, y) denotes PacMan's current position
    - energy_level marks the current energy level of PacMan
    - food_grid is a grid showing whether a cell of the maze has any food on it.
    """

    IMPOSSIBLE_TO_SOLVE_STATE_HEURISTIC_VALUE = 1000000

    def __init__(self, game_state, pacman_energy_level=10, food_energy_level=3, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        pacman_energy_level: The initial energy of PacMan
        food_energy_level: The extra energy given by each food dot.
        warn: If set to true, the validity of the initial game state is initialized.
        visualize: If set to true, the expanded nodes are marked in the maze layout, in the graphical window.
        """
        self.walls = game_state.getWalls()
        self.startState = (game_state.getPacmanPosition(), pacman_energy_level, game_state.getFood())
        self.foodEnergyLevel = food_energy_level
        self.mazeExitPosition = game_state.getMazeExitPosition()
        self.visualize = visualize
        if warn and (self.mazeExitPosition == ()):
            print 'Warning: this does not look like a regular hunger games search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state[0] == self.mazeExitPosition

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state[0])
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        x, y = state[0]
        food_grid = state[2]
        energy_level = state[1]

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_energy_level = energy_level - 1
                next_food_grid = food_grid.copy()
                if next_food_grid[nextx][nexty]:
                    next_energy_level += self.foodEnergyLevel
                    next_food_grid[nextx][nexty] = False
                if next_energy_level > 0:
                    # Else: invalid successor state, because PacMan would die because of hunger
                    next_state = ((nextx, nexty), next_energy_level, next_food_grid)
                    cost = 1
                    successors.append((next_state, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state[0])

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move (stepping on a wall, or loosing all the energy), return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()[0]
        food_grid = self.getStartState()[2]
        cost = 0
        energy_level = self.getStartState()[1]
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            energy_level = energy_level - 1
            if food_grid[x][y]:
                energy_level += self.foodEnergyLevel
            if energy_level < 0 or self.walls[x][y]: return 999999
            cost += 1
        return cost


def hungerGamesEuclideanHeuristic(state, problem):
    """
    The Euclidean distance heuristic for a HungerGamesSearchProblem

    Heuristic identifier in the documentation: A
    """
    curr_pos = state[0]
    goal = problem.mazeExitPosition
    return ((curr_pos[0] - goal[0]) ** 2 + (curr_pos[1] - goal[1]) ** 2) ** 0.5


def manhattanDistance(pointA, pointB):
    """Helper function for computing the ManhattanDistance between two points given via their (x, y) coordinates"""
    return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])


def hungerGamesManhattanHeuristic(state, problem):
    """
    The Manhattan distance heuristic for a HungerGamesSearchProblem

    Heuristic identifier in the documentation: B
    """
    curr_pos = state[0]
    goal = problem.mazeExitPosition
    return manhattanDistance(curr_pos, goal)


def isPosInRectangle(corner1, corner2, pos):
    """
    Returns True if (x, y) position is located inside the rectangle defined by the two corners,
    which are on one of the diagonals.
    """
    rectangle_low_bound = min(corner1[0], corner2[0])
    rectangle_high_bound = max(corner1[0], corner2[0])
    rectangle_left_bound = min(corner1[1], corner2[1])
    rectangle_right_bound = max(corner1[1], corner2[1])
    return rectangle_high_bound >= pos[
        0] >= rectangle_low_bound and rectangle_right_bound >= pos[1] >= rectangle_left_bound


def noFoodDotsInRectange(corner1, corner2, food_grid):
    """
    Computes the number of food dots located inside the rectangle defined by two corners,
    which are on one of the diagonals.

    food_grid: boolean matrix, with food_grid[x][y] True iff there is a food dot in the location (x, y).
    corner1, corner2: 2 tuples in the format (x coordinate, y coordinate),
                      marking the two diagonal corners of the rectangle in concern.
    """
    return len([food_dot for food_dot in food_grid.asList() if isPosInRectangle(corner1, corner2, food_dot)])


def buildGoalOrientedIntegerFoodGridRectangle(start_corner_pos, goal_corner_pos, food_grid):
    """
    Builds a matrix with the content of food_grid inside the rectangle defined by the 2 corners
    (start_corner_pos and goal_corner_pos), which represent the endpoints of one of the diagonals of the rectangle.

    The matrix may be mirrored horizontally and/or vertically,
    such that start_corner_pos ends up being mapped to position (0, 0) of the final matrix,
    and goal_corner_pos ends up being mapped to position (n-1, m-1) of the final matrix,
    where n and m are the number of rows and columns in the rectangle.
    """
    rows_step = 1
    cols_step = 1
    if goal_corner_pos[0] < start_corner_pos[0]:
        rows_step = -1
    if goal_corner_pos[1] < start_corner_pos[1]:
        cols_step = -1
    start_row = start_corner_pos[0]
    start_col = start_corner_pos[1]

    res = [[int(food_grid[start_row + i * rows_step][start_col + j * cols_step]) for j in
            range(abs(goal_corner_pos[1] - start_col) + 1)] for i in range(abs(goal_corner_pos[0] - start_row) + 1)]
    return res


def hungerGamesManhattanAndMaxFoodOnShortestPathHeuristic(state, problem):
    """
    A heuristic for the HungerGamesSearchProblem, based on the following idea:
    - any path from the current state to the goal takes at least manhattan distance steps
    - if there is a path from the current position to the maze exit position,
    such that the manhattan distance <= current energy level + energy level gained from food dots on the path,
    then it's possible, that a path satisfying all problem constraints with cost manhattan distance exists
    - if there is no such path, then PacMan must step at least once in the "wrong direction".
    By wrong direction we mean that for obtaining a path with manhattan distance cost,
    if the maze exit is to the South from PacMan's position, then any step to the North is wrong and vice-versa.
    Similarly, if the exit is to the East from PacMan's position, then any step to the West is wrong, and vice-versa.
    Moreover, if PacMan takes one step to any wrong direction, than that step must be "recovered" later,
    i.e. annulled with a backwards step.
    Thus, we can guarantee that if no path with manhattan distance cost exists,
    then any path has the cost >= manhattan distance + 2.

    Note that this heuristic takes into account only the energy level required for the entire path to the goal,
    but does not verify whether there is enough energy at all steps of the path.

    To verify whether a path with the above characteristics exists, a dynamic programming approach is used,
    based on the formula
    max no. food dots to (x, y) = max(max no. food dots to (x-1, y), max no. food dots to (x, y-1)) +
                                 + no. food dots on position (x, y)
    Please refer to the documentation for more information.

    Heuristic identifier in the documentation: D
    """
    curr_pos = state[0]
    energy_level = state[1]
    food_grid = state[2]
    goal = problem.mazeExitPosition

    # compute in max_food_dot_grid[x][y] the maximum number of food dots that can be eaten by PacMan along a path from
    # (0, 0) to (x, y), with only steps to the right (y++) or to the left(x++).
    # O(n*m) dynamic programming algorithm, where n and m are the sizes of the grid
    max_food_dot_grid = buildGoalOrientedIntegerFoodGridRectangle(curr_pos, goal, food_grid)
    for j in range(1, len(max_food_dot_grid[0])):
        max_food_dot_grid[0][j] += max_food_dot_grid[0][j - 1]

    for i in range(1, len(max_food_dot_grid)):
        max_food_dot_grid[i][0] += max_food_dot_grid[i - 1][0]

    for i in range(1, len(max_food_dot_grid)):
        for j in range(1, len(max_food_dot_grid[i])):
            max_food_dot_grid[i][j] += max(max_food_dot_grid[i - 1][j], max_food_dot_grid[i][j - 1])

    # Find the maximum number of food dots that can be eaten by PacMan through a minimum-cost path
    # from the starting position (0, 0) to the goal.
    max_food_on_shortest_path = max_food_dot_grid[len(max_food_dot_grid) - 1][len(max_food_dot_grid[0]) - 1]
    shortest_path_length = manhattanDistance(curr_pos, goal)

    # If by eating the maximum amount of food dots that can be found on a minimum cost path PacMan still wouldn't have
    # enough energy to reach the goal, then at least 1 step in the wrong direction + 1 annulment step is needed,
    # additionally to the cost of manhattan distance.
    if energy_level + problem.foodEnergyLevel * max_food_on_shortest_path < shortest_path_length:
        # Not enough food on any minimum cost path
        return shortest_path_length + 2
    else:
        # Possibly enough food on the shortest path
        return shortest_path_length


def buildMaxEnergyLevelGrid(init_energy_level, food_grid, food_energy_level):
    """
    Given
    - the initial energy of PacMan, assumed to be located in position (0, 0) of the food_grid,
    - the energy level given by a food dot (food_energy_level)
    - the food_grid with the maze exit situated in the top-right corner of the grid,
    and the current position of PacMan being (0, 0).
    Computes the maximum energy level which PacMan may have when leaving location (x, y) of the grid, assuming that
    PacMan reached this location via a path from (0, 0) with x+y steps (i.e. a minimum-cost path with steps only into
    the correct directions).
    Returns a matrix with the values for all (x, y) locations (of the same size as food_grid).

    If a location (x0, y0) is not reachable according to the rules of HungerGames, with only steps into the correct
    directions, then -1 is placed on the given location.

    Note: the "leaving energy" is computed, not the "reaching energy", meaning that the energy given by the potential
    doos dot on psotion (x, y) is added to the value computed for location (x, y).

    For the computations, a dynamic programming algorithm is used, based on the formula
    max energy on (x, y) = max(max energy on (x-1, y), max energy on (x, y-1)) +
                           + the energy given by the food dots on location (x, y)
                           - 1
    Where -1 is due to the cost of stepping from (x-1, y) or from (x, y-1) to (x, y).
    """
    from copy import deepcopy
    max_energy_level_grid = deepcopy(food_grid)
    max_energy_level_grid[0][0] = init_energy_level

    for j in range(1, len(max_energy_level_grid[0])):
        if max_energy_level_grid[0][j - 1] > 0:
            max_energy_level_grid[0][j] = max_energy_level_grid[0][j - 1] + max_energy_level_grid[0][
                j] * food_energy_level - 1
        else:
            max_energy_level_grid[0][j] = -1

    for i in range(1, len(max_energy_level_grid)):
        if max_energy_level_grid[i - 1][0] > 0:
            max_energy_level_grid[i][0] = max_energy_level_grid[i - 1][0] + max_energy_level_grid[i][
                0] * food_energy_level - 1
        else:
            max_energy_level_grid[i][0] = -1

    for i in range(1, len(max_energy_level_grid)):
        for j in range(1, len(max_energy_level_grid[i])):
            max_parent_energy_level = max(max_energy_level_grid[i - 1][j], max_energy_level_grid[i][j - 1])
            if max_parent_energy_level > 0:
                max_energy_level_grid[i][j] = max_parent_energy_level + max_energy_level_grid[i][
                    j] * food_energy_level - 1
            else:
                max_energy_level_grid[i][j] = -1

    return max_energy_level_grid


def hungerGamesManhattanShortestPathVerificationHeuristic(state, problem):
    """
    A heuristic for the HungerGamesSearchProblem, which adds an improvement to
    hungerGamesManhattan2MaxFoodOnShortestPathHeuristic, in that in doesn't only verify whether there exists any
    minimum-cost (=manhattan distance) path from the current state to the goal such that
    manhattan distance <= current energy level + energy level gained from food dots on the path,
    but it considers only the minimum-cost paths along which PacMan does not reach an energy level of 0 at any point.

    To verify whether a path with the above characteristics exists, a dynamic programming approach is used,
    as explained in buildMaxEnergyLevelGrid.
    Please refer to the documentation for more information.

    Heuristic identifier in the documentation: E
    """
    curr_pos = state[0]
    energy_level = state[1]
    food_grid = state[2]
    goal = problem.mazeExitPosition

    goal_oriented_food_grid = buildGoalOrientedIntegerFoodGridRectangle(curr_pos, goal, food_grid)

    max_energy_level_grid = buildMaxEnergyLevelGrid(energy_level, goal_oriented_food_grid, problem.foodEnergyLevel)

    shortest_path_length = manhattanDistance(curr_pos, goal)

    if max_energy_level_grid[len(max_energy_level_grid) - 1][len(max_energy_level_grid[0]) - 1] < 0:
        # Not enough food
        return shortest_path_length + 2
    else:
        # Possibly enough food on the shortest path
        return shortest_path_length


def buildMinEnergyLevelGrid(food_grid, food_energy_level):
    """
    Given
    - the energy level given by a food dot (food_energy_level)
    - the food_grid with the maze exit situated in the (n-2, m-2) location,
    where n and m give the height and the width of the grid
    Computes the minimum energy level which PacMan must have when reaching location (x, y) of the grid,
    such that a valid, minimum cost (=manhattan distance((x, y), goal) path to the goal (n-2, m-2),
    according to the rules of HungerGames, exists.

    Returns a matrix with the values for all (x, y) locations (of the same size as food_grid).

    If the energy level when reaching (x, y) does not matter, because the energy gained from the food dot on (x, y)
    is enough anyway, then a 0 value is assigned to location (x, y).

    For the computations, a dynamic programming algorithm is used, based on the formula
    min energy when (x, y) = min(min energy when reaching (x+1, y), min energy when reaching (x, y-1)) +
                          - the energy given by the food dots on location (x, y)
                          + 1
    Where +1 is due to the cost of stepping from (x, y) to (x, y+1) or to (x+1, y).
    """
    # assumes minimum 3 rows and 3 columns in the grid
    no_rows = len(food_grid)
    no_cols = len(food_grid[0])
    from copy import deepcopy
    min_energy_level_grid = deepcopy(food_grid)
    min_energy_level_grid[no_rows - 1][no_cols - 1] = 0
    min_energy_level_grid[no_rows - 2][no_cols - 2] = 0

    for j in range(no_cols - 2, -1, -1):
        min_energy_level_grid[no_rows - 1][j] = max(0, min_energy_level_grid[no_rows - 1][j + 1] - food_energy_level *
                                                    food_grid[no_rows - 1][j] + 1)

    for j in range(no_cols - 3, -1, -1):
        min_energy_level_grid[no_rows - 2][j] = max(0, min_energy_level_grid[no_rows - 2][j + 1] - food_energy_level *
                                                    food_grid[no_rows - 2][
                                                        j] + 1)

    for i in range(no_rows - 2, -1, -1):
        min_energy_level_grid[i][no_cols - 1] = max(0, min_energy_level_grid[i + 1][no_cols - 1] - food_energy_level *
                                                    food_grid[i][no_cols - 1] + 1)

    for i in range(no_rows - 3, -1, -1):
        min_energy_level_grid[i][no_cols - 2] = max(0, min_energy_level_grid[i + 1][no_cols - 2] - food_energy_level *
                                                    food_grid[i][
                                                        no_cols - 2] + 1)

    for i in range(no_rows - 3, -1, -1):
        for j in range(no_cols - 3, -1, -1):
            min_child_energy_level = min(min_energy_level_grid[i][j + 1], min_energy_level_grid[i + 1][j])
            min_energy_level_grid[i][j] = max(0, min_child_energy_level - food_energy_level * food_grid[i][j] + 1)

    return min_energy_level_grid


def extendMatrixWith0sOnAllSides(m):
    """
    Returns a matrix with 2 additional rows (the first and the last one) and 2 additional columns (the first and the
    last one) compared to m, such that the middle values are taken from m and the additional rows and columns are filled
    with 0s.
    """
    m.insert(0, [0 for i in range(len(m[0]))])
    m.append([0 for i in range(len(m[0]))])
    for row in m:
        row.insert(0, 0)
        row.append(0)
    return m


def extendRectangeCornersInEachDirection(corner1, corner2):
    """
    Given twe two tuples corner1 and corner2, both in the format (x, y), representing 2 diagonal corners of a rectangle,
    computes the coordinates of a rectangle extended with 1 row and 1 column on each side, and returns the coordinates
    of this extended rectangle.
    """
    x1, y1 = corner1
    x2, y2 = corner2

    if x2 < x1:
        x1 = x1 + 1
        x2 = x2 - 1
    else:
        x1 = x1 - 1
        x2 = x2 + 1
    if y2 < y1:
        y1 = y1 + 1
        y2 = y2 - 1
    else:
        y1 = y1 - 1
        y2 = y2 + 1

    return (x1, y1), (x2, y2)


def hungerGamesManhattanShortestPathWith1WrongStepVerificationHeuristic(state, problem):
    """
    A heuristic for the HungerGamesSearchProblem, which adds an improvement to
    hungerGamesManhattan2ShortestPathVerificationHeuristic, in that it doesn't only verify whether a minimum-cost path
    to the goal exists, and returns minimum cost + 2 for all other cases, but also verifies whether a path with cost
    manhattan distance + 2 exists, and returns manhattan distance + 4 for all other cases.

    To implement this verification, additionally to the methods in
    hungerGamesManhattan2ShortestPathVerificationHeuristic, it was verified whether 1 single step into a wrong direction
    + its annulment step is enough to reach the goal while fulfilling the constrains of HungerGamesSearchProblem.
    For this
    - the maximum possible energy level of PacMan was computed when leaving the position (x, y), assuming that PacMan
    reaches (x, y) through a path with steps only into a correct direction, from its current position.
    See buildMaxEnergyLevelGrid.
    - the minimum required energy level of PacMan when reaching position (a, b) was computed, such that the goal is
    reachable from (a, b) through a minimum-cost path (only steps in the correct direction)
    See buildMinEnergyLevelGrid.
    Thus,
    1. if the maximum possible energy level at the goal is >= 0, then and only then a path with only correct steps exists
    --> cost = manhattan distance
    2. if there is (x, y) and (a, b) such that they are neighboring positions, (a, b) is at one wrong step from
    (x, y), and the maximum energy at (x, y) - 1 >= the minimum required energy at (a, b), then (and only then)
    it is sure, that a path with just one wrong step and its annulment exists
    --> cost = manhattan distance + 2
    3. otherwise. cost >= manhattan distance + 4

    Heuristic identifier in the documentation: F
    """
    curr_pos = state[0]
    energy_level = state[1]
    food_grid = state[2]
    goal_pos = problem.mazeExitPosition

    goal_oriented_food_grid = buildGoalOrientedIntegerFoodGridRectangle(curr_pos, goal_pos, food_grid)

    max_energy_level_grid = buildMaxEnergyLevelGrid(energy_level, goal_oriented_food_grid, problem.foodEnergyLevel)

    shortest_path_length = manhattanDistance(curr_pos, goal_pos)

    if max_energy_level_grid[len(max_energy_level_grid) - 1][len(max_energy_level_grid[0]) - 1] < 0:
        # Not enough food on the shortest path. Try with 1 step to a wrong direction
        extended_curr_pos, extended_goal_pos = extendRectangeCornersInEachDirection(curr_pos, goal_pos)

        extended_goal_oriented_food_grid = buildGoalOrientedIntegerFoodGridRectangle(extended_curr_pos,
                                                                                     extended_goal_pos,
                                                                                     food_grid)
        # extend max_energy_level matrix with 0s
        max_energy_level_grid = extendMatrixWith0sOnAllSides(max_energy_level_grid)

        min_energy_level_grid = buildMinEnergyLevelGrid(extended_goal_oriented_food_grid, problem.foodEnergyLevel)

        last_inside_col = len(max_energy_level_grid[0]) - 2
        last_inside_row = len(max_energy_level_grid) - 2

        for i in range(1, last_inside_row):
            for j in range(1, last_inside_col):
                if min_energy_level_grid[i][j-1] + 1 <= max_energy_level_grid[i][j]:
                    return shortest_path_length + 2
                if min_energy_level_grid[i-1][j] + 1 <= max_energy_level_grid[i][j]:
                    return shortest_path_length + 2

        for i in range(1, last_inside_row):
            if min_energy_level_grid[i][last_inside_col - 1] + 1 <= max_energy_level_grid[i][last_inside_col]:
                return shortest_path_length + 2
            if min_energy_level_grid[i - 1][last_inside_col] + 1 <= max_energy_level_grid[i][last_inside_col]:
                return shortest_path_length + 2
            if min_energy_level_grid[i][last_inside_col + 1] + 1 <= max_energy_level_grid[i][last_inside_col]:
                return shortest_path_length + 2

        for j in range(1, last_inside_col):
            if min_energy_level_grid[last_inside_row - 1][j] + 1 <= max_energy_level_grid[last_inside_row][j]:
                return shortest_path_length + 2
            if min_energy_level_grid[last_inside_row][j - 1] + 1 <= max_energy_level_grid[last_inside_row][j]:
                return shortest_path_length + 2
            if min_energy_level_grid[last_inside_row + 1][j] + 1 <= max_energy_level_grid[last_inside_row][j]:
                return shortest_path_length + 2

        return shortest_path_length + 4
    else:
        # Possibly enough food on the shortest path
        return shortest_path_length


def hungerGamesManhattanAndStepsOutsideRectangleHeuristic(state, problem=None):
    # TODO: Bea --> documentation
    """
    Heuristic identifier in the documentation: C
    """
    (curr_position, curr_energy_level, food_grid) = state
    goal = problem.mazeExitPosition

    dist_to_exit = manhattanDistance(curr_position, goal)
    needed_energy = dist_to_exit - curr_energy_level

    # if the current energy level is not enough to reach the exit, pacman tries to accumulate food dots along the way;
    # estimate how far does pacman need to step out from the initial shortest path,
    # whose length is given by the manhattan distance;
    if needed_energy > 0 and len(food_grid.asList()) > 0:
        import math
        needed_food = int(math.ceil(float(needed_energy) / problem.foodEnergyLevel))
        no_food_dots_inside_rectangle = noFoodDotsInRectange(curr_position, goal, food_grid)

        if no_food_dots_inside_rectangle >= needed_food:
            return dist_to_exit
        else:
            remaining_needed_food = needed_food - no_food_dots_inside_rectangle
            d = 1
            no_food_dots_outside_rectangle = 0
            rectangle_bottom_left_x = min(curr_position[0], goal[0])
            rectangle_bottom_left_y = min(curr_position[1], goal[1])
            rectangle_top_right_x = max(curr_position[0], goal[0])
            rectangle_top_right_y = max(curr_position[1], goal[1])

            while no_food_dots_outside_rectangle < remaining_needed_food:
                # extend the perimeter on which we search for food
                if rectangle_bottom_left_x - 1 >= 0:
                    rectangle_bottom_left_x -= 1
                if rectangle_bottom_left_y - 1 >= 0:
                    rectangle_bottom_left_y -= 1
                if rectangle_top_right_x + 1 < problem.walls.width:
                    rectangle_top_right_x += 1
                if rectangle_top_right_y + 1 < problem.walls.height:
                    rectangle_top_right_y += 1

                # no of food dots on the perimeter at distance d
                no_food_dots_on_perimeter = 0
                for x in range(rectangle_bottom_left_x, rectangle_top_right_x):
                    no_food_dots_on_perimeter += food_grid[x][rectangle_bottom_left_y]
                    no_food_dots_on_perimeter += food_grid[x][rectangle_top_right_y]

                for y in range(rectangle_bottom_left_y + 1, rectangle_top_right_y - 1):
                    no_food_dots_on_perimeter += food_grid[rectangle_bottom_left_x][y]
                    no_food_dots_on_perimeter += food_grid[rectangle_top_right_x][y]

                no_food_dots_outside_rectangle += no_food_dots_on_perimeter
                d += 1

            # pacman had to step out at least 2 times on a distance d from the original rectangle to gather enough food supply
            return dist_to_exit + 2 * d
    else:
        return dist_to_exit


def hungerGamesClosestFoodDotReachableHeuristic(state, problem):
    """
    A heuristic for the HungerGamesSearchProblem, which verifies whether any food dot is reachable from the current
    position of PacMan with the current energy level.

    If yes, the heuristic returns 0, otherwise infinity (or a very large value, specified in the problem definition).

    Heuristic identifier in the documentation: G
    """
    (curr_position, curr_energy_level, food_grid) = state
    goal = problem.mazeExitPosition

    if manhattanDistance(goal, curr_position) <= curr_energy_level:
        return 0

    for food_dot in food_grid.asList():
        if manhattanDistance(food_dot, curr_position) <= curr_energy_level:
            return 0

    return HungerGamesSearchProblem.IMPOSSIBLE_TO_SOLVE_STATE_HEURISTIC_VALUE


def hungerGamesCombinedHeuristic(state, problem):
    """
    A heuristic for the HungerGamesSearchProblem which combines multiple previously defined heuristics:
    - if PacMan doesn't have enough energy to reach the goal and the closest food dot either, then PacMan cannot succeed
    --> based on hungerGamesClosestFoodDotReachableHeuristic, a very high value is returned
    - if the number of the food dots in the rectangle between the current position of PacMan and the goal position
    contains enough food dots to cover PacMan's energy requirements to the goal,
    assuming a path with manhattan distance steps, then
    hungerGamesManhattanShortestPathWith1WrongStepVerificationHeuristic(state, problem) is returned
    - if the number of the food dots in the rectangle between the current position of PacMan and the goal position does
    not contain enough food dots to cover PacMan's energy requirements to the goal, even if
    a path with manhattan distance steps is assumed, then
    hungerGamesManhattanAndStepsOutsideRectangleHeuristic(state, problem) is returned.

    Heuristic identifier in the documentation: H
    """
    (curr_position, curr_energy_level, food_grid) = state
    goal = problem.mazeExitPosition

    dist_to_exit = hungerGamesManhattanHeuristic(state, problem)
    needed_energy = dist_to_exit - curr_energy_level

    if needed_energy > 0 and hungerGamesClosestFoodDotReachableHeuristic(state,
                                                                         problem) == HungerGamesSearchProblem.IMPOSSIBLE_TO_SOLVE_STATE_HEURISTIC_VALUE:
        return HungerGamesSearchProblem.IMPOSSIBLE_TO_SOLVE_STATE_HEURISTIC_VALUE

    needed_food = needed_energy // problem.foodEnergyLevel
    no_food_dots_inside_rectangle = noFoodDotsInRectange(curr_position, goal, food_grid)

    if no_food_dots_inside_rectangle >= needed_food:
        return hungerGamesManhattanShortestPathWith1WrongStepVerificationHeuristic(state, problem)
    else:
        # if the current energy level is not enough to reach the exit, pacman tries to accumulate food dots along the way;
        # estimate how far does pacman need to step out from the initial shortest path,
        # whose length is given by the manhattan distance;
        return hungerGamesManhattanAndStepsOutsideRectangleHeuristic(state, problem)

"""=============================================END OF MY OWN CODE==================================================="""

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    return 0  # Default to trivial solution


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
