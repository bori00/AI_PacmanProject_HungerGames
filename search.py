# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    from game import Directions
    # The next state to be expanded
    curr_state = problem.getStartState()
    # The collection of states to which an existing path from the initial state is known.
    # The cost of each state, by which the priority queue is ordered, is the minimum known cost to reach the state from the initial state.
    frontier = PriorityQueue()
    # The set of states for which the least expensive path form the initial state to them was already found
    finished_states = {curr_state}
    # Just a notation used for the parent of the initial state, marking that the initial state has no parent state.
    root_state = (-1)
    # A dictionary, which for a key state gives as a value a tuple, in which
    # - the first element is the parent state through which the key state was reached such that the total cost of the pah from the initial state was minimal,
    # - the second element gives the action taken from this parent state to reach the key state
    # - third element is the total path cost from the parent state to the key state.
    parent_state_and_g_cost_dict = {curr_state: (root_state, Directions.STOP, 0)}

    while not problem.isGoalState(curr_state):
        finished_states.add(curr_state)
        possible_successor_states_data = problem.getSuccessors(curr_state)
        curr_state_g_cost = parent_state_and_g_cost_dict[curr_state][2]
        for possible_successor_state_data in possible_successor_states_data:
            if possible_successor_state_data[0] not in finished_states:
                possible_successor_state = possible_successor_state_data[0]
                action_to_possible_successor_state = possible_successor_state_data[1]
                cost_curr_state_successor_state = possible_successor_state_data[2]
                # Compute the estimated cost of a path through the possible successor state
                f_cost = curr_state_g_cost + cost_curr_state_successor_state + heuristic(possible_successor_state,
                                                                                         problem)
                # Update the estimated path cost through the possible successor state in the frontier
                frontier.update(possible_successor_state, f_cost)
                # If the previously best known cost from the initial state to the possible successor state is larger than
                # the newly found cost, then update the cost in the dictionary
                if possible_successor_state not in parent_state_and_g_cost_dict or \
                        parent_state_and_g_cost_dict[possible_successor_state][
                            2] > curr_state_g_cost + cost_curr_state_successor_state:
                    parent_state_and_g_cost_dict[possible_successor_state] = (
                        curr_state, action_to_possible_successor_state,
                        curr_state_g_cost + cost_curr_state_successor_state)
        if frontier.isEmpty():
            # No solution exists
            return []
        # Expand the node with the best estimated cost next
        curr_state = frontier.pop()

    # Reconstruct the list of steps to be taken from the initial state to the minimum cost goal state
    # from the parent array constructed while searching.
    step_list = []
    while parent_state_and_g_cost_dict[curr_state][0] is not root_state:
        step_list.insert(0, parent_state_and_g_cost_dict[curr_state][1])
        curr_state = parent_state_and_g_cost_dict[curr_state][0]

    return step_list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
