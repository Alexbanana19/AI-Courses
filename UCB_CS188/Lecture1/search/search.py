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
    return  [s, s, w, s, w, w, s, w]

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
    """from game import Directions
    visit = []
    action = []
    mem = {}

    startstate = problem.getStartState()
    state = (startstate, Directions.STOP, 0)

    stack = util.Stack()
    stack.push(state)
    visit.append(state[0])
    while not stack.isEmpty():
        top = stack.pop()
        stack.push(top)

        if problem.isGoalState(top[0]):
            while not stack.isEmpty():
                top = stack.pop()
                action.append(top[1])
            action.reverse()
            del action[0]
            return action

        statelist = problem.getSuccessors(top[0])
        mem[top[0]] = statelist

        if statelist == []:
            stack.pop()
            top = stack.pop()
            stack.push(top)
            temp = mem[top[0]]
            for i in range(0, len(temp)):
                if temp[i][0] not in visit:
                    stack.push(temp[i])
                    visit.append(temp[i][0])
                    break
            continue

        statelist.reverse()
        flag = 0
        for i in range(0, len(statelist)):
            if statelist[i][0] not in visit:
                stack.push(statelist[i])
                visit.append(statelist[i][0])
                flag = 1
                break

        if flag == 0:
            stack.pop()

    return []
    util.raiseNotDefined()"""
    fringe = util.Stack()
    current_state = [problem.getStartState(), []]
    successors = None
    visited_states = set()
    item = None
    while not problem.isGoalState(current_state[0]):
        (current_pos, directions) = current_state
        successors = problem.getSuccessors(current_pos)
        for item in successors:
            fringe.push((item[0], directions + [item[1]]) )
        while True:
            if fringe.isEmpty():
                return None
            item = fringe.pop()
            if item[0] not in visited_states:
                break
        current_state = item
        visited_states.add(item[0])
    # print current_state[1]
    return current_state[1]
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """from game import Directions
    queue = util.Queue()
    action = []
    mem = {}
    visit = []

    startstate = problem.getStartState()
    state = (startstate, Directions.STOP, 0)
    queue.push(state)
    visit.append(startstate)

    while True:
        top = queue.pop()
        if problem.isGoalState(top[0]):
            while top[0] != startstate :
                action.append(top[1])
                top = mem[top[0]]
            action.reverse()
            return action

        statelist = problem.getSuccessors(top[0])

        for i in range(0, len(statelist)):
            if statelist[i][0] not in visit:
                mem[statelist[i][0]] = top
                queue.push(statelist[i])
                if not problem.isGoalState(statelist[i][0]):
                    visit.append(statelist[i][0])"""
    fringe = util.Queue()
    current_state = [problem.getStartState(), []]
    successors = None
    visited_states = set()
    visited_states.add(current_state[0])
    item = None
    while not problem.isGoalState(current_state[0]):
        (current_pos, directions) = current_state
        successors = problem.getSuccessors(current_pos)
        for item in successors:
            fringe.push((item[0], directions + [item[1]]) )
        while True:
            if fringe.isEmpty():
                return None
            item = fringe.pop()
            if item[0] not in visited_states:
                break
        #print item
        current_state = item
        visited_states.add(item[0])
    # print current_state[1]
    return current_state[1]
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """from game import Directions
    pqueue = util.PriorityQueue()
    action = []
    mem = {}
    cost = {}
    visit = []

    startstate = problem.getStartState()
    state = (startstate, Directions.STOP, 0)
    pqueue.push(state, state[2])
    visit.append(startstate)
    cost[startstate] = 0

    while True:
        top = pqueue.pop()

        if problem.isGoalState(top[0]):
            while top[0] != startstate :
                action.append(top[1])
                top = mem[top[0]]
            action.reverse()
            return action

        statelist = problem.getSuccessors(top[0])

        for i in range(0, len(statelist)):
            if statelist[i][0] not in visit:
                mem[statelist[i][0]] = top
                cost[statelist[i][0]] = cost[top[0]] + statelist[i][2]
                pqueue.push(statelist[i], cost[statelist[i][0]])
                if not problem.isGoalState(statelist[i][0]):
                    visit.append(statelist[i][0])"""
    fringe = util.PriorityQueue()
    current_state = [problem.getStartState(), [], 0]
    successors = None
    visited_states = set()
    visited_states.add(current_state[0])
    item = None
    while not problem.isGoalState(current_state[0]):
        (current_pos, directions, cost) = current_state
        successors = problem.getSuccessors(current_pos)
        for item in successors:
            # print 'BBB', item
            fringe.push((item[0], directions + [item[1]], cost + item[2]), cost + item[2] )
        while True:
            if fringe.isEmpty():
                return None
            item = fringe.pop()
            if item[0] not in visited_states:
                break
        # print 'AAA', item
        current_state = (item[0], item[1], item[2])
        visited_states.add(item[0])
    # print current_state[1]
    return current_state[1]

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """from game import Directions
    pqueue = util.PriorityQueue()
    action = []
    mem = {}
    cost = {}
    visit = []

    startstate = problem.getStartState()
    state = (startstate, Directions.STOP, 0)
    pqueue.push(state, state[2])
    visit.append(startstate)
    cost[startstate] = 0

    while True:
        top = pqueue.pop()

        if problem.isGoalState(top[0]):
            while top[0] != startstate :
                action.append(top[1])
                top = mem[top[0]]
            action.reverse()
            return action

        statelist = problem.getSuccessors(top[0])

        for i in range(0, len(statelist)):
            if statelist[i][0] not in visit:
                mem[statelist[i][0]] = top
                cost[statelist[i][0]] = cost[top[0]] + statelist[i][2]
                pqueue.push(statelist[i], cost[statelist[i][0]] + \
                            heuristic(statelist[i][0], problem))
                if not problem.isGoalState(statelist[i][0]):
                    visit.append(statelist[i][0])"""
    fringe = util.PriorityQueue()
    current_state = [problem.getStartState(), [], 0]
    successors = None
    visited_states = set()
    visited_states.add(current_state[0])
    item = None

    while not problem.isGoalState(current_state[0]):
        (current_pos, directions, cost) = current_state
        successors = problem.getSuccessors(current_pos)
        for item in successors:
            # print 'BBB', item
            fringe.push((item[0], directions + [item[1]], cost + item[2]), cost + item[2] + heuristic(item[0], problem))
        while True:
            if fringe.isEmpty():
                return None
            item = fringe.pop()
            if item[0] not in visited_states:
                break
        # print 'AAA', item
        current_state = (item[0], item[1], item[2])
        visited_states.add(item[0])
    # print current_state, '\n', len(current_state[1])
    # g = len(current_state[1])
    return current_state[1]

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
