# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        import math
        import random
        exp = math.exp
        metric = util.manhattanDistance
        oldPos = currentGameState.getPacmanPosition()
        nearFoodDist = 100
        for i, item in enumerate(newFood):
            for j, foodItem in enumerate(item):
                nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
        capsules = successorGameState.getCapsules()
        foodCount = newFood.count(True)
        nearGhostDist = min(metric(newPos, ghost.getPosition()) for ghost in newGhostStates)
        posPenalty = (-10) * (oldPos == newPos)

        if(foodCount ==0):
            return 9999

        if newScaredTimes[0] <= 4:
            if not capsules:
                score = 400 * exp(-0.001 * nearFoodDist) + (-20) * foodCount + (-300) * exp(-1 * nearGhostDist) #+ random.randrange(-10, 0)
            else:
                nearCapDist = min(metric(newPos, capsules[i]) for i in range(0, len(capsules)))
                capRem = len(capsules)
                score = (-100) * capRem + (-2) * nearCapDist + 400 * exp(-0.001 * nearFoodDist) \
                        + (-20) * foodCount + (-300) * exp(-0.9 * nearGhostDist) #+ random.randrange(-10, 0)

        else:
            score = 400 * exp(-0.001 * nearFoodDist)+ (-20) * foodCount #+ random.randrange(-10, 0)
        print score

        return score




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def getValue(gameState, depth):
            if gameState.isWin() or gameState.isLose():#terminal state test
                return gameState.getScore()
            agentNum = gameState.getNumAgents()
            if depth % agentNum == 0:
                return maxAgent(gameState, depth)
            else:
                return minAgent(gameState, depth)

        def maxAgent(gameState, depth):
            if depth == self.depth * gameState.getNumAgents(): # important step, figure out what "depth" means
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            valueList = []
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                valueList.append(getValue(nextGameState, depth + 1))
            maxVal = max(valueList)
            return maxVal, actions[valueList.index(maxVal)]

        def minAgent(gameState, depth):
            if depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            valueList = []
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                valueList.append(getValue(nextGameState, depth + 1))
            minVal = min(valueList)
            return minVal, actions[valueList.index(minVal)]

        value, bestAction = getValue(gameState, 0)
        return bestAction


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import types
        Alpha = -99999
        Beta = 99999
        def getValue(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose():#terminal state test
                return gameState.getScore()
            agentNum = gameState.getNumAgents()
            if depth % agentNum == 0:
                return maxAgent(gameState, depth, alpha, beta)
            else:
                return minAgent(gameState, depth, alpha, beta)

        def maxAgent(gameState, depth, alpha, beta):
            if depth == self.depth * gameState.getNumAgents(): # important step, figure out what "depth" means
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)

            maxVal = -9999
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                tvalue = getValue(nextGameState, depth + 1, alpha, beta)
                if type(tvalue) is types.FloatType:# because the return type can be tuple
                                                   # need to judge; I konw it's ugly, but it works
                    value = tvalue
                else:
                    value = tvalue[0]
                if maxVal < value:
                    maxVal = value
                    maxAction = action
                #print "Depth: ", depth, "Value:", value, type(value), "maxVal: ", maxVal, "alpha: " , alpha, "beta: " , beta
                if maxVal > beta:
                    return maxVal, maxAction

                alpha = max(alpha, maxVal)
                #print "Depth: ", depth, "Value:", value, type(value), "maxVal: ", maxVal, "alpha: " , alpha, "beta: " , beta
            return maxVal, maxAction

        def minAgent(gameState, depth, alpha, beta):
            if depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)

            minVal = 9999
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                tvalue = getValue(nextGameState, depth + 1, alpha, beta)
                if type(tvalue) is types.FloatType:
                    value = tvalue
                else:
                    value = tvalue[0]
                if minVal > value:
                    minVal = value
                #print "Depth: ", depth, "Value:", value, type(value),"minVal: ", minVal, "alpha: " , alpha, "beta: " , beta
                if minVal < alpha:
                    return minVal

                beta = min(beta, minVal)
                #print "Depth: ", depth, "Value:", value, type(value),"minVal: ", minVal, "alpha: " , alpha, "beta: " , beta
            return minVal

        value, bestAction = getValue(gameState, 0, Alpha, Beta)
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        import types

        def getValue(gameState, depth):
            if gameState.isWin() or gameState.isLose():#terminal state test
                return gameState.getScore()
            agentNum = gameState.getNumAgents()
            if depth % agentNum == 0:
                return maxAgent(gameState, depth)
            else:
                return expAgent(gameState, depth)

        def maxAgent(gameState, depth):
            if depth == self.depth * gameState.getNumAgents(): # important step, figure out what "depth" means
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            valuelist = []
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                tvalue = getValue(nextGameState, depth + 1)
                if type(tvalue) is types.FloatType:# because the return type can be tuple
                    value = tvalue                 # need to judge; I konw it's ugly, but it works
                else:
                    value = tvalue[0]
                valuelist.append(value)

            maxVal = max(valuelist)
            maxAction = actions[valuelist.index(maxVal)]
            return maxVal, maxAction

        def expAgent(gameState, depth):
            if depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            agentIndex = depth % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)

            valuelist = []
            for action in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                tvalue = getValue(nextGameState, depth + 1)
                if type(tvalue) is types.FloatType:
                    value = tvalue
                else:
                    value = tvalue[0]
                valuelist.append(value)
            expVal = 1.0 * sum(valuelist)/ (1.0 * len(valuelist))
            return expVal

        value, bestAction = getValue(gameState, 0)
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    import math
    import random
    exp = math.exp
    metric = util.manhattanDistance
    nearFoodDist = 100
    for i, item in enumerate(newFood):
        for j, foodItem in enumerate(item):
            nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
    capsules = currentGameState.getCapsules()
    foodCount = newFood.count(True)
    nearGhostDist = min(metric(newPos, ghost.getPosition()) for ghost in newGhostStates)


    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999
    if newScaredTimes[0] <= 3:
        if not capsules:
            score = (-30) * exp(0.01 * nearFoodDist) + (-10) * foodCount + (-4000) * exp(-1.2 * nearGhostDist) + random.randrange(-5, 0)
        else:
            nearCapDist = min(metric(newPos, capsules[i]) for i in range(0, len(capsules)))
            capRem = len(capsules)
            score = (-100) * capRem + (-2) * nearCapDist + (-30) * exp(0.01 * nearFoodDist) \
                    + (-10) * foodCount + (-4000) * exp(-1.2 * nearGhostDist) + random.randrange(-5, 0)

    else:
        score = (-30) * exp(-0.01 * nearFoodDist)+ 1 * exp(-1 * nearGhostDist) +(-10) * foodCount + random.randrange(-5, 0)
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
