# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    # print newGhostStates
    metric = util.manhattanDistance
    score = 0 # = successorGameState.getScore()
    punishGhostLambdas = {0: -7000, 1: -1000, 2: -30, 3: -10, 4:-4, 5:-2}
    nearFoodBonusDict = {0: 30, 1: 20, 2: 12, 3:7, 4:4}
    foodRemPunishK = -20
    foodCount = newFood.count(True)
    if(foodCount ==0):
        return 9999
    nearFoodDist = 100
    for i, item in enumerate(newFood):
        for j, foodItem in enumerate(item):
            nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
    nearFoodBonus = nearFoodBonusDict[nearFoodDist] if nearFoodDist in nearFoodBonusDict else (3 + 1/nearFoodDist)
    foodRemPunish = foodRemPunishK*foodCount
    print foodCount, nearFoodDist
    ghostDistances = [metric(newPos, hh.getPosition()) for hh in newGhostStates]
    ghostK = sum([punishGhostLambdas[dist] for dist in ghostDistances if dist in punishGhostLambdas])

    score = score + nearFoodBonus + ghostK + foodRemPunish*foodCount
    print "score: ", score, ghostK
    return score
    # return successorGameState.getScore()

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    #print "--------------NEW CALL-------------"

    numAgents =  gameState.getNumAgents()
    #aIndex = lambda aC: aC%numAgents
    indexMod = lambda aC: aC%numAgents
    isPacman = lambda i:indexMod(i) == 0

    #ghost_f = lambda state : min(state, key= lambda item:item[1])
    #pacman_f = lambda state : max(state, key= lambda item:item[1])
    ghost_f = min
    pacman_f = max
    get_f= lambda counter: pacman_f if isPacman(counter) else ghost_f
    #def max_val(gameState, aC):
    #    return get_f(aC)([gameState.generateSuccessor(aC%numAgents, action)])


    def get_val(gState, aCounter):
        #if aIndex(aCounter) ==0:
        #    print "aCounter: ", aCounter, gState.isWin(), gState.isLose(), self.evaluationFunction(gState)
        #print " "*aCounter,   "GO DEEPER! passing:", aCounter, self.evaluationFunction(gState)

        if gState.isWin() or gState.isLose():
            return gState.getScore()
        if aCounter == self.depth*numAgents:
            return self.evaluationFunction(gameState)
        actions = [action for action in gState.getLegalActions(indexMod(aCounter)) if action != Directions.STOP]
        #if isPacman(aCounter):
        #    print " "*aCounter, "aCounter:",aCounter, "PACMAN" if isPacman(aCounter) else "GHOST" , "actionsLen:", len(actions), actions
        leafs = [get_val(gState.generateSuccessor(indexMod(aCounter), action), aCounter +  1) for action in actions ]
        ret = get_f(aCounter)(leafs)

        #print " "*aCounter, "aCounter:",aCounter, "PACMAN" if isPacman(aCounter) else "GHOST" , "actionsLen:", len(actions), actions, "ret:", ret, "leafs:", leafs
        return ret

    acs = [(get_val(gameState.generateSuccessor(0, action), 1), action) for action in gameState.getLegalActions(0) if action != Directions.STOP]

    #print max(acs)

    #raw_input('pause')
    #print max(acs)
    print max(acs)[1]
    return max(acs)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    numAgents =  gameState.getNumAgents()
    indexMod = lambda aC: aC%numAgents
    isPacman = lambda i:indexMod(i) == 0

    #ghost_f = lambda state : min(state, key= lambda item:item[1])
    #pacman_f = lambda state : max(state, key= lambda item:item[1])
    ghost_f = min
    pacman_f = max
    PLUS_INF = float("+inf")
    MINUS_INF = float("-inf")
    get_f= lambda counter: pacman_f if isPacman(counter) else ghost_f
    #print "--------------NEW CALL-------------"
    def get_val(gState, alpha, beta, aCounter):
        if gState.isWin() or gState.isLose():
            #print " "*aCounter, aCounter
            return gState.getScore(), []
        if aCounter == self.depth*numAgents:
            return self.evaluationFunction(gState), []
        #v = alpha if isPacman(aCounter) else beta
        #v = MINUS_INF if isPacman(aCounter) else PLUS_INF
        alpha_v = alpha
        alpha_action = []
        beta_v = beta
        beta_action = []
        #print " "*aCounter, aCounter, "PACMAN" if isPacman(aCounter) else "GHOST", alpha, beta
        if isPacman(aCounter):
            for action in gState.getLegalActions(indexMod(aCounter)):
                if action == Directions.STOP:
                    continue
                #print " "*aCounter,   "GO DEEPER! passing:", alpha_v, beta_v
                aspire_val, aspire_action = get_val(gState.generateSuccessor(indexMod(aCounter), action), alpha_v, beta_v, aCounter +  1)
                #print "pacman", "[aspire_val: ", aspire_val, "aspire_action: ", aspire_action, "alpha_v:", alpha_v, "beta_v:", beta_v, aspire_val > alpha_v, "]"
                if aspire_val > alpha_v:
                    #print " "*aCounter,   aspire_val ,"chosen over",alpha_v
                    alpha_v, alpha_action = aspire_val, [action]
                if beta_v <= alpha_v:
                    break

            return alpha_v, aspire_action + alpha_action
        else:
            for action in gState.getLegalActions(indexMod(aCounter)):
                if action == Directions.STOP:
                    continue
                #print "ghost loop: ", alpha_v, alpha_action, beta_v, beta_action
                aspire_val, aspire_action = get_val(gState.generateSuccessor(indexMod(aCounter), action), alpha_v, beta_v, aCounter +  1)
                #print "ghost", "[aspire_val: ", aspire_val, "aspire_action: ", aspire_action, "alpha_v:", alpha_v, "beta_v:", beta_v, aspire_val < beta_v, "]"
                if aspire_val < beta_v:
                    beta_v, beta_action = aspire_val, [action]
                if beta_v <= alpha_v:
                    break
            return beta_v, aspire_action + beta_action
    #print get_val(gameState, MINUS_INF, PLUS_INF, 0)
    return get_val(gameState, MINUS_INF, PLUS_INF, 0)[1][-1]

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

    numAgents = gameState.getNumAgents()
    indexMod = lambda aC: aC%numAgents
    isPacman = lambda i:indexMod(i) == 0

    #print "--------------NEW CALL-------------"
    def get_val(gState, aCounter):
        if gState.isWin() or gState.isLose():
            return gState.getScore()
        if aCounter == self.depth*numAgents:
            return self.evaluationFunction(gState)

        #print " "*aCounter, aCounter, "PACMAN" if isPacman(aCounter) else "GHOST"
        if isPacman(aCounter):
            b_val = max([get_val(gState.generateSuccessor(indexMod(aCounter), action), aCounter + 1) for action in gState.getLegalActions(indexMod(aCounter)) if action != Directions.STOP])
        else:
            v_list = [(get_val(gState.generateSuccessor(indexMod(aCounter), action), aCounter + 1)) for action in gState.getLegalActions(indexMod(aCounter)) if action != Directions.STOP]
            #print v_list
            b_val = sum(v_list)/len(v_list)
        #print " "*aCounter, aCounter, b_val
        return b_val
    return max([(get_val(gameState.generateSuccessor(0, action), 1), action) for action in gameState.getLegalActions(0) if action != Directions.STOP])[1]

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
    metric = util.manhattanDistance
    ghostDistances = [metric(newPos, hh.getPosition()) for hh in newGhostStates]
    if any([gh == 0 for gh in ghostDistances]):
        return -999

    score = currentGameState.getScore() # = successorGameState.getScore()

    foodCount = currentGameState.getNumFood()
    if(foodCount ==0):
        return 9999
    nearFoodDist = 100
    for i, item in enumerate(newFood):
        for j, foodItem in enumerate(item):
            nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
    ghostFun = lambda d : -20 + d**4 if d  < 3 else -1.0/d
    #nearFoodBonus = nearFoodBonusDict[nearFoodDist] if nearFoodDist in nearFoodBonusDict else (30 - 2*nearFoodDist)
    ghostK = sum([ghostFun(ghostDistances[i]) if newScaredTimes[i] < 1 else 0 for i in range(len(ghostDistances))])
    nearFoodBonus = 1.0/nearFoodDist
    foodRemPunish = -1.5
    peleteRemPunish = -8 if all((t == 0 for t in newScaredTimes)) else 0
    if all((t > 0 for t in newScaredTimes)):
        ghostK *= (-1)

    pelets = currentGameState.getCapsules()
    pelets.sort()
    #print pelets
    nearPeletDist = 100
    nearPeletDist = min(nearPeletDist, [metric(newPos, pelet) for pelet in pelets])
    nearPeletBonus = 1.0/nearPeletDist
    peleteRemaining = len(pelets)

    #print foodCount, nearFoodDist

    score = score + nearFoodBonus + 2*ghostK + 3*nearPeletBonus + foodRemPunish* foodCount + peleteRemaining *peleteRemPunish
    #print "score:", score, "nearFoodBonus:",nearFoodBonus, "ghostK:",ghostK, "peleteRemPunish:", peleteRemPunish, newScaredTimes
    return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
