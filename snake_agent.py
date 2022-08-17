import numpy as np
import helper
import random
import math

# Alex Thwin
# athwin
# CSE 140

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        #print("IN helper_func")
        snakeX, snakeY, snakeBod, foodX, foodY = state

        # Since our coordinates are in multiples of 40, we can use math.floor
        # after dividing by 40 to get single digit coords so they are easier
        # to work with
        snakeX = math.floor(snakeX/40)
        snakeY = math.floor(snakeY/40)
        snakeBodSing = []
        for i, j in snakeBod:
            snakeBodSing.append((math.floor(i/40), math.floor(j/40)))
        foodX = math.floor(foodX/40)
        foodY = math.floor(foodY/40)

        # adjWall = adjoining wall coordinates [x,y]
        # For x: 0 = no adjoining wall on x axis, 1 = wall left of snake, 2 = wall right of snake
        # For y: 0 = no adjoining wall on y axis, 1 = wall top of snake, 2 = wall bot of snake
        adjWall = [0, 0]
        
        # Check the X coord of the snake and update adjWall
        if snakeX == 1:
            adjWall[0] = 1
        elif snakeX == 12:
            adjWall[0] = 2
        else:
            adjWall[0] = 0

        # Check the Y coord of the snake and update adjWall
        if snakeY == 1:
            adjWall[1] = 1
        elif snakeY == 12:
            adjWall[1] = 2
        else:
            adjWall[1] = 0

        # body = [top, bot, left, right]
        # 1 = adjacent to position in [top,bot,left,right]
        # 0 = not adjacent
        body = [0,0,0,0]

        if (snakeX, snakeY-1) in snakeBodSing:
            adj = 1
        else:
            adj = 0
        body.append(adj)

        if (snakeX, snakeY+1) in snakeBodSing:
            adj = 1
        else:
            adj = 0
        body.append(adj)

        if (snakeX-1, snakeY) in snakeBodSing:
            adj = 1
        else:
            adj = 0
        body.append(adj)

        if (snakeX+1, snakeY) in snakeBodSing:
            adj = 1
        else:
            adj = 0
        body.append(adj)

        # foodDir = food coordinates [x,y]
        # For x: 0 = same x axis, 1 = left of snake, 2 = right of snake
        # For y: 0 = same y axis, 1 = top of snake, 2 = bot of snake
        foodDir = [0, 0]

        # Check the X coord of the snake and update foodDir
        if (foodX - snakeX) > 0:
            foodDir[0] = 2
        elif (foodX - snakeX) < 0:
            foodDir[0] = 1
        else:
            foodDir[0] = 0

        # Check the Y coord of the snake and update foodDir
        if (foodY - snakeY) > 0:
            foodDir[1] = 2
        elif (foodY - snakeY) < 0:
            foodDir[1] = 1
        else:
            foodDir[1] = 0

        return (adjWall[0], adjWall[1], foodDir[0], foodDir[1], body[0], body[1], body[2], body[3])


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #
    # ========================================================
    # THIS FUNCTION WAS CHANGED
    # ========================================================

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.

    # References: https://towardsdatascience.com/teaching-a-computer-how-to-play-snake-with-q-learning-93d0a316ddc0
    def agent_action(self, state, points, dead):
        #print("IN AGENT_ACTION")

        # Get current state
        currState = self.helper_func(state)

        # Make a deep copy of the state
        copyState = state.copy()
        copyState[2] = state[2].copy()

        # Goes to negative reward
        if dead:
            lastMove = self.helper_func(self.s)
            self.Q[lastMove[0]][lastMove[1]][lastMove[2]][lastMove[3]][lastMove[4]][lastMove[5]][lastMove[6]][lastMove[7]][self.a] = self.updateQVals(self.s, self.a, state, dead, points)
            self.reset()
            return

        if self._train and (self.s != None) and (self.a != None):
            lastMove = self.helper_func(self.s)
            newQ = self.updateQVals(self.s, self.a, state, dead, points)
            self.Q[lastMove[0]][lastMove[1]][lastMove[2]][lastMove[3]][lastMove[4]][lastMove[5]][lastMove[6]][lastMove[7]][self.a] = newQ

        util = [0, 0, 0, 0]
        for i in range(len(util)):
            valueQ = self.Q[currState[0]][currState[1]][currState[2]][currState[3]][currState[4]][currState[5]][currState[6]][currState[7]][i]
            valueN = self.N[currState[0]][currState[1]][currState[2]][currState[3]][currState[4]][currState[5]][currState[6]][currState[7]][i]

            if self.Ne < valueN:
                util[i] = valueQ
            else:
                util[i] = 1

        action = np.argmax(util)
        maxQVal = max(util)

        for i in range(len(util)-1, -1, -1):
            if util[i] == maxQVal:
                action = i
                break

        self.N[currState[0]][currState[1]][currState[2]][currState[3]][currState[4]][currState[5]][currState[6]][currState[7]][action] += 1
        self.s = copyState
        self.s[2] = copyState[2].copy()
        self.a = action
        self.points = points

        #UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        return action

    # Added this function based on reference
    def updateQVals(self, lastMove, lastAct, state, dead, points):
        last_move = self.helper_func(lastMove)
        reward = self.compute_reward(points, dead)

        alpha = self.LPC / (self.LPC + self.N[last_move[0]][last_move[1]][last_move[2]][last_move[3]]
                          [last_move[4]][last_move[5]][last_move[6]][last_move[7]][lastAct])
        curr = self.helper_func(state)
        top = self.Q[curr[0]][curr[1]][curr[2]][curr[3]][curr[4]][curr[5]][curr[6]][curr[7]][0]
        bottom = self.Q[curr[0]][curr[1]][curr[2]][curr[3]][curr[4]][curr[5]][curr[6]][curr[7]][1]
        left = self.Q[curr[0]][curr[1]][curr[2]][curr[3]][curr[4]][curr[5]][curr[6]][curr[7]][2]
        right = self.Q[curr[0]][curr[1]][curr[2]][curr[3]][curr[4]][curr[5]][curr[6]][curr[7]][3]
        maxAlpha = max(top, bottom, left, right)
        valueQ = self.Q[last_move[0]][last_move[1]][last_move[2]][last_move[3]][last_move[4]][last_move[5]][last_move[6]][last_move[7]][lastAct]
        
        return valueQ + alpha * (reward + self.gamma * maxAlpha - valueQ)