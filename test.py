import numpy as np
import random
import time
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState

def mcts_select(state, iters=100):
  #  state = np.array([[1, 0, -1], [-1, 1, 1], [0, 0, -1]])
  initial_board_state = TicTacToeGameState(state = state, next_to_move=-1)
  #  print(initial_board_state.board)
  
  root = TwoPlayersGameMonteCarloTreeSearchNode(state = initial_board_state)
  mcts = MonteCarloTreeSearch(root)
  best_node = mcts.best_action(iters)

  return best_node.state.board, best_node.state.game_result

def random_move(state):

  board_state = TicTacToeGameState(state = state, next_to_move=1)
  legal_actions = board_state.get_legal_actions()
  random_act = random.choice(legal_actions)
  next_board_state = board_state.move(random_act)
  #  print(next_board_state.board)

  return next_board_state.board, next_board_state.game_result




test_len = 1000

#  for mcts_iters in [1, 10, 100, 500, 1000, 2000, 5000, 10000]:
#  for mcts_iters in [1, 10, 20, 50, 100, 200, 500]:
for mcts_iters in [200]:
  mcts_wins = 0
  # for _ in range(test_len):
  # 
  #   state = np.zeros((3,3))
  #   board_state = TicTacToeGameState(state = state, next_to_move=1)
  # 
  #   while True:
  #     #  print(state)
  #     mcts_moved_state, winner = mcts_select(state, mcts_iters)
  #     if winner is not None:
  #       if winner == -1:
  #         mcts_wins += 1
  #       break
  #     else:
  #       next_state, winner = random_move(mcts_moved_state)
  #       if winner is not None:
  #         break
  #       else:
  #         state = next_state

  # print('iters ', mcts_iters, ' mcts win ', float(mcts_wins/test_len))

  start = time.time()
  for _ in range(100):
    state = np.zeros((3,3))
    mcts_select(state, mcts_iters)
  end = time.time()
  
  print('iters ', mcts_iters, ' run time', end-start)



