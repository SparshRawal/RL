import pickle

SWEEP_RESULTS_FILE = '/Users/tanishsurana/rl/uoe-rl2023-coursework/rl2023/exercise3/DQN--CartPole-v1--epsilon_start:1.0_epsilon_decay:1.0--9.pt'

results = pickle.load(open(SWEEP_RESULTS_FILE, 'rb'))