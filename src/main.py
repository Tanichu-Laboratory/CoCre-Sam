import random
import string
from collections import defaultdict, Counter
import numpy as np
import os

scale = 1
num_sample = 100000

random.seed(7)

N = 6

max_steps = 2000
sampling_tail_ratio = 0.95
num_ite = 10

tau = 0.1
eta = 0.1
sigma = 0.3
D = tau*eta
num_agent = 2

max_word = 20 # max character length

plot_letters = ['$A$', '$B$', '$C$', '$D$', '$E$', '$F$', 
                '$G$', '$H$', '$I$', '$J$', '$K$', '$L$', 
                '$M$', '$N$', '$O$', '$P$', '$Q$', '$R$', 
                '$S$', '$T$', '$U$', '$V$', '$W$', '$X$', 
                '$Y$', '$Z$', '$EOS$']

bos_symbol = '^'

save_dir = "save"
if not os.path.isdir(save_dir):
   os.mkdir(save_dir)

def data_flower():
    all_flower = [
    'aster', 'aloe', 'azalea', 'amaryllis', 'broom', 'buttercup', 'begonia', 'bluebell', 'crocus', 'cosmos', 'camellia', 'clematis', 'daisy', 'dahlia', 'dogwood', 'delphinium','edelweiss', 'everlast', 'elm', 'epiphyllum', 'freesia', 'foxglove', 'fuchsia', 'fennel','geranium', 'gardenia', 'globe', 'gazania', 'hibiscus', 'heather', 'hyacinth', 'holly','iris', 'impatiens', 'ivy', 'indigo', 'jasmine', 'jacobinia', 'jonquil', 'jewelweed','kalmia', 'kerria', 'kalanchoe', 'knautia', 'lily', 'lotus', 'lavender', 'linaria','mallow', 'magnolia', 'mint', 'mallowleaf', 'nasturtium', 'narcissus', 'nicotiana', 'nemesia','orchid', 'oleander', 'oxeye', 'osmanthus', 'peony', 'poppy', 'petunia', 'primrose','quince', 'queen', 'quaker', 'quassia', 'rose', 'ranunculus', 'rue', 'rockrose','snapdragon', 'stock', 'sunflower', 'snowdrop', 'tulip', 'thyme', 'tansy', 'toadflax', 'umbrella', 'ursinia', 'utricularia', 'ulex', 'violet', 'veronica', 'vinca', 'valerian','wisteria', 'wallflower', 'waxplant', 'wintergreen', 'xeranthemum', 'xylosma', 'xenia', 'xyris', 'yucca', 'yellowbell', 'yarrow', 'yerba', 'zinnia', 'zephyr', 'zantedeschia', 'zigadenus'
    ]

    weights_color = [
    12, 3, 15, 17, 4, 13, 15, 10, 12, 13, 9, 14, 8, 18, 6, 11, 3, 2, 1, 17, 15, 14, 20, 2, 12, 4, 5, 20, 18, 10, 14, 3, 11, 12, 1, 9, 6, 11, 8, 13, 10, 9, 14, 12, 8, 9, 10, 13, 11, 7, 2, 3, 16, 15, 7, 12, 19, 10, 8, 5, 18, 17, 16, 13, 8, 6, 4, 3, 20, 18, 3, 9, 17, 8, 13, 2, 19, 4, 6, 12, 2, 10, 5, 6, 10, 9, 11, 8, 11, 13, 4, 3, 7, 1, 8, 5, 4, 12, 6, 3, 19, 9, 17, 6
    ]

    data1 = random.choices(all_flower, weights=weights_color, k=num_sample)

    max_weight = max(weights_color)
    weights_color = [max_weight - w + 1 for w in weights_color]

    data2 = random.choices(all_flower, weights=weights_color, k=num_sample)

    return data1, data2, all_flower


class CharNGramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)
        self.alphabet = list(string.ascii_lowercase) + ['$']  # a〜z + EOS

    def train(self, names):
        for name in names:
            name = '^' * (self.n-1) + name.lower() + '$'
            for i in range(len(name) - self.n + 1):
                context = name[i:i+self.n-1]
                next_char = name[i+self.n-1]
                self.model[context][next_char] += 1

    def get_probabilities(self, context, smoothing=1e-9):
        next_chars = self.model.get(context, {})
        smoothed_counts = {
            char: next_chars.get(char, 0) + smoothing for char in self.alphabet
        }
        total = sum(smoothed_counts.values())
        return {char: count / total for char, count in smoothed_counts.items()}


# compute gradient
# energy = 0.5 * np.log(1 + r^2 / sigma^2)
def compute_score(gxy, xy):
  r_vec = xy - gxy
  r2 = np.dot(r_vec, r_vec)
  grad = (2 * r_vec) / (r2 + sigma**2)
  return grad

class Agent_two:
    def __init__(self, model1, model2, eta, D):
      self.model1 = model1
      self.model2 = model2
      self.eta = eta
      self.D = D

    def gradient_energy(self, position, current_context_ngram):
        e1 = np.zeros_like(position)
        e2 = np.zeros_like(position)

        probabilities1 = self.model1.get_probabilities("".join(current_context_ngram))
        probabilities2 = self.model2.get_probabilities("".join(current_context_ngram))

        scores = []
        probs1 = []
        probs2 = []
        for symbol in symbols:
            scores = compute_score(goal_locations[symbol], position)
            probs1 = probabilities1[symbol]
            probs2 = probabilities2[symbol]
        
            e1 += probs1 * scores + probs2 * scores

        return e1, probabilities1, probabilities2

    def get_action(self, position, current_context_ngram):
        e1, probs1, probs2 = self.gradient_energy(position, current_context_ngram)
        noise1 = np.sqrt(num_agent * 2 * self.D) * np.random.randn(2)
        
        action1 = - self.eta * e1 + noise1
        return action1, probs1, probs2

    # update position
    def update_position_two(self,current_position, action1):
      next_position = current_position + action1
      next_position = np.clip(next_position, [-1*scale,-1*scale], grid_size)
      return next_position

# decide character
def find_closest_symbol_vote(positions):
  closest_counts = []
  for pos in positions:
    min_distance = float('inf')
    closest_symbol = None
    for symbol, goal in goal_locations.items():
        distance = np.linalg.norm(pos - goal)
        if distance < min_distance:
            min_distance = distance
            closest_symbol = symbol
    closest_counts.append(closest_symbol)
  counter=Counter(closest_counts)
  most_common_symbol = counter.most_common(1)[0][0]
  return most_common_symbol, min_distance


def main(num_trial = 100):
  words_ab = []
  for iii in range(num_trial):
    position = initial_position.copy()
    context = [bos_symbol] * (model1.n - 1)

    word = []
    trajectories = []
    probs_all_a = []
    probs_all_b = []
    stop = False

    multi_agent = Agent_two(model1, model2, eta, D)
    image_id = 0
    while stop==False:
      trajectory = [position.copy()]

      for step in range(max_steps):
          a1, p1, p2 = multi_agent.get_action(position, context[-(model1.n - 1):])
          position = multi_agent.update_position_two(position, a1)
          trajectory.append(position.copy())

      closest_symbol, min_distance = find_closest_symbol_vote(trajectory[int(max_steps*sampling_tail_ratio):])

      trajectories.append(np.array(trajectory))
      image_id += 1

      context.append(closest_symbol)
      context.pop(0)
      word.append(closest_symbol)

      if closest_symbol=="$":
        stop = True
      elif len(trajectories) > max_word:
        stop = True
      else:
        pass

    words_ab.append(word)
    np.save(save_dir+"/%03d.npy"%iii, np.array(trajectories))

  dic_ab = {}
  for ws in words_ab:
    out = ''.join(ws).replace('$', '')
    if out in dic_ab.keys():
      dic_ab[out] += 1
    else:
      dic_ab[out] = 1

  print(sorted(dic_ab.items(), key=lambda x: x[1], reverse=True))
  return dic_ab, multi_agent

if __name__=="__main__":
  # Env
  goal_locations = {}
  initial_position = np.array([3.0*scale, 0.0*scale])
  symbols = list(string.ascii_lowercase) + ['$']
  grid_size = (7*scale, 4*scale)
  difference = 0
  for i, symbol in enumerate(symbols):
      i+=difference
      row = (i // (grid_size[0]//scale))*scale
      col = (i % (grid_size[0]//scale))*scale
      if np.array_equal(initial_position, np.array([col, row])):
        difference =1
        i+=difference
        row = (i // (grid_size[0]//scale))*scale
        col = (i % (grid_size[0]//scale))*scale
      goal_locations[symbol] = np.array([float(col), float(row)])

  # data
  data1, data2, all_flower = data_flower()
  
  # Train language model
  model1 = CharNGramModel(N)
  model1.train(data1)
  model2 = CharNGramModel(N)
  model2.train(data2)

  # main  
  dic_ab, multi_agent = main(num_ite) 
