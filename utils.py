import random
from matplotlib import pyplot as plt
import torchvision

def split_integer(prototype, part):
    if not (isinstance(prototype, int) and isinstance(part, int)):
        raise ValueError('Input params should be integer.')
    if part == 0:
        return []
    if part < 0:
        raise ValueError('Split part should not be a minus.')
    if prototype <= part:
        raise ValueError('Split part should be more than split prototype.')
    board_set = set()
    while len(board_set) < part - 1:
        board_set.add(random.randrange(0, prototype+1))
    board_list = list(board_set)
    board_list.append(0)
    board_list.append(prototype)
    board_list.sort()
    return [board_list[i + 1] - board_list[i] for i in range(part)]



def show_image(img):
    img = torchvision.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()



import math
import torch

def preprocess_gradients(x):
    x = x.view(-1,1)
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)
    return torch.cat((x1, x2), 1)
