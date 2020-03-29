import torch.nn
from torch import autograd

x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**2 * x + b * x + c

grads = autograd.grad(y, [a, b, c])
print('after', grads[0], grads[1], grads[2])
