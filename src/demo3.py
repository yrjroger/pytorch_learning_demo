import torch
import torch.nn.functional as F
import torch.autograd

x = torch.ones(1)
print('x', x)
w = torch.full([1], 2, requires_grad=True)
mse = F.mse_loss(torch.ones(1), x*w)
print('mse', mse)

# print('grad', torch.autograd.grad(mse, [w]))
print('grad1', w.grad)
mse.backward()
print('grad2', w.grad)
