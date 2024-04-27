import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
data1 = [[2,3,6], [4,5,7]]
x_data1 = torch.tensor(data1)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.zeros_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")



x_rand = torch.rand_like(x_data1, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

shape = (3,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(rand_tensor[0][0], "\n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
# print("\n\n")

tensor = torch.rand(3,3)
print(tensor)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
# print('First row: ',tensor[0])
# print('First column: ', tensor[:, 0])
# print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
print("\n\n")

t1 = torch.cat([tensor, tensor, tensor], dim=0) #if dim=0 moltiplica le righe, else le colonne
print(t1)
print("\n\n")
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
agg = tensor.sum()
agg_item = agg.item()
