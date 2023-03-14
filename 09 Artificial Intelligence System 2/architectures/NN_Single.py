import torch.nn as nn

class NN_Single(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, activationFunction, device):
		super(NN_Single, self).__init__()
		self.linear = nn.Linear(input_size, output_size).to(device)

	def forward(self, x):
		out = self.linear(x)
		return abs(out)