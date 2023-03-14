import torch.nn as nn

class NN_Deep(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, activationFunction, device):
		super(NN_Deep, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size).to(device)
		self.activation1 = activationFunction.to(device)
		self.linear2 = nn.Linear(hidden_size, hidden_size).to(device)
		self.activation2 = activationFunction.to(device)
		self.linear3 = nn.Linear(hidden_size, hidden_size).to(device)
		self.activation3 = activationFunction.to(device)
		self.linear4 = nn.Linear(hidden_size, hidden_size).to(device)
		self.activation4 = activationFunction.to(device)
		self.linear5 = nn.Linear(hidden_size, hidden_size).to(device)
		self.activation5 = activationFunction.to(device)
		self.linear6 = nn.Linear(hidden_size, hidden_size).to(device)
		self.activation6 = activationFunction.to(device)
		self.linear7 = nn.Linear(hidden_size, output_size).to(device)

	def forward(self, x):
		out = self.linear1(x)
		out = self.activation1(out)
		out = self.linear2(out)
		out = self.activation2(out)
		out = self.linear3(out)
		out = self.activation3(out)
		out = self.linear4(out)
		out = self.activation4(out)
		out = self.linear5(out)
		out = self.activation5(out)
		out = self.linear6(out)
		out = self.activation6(out)
		out = self.linear7(out)
		return abs(out)