import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):

	def __init__(self, embeddings, n_features=36,
		hidden_size=200, n_classes=3, dropout_prob=0.5):

		super(ParserModel, self).__init__()
		self.n_features = n_features
		self.n_classes = n_classes
		self.dropout_prob = dropout_prob
		self.embed_size = embeddings.shape[1]
		self.hidden_size = hidden_size
		self.embeddings = nn.Parameter(torch.tensor(embeddings))

		# 1) input -> hidden 
		self.embed_to_hidden_weight = nn.Parameter(
			torch.empty(self.hidden_size, self.n_features * self.embed_size)
		)
		nn.init.xavier_uniform_(self.embed_to_hidden_weight)
		self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
		nn.init.uniform_(self.embed_to_hidden_bias, -1e-4, 1e-4)

		# 2) dropout
		self.dropout = nn.Dropout(p=self.dropout_prob)

		# 3) hidden -> output
		self.hidden_to_logits_weight = nn.Parameter(
			torch.empty(self.n_classes, self.hidden_size)
		)
		nn.init.xavier_uniform_(self.hidden_to_logits_weight)
		self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
		nn.init.uniform_(self.hidden_to_logits_bias, -1e-4, 1e-4)

	def embedding_lookup(self, w):

		x = torch.index_select(self.embeddings, 0, w.reshape(-1)).view(w.shape[0], -1)

		return x


	def forward(self, w):

		# input_size = 1080
		# hidden_size = 200

		x = self.embedding_lookup(w) # (4, 1080)
		h = F.relu(x @ self.embed_to_hidden_weight.T + self.embed_to_hidden_bias)
		# W.T: (1080, 200) h: (4, 200) 
		h = self.dropout(h) # (4, 200)
		logits = h @ self.hidden_to_logits_weight.T + self.hidden_to_logits_bias # (4, 3)

		return logits


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
	parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
	parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
	args = parser.parse_args()

	embeddings = np.zeros((100, 30), dtype=np.float32)
	model = ParserModel(embeddings)

	def check_embedding():
		inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
		selected = model.embedding_lookup(inds)
		assert np.all(selected.data.numpy() == 0), "The result of embedding lookup: " \
									  + repr(selected) + " contains non-zero elements."

	def check_forward():
		inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
		out = model(inputs)
		expected_out_shape = (4, 3)
		assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
												" which doesn't match expected " + repr(expected_out_shape)

	if args.embedding:
		check_embedding()
		print("Embedding_lookup sanity check passes!")

	if args.forward:
		check_forward()
		print("Forward sanity check passes!")
