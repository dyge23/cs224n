"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter



@dataclass
class ModelConfig:
	d_model: int
	n_heads: int
	n_layers: int
	context_length: int
	vocab_size: int


class CausalAttention(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()

		# Using attention dim from attention is all you need
		assert config.d_model % config.n_heads == 0
		self.d_attention = int(config.d_model / config.n_heads)

		# using n_heads
		self.n_heads = config.n_heads

		self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
		self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
		self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

		self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

		# Causal mask
		self.register_buffer(
			"causal_mask",
			torch.tril(torch.ones(config.context_length, config.context_length)).view(
				1, 1, config.context_length, config.context_length
			),
			persistent=False,
		)

	def forward(
		self, x: Float[Tensor, "batch seq_len d_model"],
		use_cache: bool = False,
		past_kv: tuple = None # (batch, past_len, d_model)
	) -> Float[Tensor, "batch seq_len d_model"]:
		"""
		Causal self-attention with an optional K/V cache.

		When use_cache=True, the K and V of all previously seen tokens are
		concatenated onto the K/V of the current tokens, so autoregressive
		generation only recomputes attention for the newest token instead of
		re-running the whole sequence at every step.

		Returns:
			(batch, seq_len, d_model)
			if use_cache: additionally returns the updated (k, v) of shape
			(batch, past_len + seq_len, d_model)
		"""

		batch_size, seq_len, _ = x.shape

		# 1) Project the input to queries, keys and values
		q = self.W_q(x)
		k = self.W_k(x)
		v = self.W_v(x)

		# 2) Append the cached kv if this is a later generation step (KV Cache)
		if use_cache and past_kv is not None:
			past_k, past_v = past_kv
			k = torch.cat([past_k, k], dim=1) # (batch, past_len, d_model)
			v = torch.cat([past_v, v], dim=1)

		# past_len = total_len - seq_len
		past_len = k.shape[1] - seq_len
		k_full, v_full = k, v

		# 3）Split into heads
		q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
		k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
		v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

		# 4) Scaled dot-product attention scores
		scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_attention)

		# 5) causal_mask
		mask = self.causal_mask[:, :, past_len : past_len + seq_len, : k.shape[-2]]
		scores = scores.masked_fill(mask == 0, float("-inf"))

		# 6) final dot
		probs = softmax(scores, dim=-1)
		context = probs @ v

		# 
		context = rearrange(context, "b h s d -> b s (h d)")
		out = self.W_o(context)

		if use_cache:
			return out, (k_full, v_full)

		return out


class GELU(nn.Module):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
	Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
	"""

	def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
		return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip


class MLP(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()

		self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
		self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
		self.gelu = GELU()

	def forward(
		self, x: Float[Tensor, "batch seq_len d_model"]
	) -> Float[Tensor, "batch seq_len d_model"]:
	
		return self.fc2(self.gelu(self.fc1(x)))
		


class DecoderBlock(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()

		self.mlp = MLP(config)
		self.attention = CausalAttention(config)
		self.pre_layer_norm = nn.LayerNorm(config.d_model)
		self.post_layer_norm = nn.LayerNorm(config.d_model)

	def forward(
		self, x: Float[Tensor, "batch seq_len d_model"],
		use_cache: bool = False,
		past_kv: tuple = None
	) -> Float[Tensor, "batch seq_len d_model"]:

		residual = x
		x = self.pre_layer_norm(x)

		if use_cache:
			attn_out, new_kv = self.attention(x, use_cache=True, past_kv=past_kv)
			x = residual + attn_out
		else:
			x = residual + self.attention(x)

		x = x + self.mlp(self.post_layer_norm(x))

		if use_cache:
			return x, new_kv
		return x

class Transformer(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()

		self.config = config
		self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
		self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
		self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
		self.final_layer_norm = nn.LayerNorm(config.d_model)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

		self._init_weights()

	def _init_weights(self):

		for module in self.modules():
			if isinstance(module, nn.Linear):
				torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
				if module.bias is not None:
					torch.nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			elif isinstance(module, nn.LayerNorm):
				torch.nn.init.zeros_(module.bias)
				torch.nn.init.ones_(module.weight)

		# init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
		for pn, p in self.named_parameters():
			if pn.endswith("c_proj.weight"):
				torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

	def forward(
		self, x: Int[Tensor, "batch_size seq_len"],
		use_cache: bool = False,
		past_key_values: list = None # list of (k, v) tuples, one per decoder block
	) -> Float[Tensor, "batch seq_len vocab_size"]:

		batch_size, seq_len = x.shape

		past_len = 0
		if use_cache and past_key_values is not None:
			# length of the cached k in layer 0
			past_len = past_key_values[0][0].shape[1]

		# position embedding
		# unsequeeze: (1, seq_len)
		positions = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

		# token embedding + position embedding
		x = self.embeddings(x) + self.position_embeddings(positions)

		# when inferring, use cache
		if use_cache:
			# collect new token and add to past_kv
			new_past_key_values = []
			# iterate over all the layers of the neural network
			for i, block in enumerate(self.backbone):
				past_kv = past_key_values[i] if past_key_values is not None else None
				# output: out_x and new_kv
				x, kv = block(x, use_cache=True, past_kv=past_kv)
				# append new_kv
				new_past_key_values.append(kv)

			# lm_head: pos_vec -> vec_score
 			# output logits and final_new_kv
			x = self.final_layer_norm(x)
			return self.lm_head(x), new_past_key_values

		for block in self.backbone:
			x = block(x)

		x = self.final_layer_norm(x)
		return self.lm_head(x)
	
	@torch.no_grad()
	def generate(
		self,
		x: Int[Tensor, "batch_size seq_len"],
		num_new_tokens: int,
	) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

		x = x[:, -self.config.context_length :]
		past_key_values = None

		for _ in range(num_new_tokens):
			if past_key_values is None:
				# cache not found, use whole prompt -> new token
				logits, past_key_values = self(x, use_cache=True)
			else:
				# cache hit: use only one current token + kv -> new token
				logits, past_key_values = self(
					x[:, -1:], use_cache=True,
					past_key_values=past_key_values
				)

			# determine next new token
			next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
			x = torch.cat([x, next_token], dim=1)

		return x

	def get_loss_on_batch(
		self,
		input_ids: Int[Tensor, "batch_size seq_len"],

	) -> Float[Tensor, ""]:

		# (batch, seq_len, vocab_size)
		logits = self(input_ids)

		# Next-token prediction: use the logits at position i to predict token i+1
		shift_logits = logits[:, :-1, :].contiguous() # log1, log2, log3
		shift_labels = input_ids[:, 1:].contiguous() # token2, token3, token4

		# compute cross_entropy
		return F.cross_entropy(
			shift_logits.reshape(-1, shift_logits.size(-1)),
			shift_labels.reshape(-1),
		)

	@classmethod
	def from_pretrained(cls):
		"""
		We simply always load up the GPT-2 model
		"""

		# Config for GPT-2
		config = ModelConfig(
			d_model=768,
			n_heads=12,
			n_layers=12,
			context_length=1024,
			vocab_size=50257,
		)

		model = cls(config)

		# Load weights from HuggingFace
		model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
		converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

		model.load_state_dict(converted_state_dict)

		return model


if __name__ == "__main__":
	# Uncomment this if you are not logged in
	# huggingface_hub.login()

	model = Transformer.from_pretrained()
