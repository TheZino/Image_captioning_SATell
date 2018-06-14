import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from model.models import Decoder
from utils.build_vocab import Vocabulary
from utils.helper import attention_visualization


def get_encoder():
	vgg = models.vgg16(pretrained=True)
	model = nn.Sequential(*(vgg.features[i] for i in range(29)))
	return model

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
	image = Image.open(image_path)
	image = image.resize([224, 224], Image.LANCZOS)

	if transform is not None:
		image = transform(image).unsqueeze(0)

	return image

def main(args):
	# Image preprocessing
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
							 (0.229, 0.224, 0.225))])

	# Load vocabulary wrapper
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	# Build Models
	encoder = get_encoder()
	encoder.eval()
	decoder = Decoder(vis_dim=args.vis_dim,
					vis_num=args.vis_num,
					embed_dim=args.embed_size,
					hidden_dim=args.hidden_size,
					vocab_size=len(vocab),
					num_layers=args.num_layers,
					dropout_ratio=args.dropout_ratio)


	# Load the trained model parameters
	decoder = torch.load(args.decoder_path)

	# Prepare Image
	image = load_image(args.image, transform)
	image_tensor = to_var(image, volatile=True)

	# If use gpu
	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()

	# Generate caption from image
	feature = encoder(image_tensor)
	# features b_s x 512 x 14 x 14
	feature = feature.transpose(1,2).transpose(2,3).contiguous()
	# features b_s x 14 x 14 x 512
	feature = feature.view(feature.shape[0],feature.shape[1]*feature.shape[2], feature.shape[3])
	# features b_s x 196 x 512
	sampled_ids, alphas = decoder.sample(feature)
	sampled_ids = sampled_ids.cpu().data.numpy()

	# Decode word_ids to words
	sampled_caption = []
	for word_id in sampled_ids:
		word = vocab.idx2word[word_id]
		sampled_caption.append(word)
		if word == '<end>':
			break
	sentence = ' '.join(sampled_caption)

	# Print out image and generated caption.
	print (sentence)

	attention_visualization(image, sentence, alphas)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, required=True,
						help='input image for generating caption')
	parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-3000.pkl',
						help='path for trained decoder')
	parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
						help='path for vocabulary wrapper')

	# Model parameters (should be same as paramters in train.py)
	parser.add_argument('--embed_size', type=int , default=256,
						help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512,
						help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=1 ,
						help='number of layers in lstm')
	parser.add_argument('--dropout_ratio', type=float, default=0.5)
	parser.add_argument('--vis_dim', type=int, default=512)
	parser.add_argument('--vis_num', type=int, default=196)
	parser.add_argument('--embed_dim', type=int, default=512)

	args = parser.parse_args()
	main(args)
