    import argparse
    import pickle

    import numpy as np

    import torch.optim as optim
    import torchvision.models as models
    from models import *
    from torch.autograd import Variable
    from torch.nn.utils.rnn import pack_padded_sequence
    from torchvision import transforms
    from utils.build_vocab import Vocabulary
    from utils.data_loader import get_loader

    cuda_check = False
    if torch.cuda.is_available():
    	cuda_check = True


    def train(dataloader, model, optimizer, criterion, epoch, total_epoch):
    	total_step = len(dataloader)

    	for i, (inputt, targets, lengths) in enumerate(dataloader):

    		optimizer.zero_grad()

    		inputt = Variable(inputt)
    		targets = Variable(targets)
    		if cuda_check:
    			features = inputt.cuda()
    			targets = targets.cuda()

    		predicts = model(features, targets[:, :-1], [l - 1 for l in lengths])
    		predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
    		targets = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first=True)[0]

    		loss = criterion(predicts, targets)
    		loss.backward()
    		optimizer.step()

    		if (i+1)%10 == 0:
    			print ('Epoch [%d/%d]: [%d/%d], loss: %5.4f, perplexity: %5.4f.'%(epoch, total_epoch,i,
    																			 total_step,loss.data[0],
    																			 np.exp(loss.data[0])))

    def main(args):
    	# dataset setting
    	image_root = args.image_root
    	ann_path = args.ann_path
    	vocab_path = args.vocab_path
    	batch_size = args.batch_size
    	shuffle = args.shuffle
    	num_workers = args.num_workers

    	with open(vocab_path, 'rb') as f:
    		vocab = pickle.load(f)

    	dataloader = get_loader(image_root, ann_path, vocab, batch_size,
    							 shuffle=True, num_workers=args.num_workers)

    	# model setting
    	vis_dim = args.vis_dim
    	vis_num = args.vis_num
    	embed_dim = args.embed_dim
    	hidden_dim = args.hidden_dim
    	vocab_size =args.vocab_size
    	num_layers = args.num_layers
    	dropout_ratio = args.dropout_ratio

    	model = Decoder(vis_dim=vis_dim,
    					vis_num=vis_num,
    					embed_dim=embed_dim,
    					hidden_dim=hidden_dim,
    					vocab_size=vocab_size,
    					num_layers=num_layers,
    					dropout_ratio=dropout_ratio)

    	# optimizer setting
    	lr = args.lr
    	num_epochs = args.num_epochs
    	optimizer = optim.Adam(model.parameters(), lr=lr)

    	# criterion
    	criterion = nn.CrossEntropyLoss()
    	if cuda_check:
    		model.cuda()
    		criterion.cuda()

    	model.train()

    	print('Number of epochs:', num_epochs)
    	for epoch in range(num_epochs):
    		train(dataloader=dataloader, model=model, optimizer=optimizer, criterion=criterion,
    			  epoch=epoch, total_epoch=num_epochs)
    		torch.save(model, './checkpoints/model_%d.pth'%(epoch))



    if __name__ == '__main__':
    	parser = argparse.ArgumentParser()
    	# data loader
    	parser.add_argument('--image_root', type=str,
    						default='/mnt/exhdd/hdd/data/zini/COCO/features2014')
    	parser.add_argument('--ann_path', type=str,
    						default='/mnt/exhdd/hdd/data/zini/COCO/annotations/captions_train2014.json')
    	parser.add_argument('--vocab_path', type=str,
    						default='./data/vocab.pkl')
    	parser.add_argument('--batch_size', type=int, default=128)
    	parser.add_argument('--shuffle', type=bool, default=False)
    	parser.add_argument('--num_workers', type=int, default=4)

    	# model setting
    	parser.add_argument('--vis_dim', type=int, default=512)
    	parser.add_argument('--vis_num', type=int, default=196)
    	parser.add_argument('--embed_dim', type=int, default=512)
    	parser.add_argument('--hidden_dim', type=int, default=512)
    	parser.add_argument('--vocab_size', type=int, default=10000)
    	parser.add_argument('--num_layers', type=int, default=1)
    	parser.add_argument('--dropout_ratio', type=float, default=0.5)

    	# optimizer setting
    	parser.add_argument('--lr', type=float, default=0.0001)
    	parser.add_argument('--num_epochs', type=int, default=120)

    	args = parser.parse_args()
    	print (args)
    	main(args)
