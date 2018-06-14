import argparse

	 import torch
	 import torch.nn as nn
	 import torchvision.models as models
	 import torchvision.transforms as transforms
	 from torch.autograd import Variable
	 from utils.data_loader_fe import get_loader


	 def get_model(model_path=None):

		 vgg = models.vgg16(pretrained=True)
		 model = nn.Sequential(*(vgg.features[i] for i in range(29)))

		 return model


	 def get_transform():
		 normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		 transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
		 return transform


	 def extract_features(root, files, transform, batch_size, shuffle, num_workers, model, save_path):

		 dataloader = get_loader(root, files, transform, batch_size, shuffle, num_workers)
		 model = model.cuda()
		 model.eval()

		 features = []
		 imnames = []
		 n_iters = len(dataloader)
		 for i, (images, names) in enumerate(dataloader):
			 images = Variable(images).cuda()
			 feas = model(images)
			 imnames.extend(names)
			 name = names[0].split('.')[0]
			 feas = feas.squeeze(0).view(512,196).transpose(0,1).contiguous()
			 # import pdb; pdb.set_trace()
			 torch.save(feas.data.cpu(), save_path + '/' + name + '.pth')
			 if (i+1)%100 == 0:
				 print('iter [%d/%d] finsihed.'%(i, n_iters))

		 return torch.cat(features, 0), imnames


	 def main(args):

		 root = args.root
		 files = args.files
		 transform = get_transform()
		 batch_size = args.batch_size
		 shuffle = args.shuffle
		 num_workers = args.num_workers
		 save_path = args.save_path
		 model = get_model(args.model_path).cuda()
		 model.eval()
		 features, names = extract_features(root, files, transform, batch_size, shuffle, num_workers, model, save_path)



	 if __name__ == '__main__':
		 parser = argparse.ArgumentParser()
		 parser.add_argument('--root', type=str,
							 default='/mnt/exhdd/hdd/data/zini/COCO/train2014',
							 help='the directory that contains images.')
		 parser.add_argument('--files', type=str, default=None, help='file lists')
		 parser.add_argument('--batch_size', type=int, default=1, help='batch size')
		 parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle the dataset')
		 parser.add_argument('--num_workers', type=int, default=0, help='the number of threads for data loader')
		 parser.add_argument('--model_path', type=str, default='',
							 help='The path to the feature extraction model')
		 parser.add_argument('--save_path', type=str, default='/mnt/exhdd/hdd/data/zini/COCO/features2014', help='Where to save the files.')
		 args = parser.parse_args()
		 print(args)
		 main(args)
		 print('Done.')
