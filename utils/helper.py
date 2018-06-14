
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import skimage.transform
    from PIL import Image

    __all__=['decode_captions',
    		'attention_visualization']

    def decode_captions(captions, idx_to_word):
    	N, D = captions.shape
    	decoded = []
    	for idx in xrange(N):
    		words = []
    		for wid in xrange(D):
    			word = idx_to_word[captions[idx, wid]]
    			if word == '<end>' or word == '<start>' or word == '<unk>':
    				words.append('.')
    			else:
    				words.append(word)
    		decoded.append(words)
    	return decoded

    def attention_visualization(image, caption, alphas):

    	image = image.squeeze(0)
    	im_norm = (image - image.min()) / (image.max()-image.min())
    	arr = im_norm.transpose(0,1).transpose(1,2).numpy()

    	plt.subplot(451)
    	plt.imshow((arr*255).astype(np.uint8))

    	plt.axis('off')

    	words = caption.split()
    	# import pdb; pdb.set_trace()
    	for t in range(len(words)):
    		if t > 18:
    			break
    		if t == 0:
    			plt.subplot(4, 5, t+2)
    			plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
    			plt.imshow((arr*255).astype(np.uint8))
    			alp_curr = np.zeros((14,14))
    			alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
    			plt.imshow(alp_img, alpha=0.85)
    			plt.axis('off')
    		else:
    			plt.subplot(4, 5, t+2)
    			plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
    			plt.imshow((arr*255).astype(np.uint8))
    			alp_curr = alphas[t].view(14, 14)
    			alp_img = skimage.transform.pyramid_expand(alp_curr.cpu().data.numpy(), upscale=16, sigma=20)
    			plt.imshow(alp_img, alpha=0.85)
    			plt.axis('off')
    	plt.show()
