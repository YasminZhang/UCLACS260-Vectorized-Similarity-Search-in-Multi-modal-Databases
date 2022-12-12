import torch
import clip
from PIL import Image
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import pandas as pd
import numpy as np
from tqdm import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


path2data = './images'
path2json = './annotations/captions_val2014.json'


coco = dset.CocoCaptions(root = path2data,
                        annFile = path2json)



sample_10 = random.sample(range(len(coco)), 10)
images = []
texts = []
embeddings = []
with torch.no_grad():
    for j,i in enumerate(sample_10):
        image, text = coco[i]
        image.save(f'sample_10/{j}.png')
        texts.append(text)
        
        text_ = clip.tokenize(text).to(device)
        image_ = preprocess(image).unsqueeze(0).to(device)
        #print(image.shape) 
        image_feature = model.encode_image(image_)
        text_feature = model.encode_text(text_)
    
        #print(f'image_feature:{image_feature.shape}')
        #print(f'text_feature:{text_feature.shape}')
    
        image_text = torch.cat((image_feature,text_feature),0).reshape(1,-1)
        if image_text.shape[1] != 3072:
            image_text = image_text[:, :3072]
        embeddings.append(image_text.reshape(1,-1)) 
        
        torch.cuda.empty_cache()
        

dat = pd.DataFrame(texts)
dat.to_csv('sample_10/texts.csv')
embeddings = torch.cat(embeddings, 0)
print('final_embedding:',embeddings.shape)
np.save('sample_10/sample_embedding', embeddings.detach().to('cpu').numpy())

