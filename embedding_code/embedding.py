import torch
import clip
from PIL import Image
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



path2data = './images'
path2json = './annotations/captions_val2014.json'


coco = dset.CocoCaptions(root = path2data,
                        annFile = path2json)
                        #transform=transforms.PILToTensor())


print('Number of samples: ', len(coco))


from tqdm import tqdm
embeddings = []
with torch.no_grad():
    for i in tqdm(range(len(coco))):
        image, text = coco[i]
        text = clip.tokenize(text).to(device)
        image = preprocess(image).unsqueeze(0).to(device)
        #print(image.shape) 
        image_feature = model.encode_image(image)
        text_feature = model.encode_text(text)
    
        #print(f'image_feature:{image_feature.shape}')
        #print(f'text_feature:{text_feature.shape}')
    
        image_text = torch.cat((image_feature,text_feature),0).reshape(1,-1)
        if image_text.shape[1] != 3072: # Data cleaning
            image_text = image_text[:, :3072]
        embeddings.append(image_text.reshape(1,-1)) 
        
        torch.cuda.empty_cache()
        


embeddings = torch.cat(embeddings, 0)
print('final_embedding:',embeddings.shape)
import numpy as np
np.save('2014_val_embeddings', embeddings.detach().to('cpu').numpy())

