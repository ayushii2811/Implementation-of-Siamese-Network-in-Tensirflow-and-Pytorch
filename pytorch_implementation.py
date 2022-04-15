!pip install segmentation-models-pytorch
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install --upgrade opencv-contrib-python
!git clone https://github.com/parth1620/Person-Re-Id-Dataset


import sys
sys.path.append('/content/Person-Re-Id-Dataset')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from skimage import io
from sklearn.model_selection import train_test_split

from tqdm import tqdm

DATA_DIR = '/content/Person-Re-Id-Dataset/train/'
CSV_FILE = '/content/Person-Re-Id-Dataset/train.csv'

batch_size = 32
lr = 0.001
epochs = 15

device = 'cuda'     #since we are on gpu device

df = pd.read_csv(CSV_FILE)
#df.head()

#the image path is not complete, it's the image file name
#so we will add the data directory name to it
row = df.iloc[4]

anc = io.imread(DATA_DIR + row['Anchor'])
pos = io.imread(DATA_DIR + row['Positive'])
neg = io.imread(DATA_DIR + row['Negative'])

f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(10,5))

#ax1.set_title('Anchor')
#ax1.imshow(anc)

#ax2.set_title('Positive')
#ax2.imshow(pos)

#ax3.set_title('Negative')
#ax3.imshow(neg)

X_train, X_test = train_test_split(df, test_size = 0.2, random_state = 42)


#write custom dataset to return anchor, positive, negative when an index is set

class APN_Dataset(Dataset):

  def __init__(self, df):
    self.df = df

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]

    anc = io.imread(DATA_DIR + row.Anchor)
    pos = io.imread(DATA_DIR + row.Positive)
    neg = io.imread(DATA_DIR + row.Negative)

    #convert image into torch tensor
    anc = torch.from_numpy(anc).permute(2, 0, 1)/255.0
    pos = torch.from_numpy(pos).permute(2, 0, 1)/255.0
    neg = torch.from_numpy(neg).permute(2, 0, 1)/255.0

    return anc, pos, neg


X_train = APN_Dataset(X_train)
X_test = APN_Dataset(X_test)

#print(f'size of trainset: {len(X_train)}')
#print(f'size of testset: {len(X_test)}')

idx = 40
A,P,N = X_train[idx]

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (10,5))

#ax1.set_title('Anchor')
#ax1.imshow(A.numpy().transpose((1,2,0)), cmap = 'gray')

#ax2.set_title('Positive')
#ax2.imshow(P.numpy().transpose((1,2,0)), cmap = 'gray')

#ax3.set_title('Negative')
#ax3.imshow(N.numpy().transpose((1,2,0)), cmap = 'gray')

#load dataset into batches using DataLoader()
trainloader = DataLoader(X_train, batch_size = batch_size, shuffle = True)
testloader = DataLoader(X_test, batch_size = batch_size, shuffle = True)

#print(f"No. of batches in trainloader : {len(trainloader)}")
#print(f"No. of batches in validloader : {len(testloader)}")

for A,P,N in trainloader:
  break;

#print(f"One image batch shape : {A.shape}")
#in output, 3-no. channels, 128,64- height, width, 32-batch size


class APN_Model(nn.Module):

  def __init__(self, emb_size = 512):
    super(APN_Model, self).__init__()

    self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
    self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features= emb_size)

  def forward(self, images):

    embeddings = self.efficientnet(images)

    return embeddings

model = APN_Model()
model.to(device)


def train_fn(model, dataloader, optimizer, criterion):

  model.train()    #specify it because it turns ON the dropout and batch norms
  total_loss = 0.0

  #tqdm in order to track batches
  for A,P,N in tqdm(dataloader):
    #transfer anc, pos, neg to GPU
    A,P,N = A.to(device), P.to(device), N.to(device)

    A_embs = model(A)
    P_embs = model(P)
    N_embs = model(N)

    loss = criterion(A_embs, P_embs, N_embs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss/len(dataloader)


#same as train except we dont do any gradient computation and we dont need an optimizer
def eval_fn(model, dataloader, criterion):

  model.eval()    #specify it because it turns Off the Dropout and batch norms
  total_loss = 0.0

  with torch.no_grad():
    #tqdm in order to track batches
    for A,P,N in tqdm(dataloader):
      #transfer anc, pos, neg to GPU
      A,P,N = A.to(device), P.to(device), N.to(device)

      A_embs = model(A)
      P_embs = model(P)
      N_embs = model(N)

      loss = criterion(A_embs, P_embs, N_embs)


      total_loss += loss.item()

    return total_loss/len(dataloader)


#declare loss fn and optimizer
criterion = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_valid_loss = np.Inf

for i in range(epochs):

  train_loss = train_fn(model, trainloader, optimizer, criterion)
  test_loss = eval_fn(model, testloader, criterion)

  if test_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    best_valid_loss = test_loss
    #print('SAVED_ WEIGHTS_SUCCESS')

  #print(f'Epochs : {i+1} train loss: {train_loss} test loss: {test_loss}')


#find the embeddings
#pass each anchor image to the trained model to get the embeddings, save them in the form of csv file
def get_encoding_csv(model, anc_img_names):
  anc_img_names_arr = np.array(anc_img_names)
  encodings = []

  model.eval()
  with torch.no_grad():
    #tqdm will show progress bar, one by one name will come and we read the image
    for i in tqdm(anc_img_names_arr):
      A = io.imread(DATA_DIR+i)
      A = torch.from_numpy(A).permute(2,0,1)/255.0
      A = A.to(device)
      A_enc = model(A.unsqueeze(0))  #converts (channel, height, width) --> (batch size, channel, height, width)
      encodings.append(A_enc.squeeze().cpu().detach())#.numpy())

      encodings = np.array(encodings)
      encodings = pd.DataFrame(encodings)
      df_enc = pd.concat([pd.DataFrame(anc_img_names_arr), encodings], axis = 1)
      #df_enc = encodings
      #df_enc['img_name'] = anc_img_names_arr.tolist()

    #return encodings
    return df_enc


model.load_state_dict(torch.load('best_model.pt'))
df_enc = get_encoding_csv(model, df.Anchor)


df_enc.to_csv('database.csv', index = False)
df_enc.head()


def euclidean_dist(img_enc, anc_enc_arr):
  dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
  return dist


idx = 1
img_name = df_enc['Anchor'].iloc[idx]
img_path = DATA_DIR + img_name

img = io.imread(img_path)
img = torch.from_numpy(img).permute(2,0,1)/255.0

model.eval()
with torch.no_grad():
  img = img.to(device)
  img_enc = model(img.unsqueeze(0))
  img_enc = img_enc.detch().cpu().numpy()


anc_enc_arr = df_enc.iloc[:, 1:].to_numpy()
anc_img_names = df_enc['Anchor']


distance = []

for i in range(anc_enc_arr,.shape[0]):
  dist = euclidean_dist(img_enc, anc_enc_arr[i:i+1, :])
  distance = np.append(distance, dist)


closest_idx = np.argsort(distance)


from utils import plot_closest_imgs

plot_closest_imgs(anc_img_names, DATA_DIR, img, img_path, closest_idx, distance, no_of_closest = 5);
