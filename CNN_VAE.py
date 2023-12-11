import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------------------------------------
# hyperparams
# -----------------------------------------------------------------------------

np.random.seed(0)
batch_size = 64
num_epochs = 1


print('Starting....')
# -----------------------------------------------------------------------------

data_folder = './ProcessedData'
file_list = os.listdir(data_folder)

one_list = np.load('./one_list.npy', allow_pickle=True)
zero_list = np.load('./zero_list.npy', allow_pickle=True)
print(f' length list 1: {len(one_list)}')
print(f'length list 0: {len(zero_list)}')
# print('Done!')

# -----------------------------------------------------------------------------
# shuffling + splitting for test and train
# -----------------------------------------------------------------------------
label_0_files = zero_list
label_1_files = (one_list)


print(type(label_0_files), type(label_1_files))
len_zero = (len(zero_list)) //2
len_one = (len(one_list)) //2

# print(f'Half {len(len_zero)}')
# print(label_0_files[:100].shape+ label_1_files[:100].shape)

np.random.shuffle(label_0_files)
np.random.shuffle(label_1_files)

half = 11300
train_files = list(label_0_files)[:half] + list(label_1_files)[:half]
test_files = list(label_0_files)[half:] + list(label_1_files)[half:]


# -----------------------------------------------------------------------------
# creating a customdataset
# -----------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, file_list, data_folder):
        self.file_list = file_list
        self.data_folder = data_folder

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
     
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = np.load(file_path, allow_pickle=True)

        mfcc_lfcc_array = data[0]
        cqcc_lpcc_array = data[1]
        label = data[2]

        concatenated_features = np.concatenate((mfcc_lfcc_array[0], mfcc_lfcc_array[1]), axis=-1)
        concatenated_features = np.expand_dims(concatenated_features, axis=0)
        

        return concatenated_features, label


train_dataset = CustomDataset(train_files, data_folder)
test_dataset = CustomDataset(test_files, data_folder)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# getting the shapes for inputs and labels
# -----------------------------------------------------------------------------
print('-----------Going through dataloaders to see the shapes')
for inputs, labels in train_loader:
    print(inputs.shape)
    print(labels.shape)
    break

for inputs, labels in test_loader:
    print(inputs.shape)
    print(labels.shape)
    break


labels_count = {0: 0, 1: 0}

for features, label in train_dataset:
    labels_count[label] += 1

# print("Label 0 count:", labels_count[0])
# print("Label 1 count:", labels_count[1])

# # -----------------------------------------------------------------------------
# # VAE
# # -----------------------------------------------------------------------------

input_size =  4
hidden_size = 512
latent_size = 128
learning_rate = 0.001
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc_mu = nn.Linear(3328, latent_size)
        self.fc_logvar = nn.Linear(3328, latent_size)

        # Decoder
        self.fc_decode = nn.Linear(latent_size, 3328)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        # print(f'SHAPE {x.shape}')
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # print(f'shape of mean {mu.shape}')
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc_decode(z))
        # print(f'SHAPE OF Z {z.shape}')
        z = z.view(z.size(0), 64, 13, 4)  
        # print(f'SHAPE OF Z {z.shape}')
        x = F.relu(self.deconv1(z))
        x = F.relu(self.deconv2(z))
        x_recon = torch.sigmoid(self.deconv3(x))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar




# # Create VAE model, loss function, and optimizer
vae_model = VAE().double() 
criterion = nn.MSELoss().double() 
optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

print('-------------VAE training starting....')
# Training loop
for epoch in range(num_epochs):
    for data, _ in train_loader:
        # Get input data
        inputs = data.double() 
        # print(inputs.shape)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        recon, mu, logvar = vae_model(inputs)
        # print(mu.shape) # 64, 128

        # Compute loss
        reconstruction_loss = criterion(recon, inputs)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_divergence

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print training statistics
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# # Save the trained model
torch.save(vae_model.state_dict(), './vae.pth')


# -----------------------------------------------------------------------------
# CNN
# -----------------------------------------------------------------------------
in_size = 1

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # self.fc_decode = nn.Linear(128, 3328)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
    
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 , 12)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(12, 2)  

    def forward(self, x):
        # print('------------')
        # x = F.relu(self.fc_decode(x))
        # print(f'shape of x {x.shape}')
        # x = x.view(x.size(0), 64, 13, 4)
        print(f'shape of x {x.shape}')  
  
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
      
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        print(f' after flattening {x.shape}')
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu3(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

model = AudioClassifier()
vae_model = VAE()
PATH = './vae.pth'
state_dict = torch.load(PATH)
vae_model.load_state_dict(state_dict)
vae_model.eval()
# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
# print('-----------Training')
# loss_values = []
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, labels in train_loader:
   
#         inputs = torch.tensor(inputs, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.long)
#         mu, logvar = vae_model.encode(inputs)
#         z = vae_model.reparameterize(mu, logvar)

       
#         # print(f' Shape of latent is {z.shape}')
#         outputs = model(z)
#         # print(outputs.shape, labels.shape)

  
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     loss_values.append(loss.item())
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# plt.plot(range(1, num_epochs+1), loss_values, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.legend()
# plt.savefig('../los_per_epoch_vae.png')
# plt.show()


model.eval()
correct = 0
predicted_labels = []
total = 0
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
    
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        mu, logvar = vae_model.encode(inputs)
        z = vae_model.reparameterize(mu, logvar)
        z = torch.unsqueeze(z, 0)

       
        print(f' Shape of latent is {z.shape}')
        outputs = model(z)
        # outputs = torch.unsqueeze(outputs , 0)

        _, predicted = torch.max(outputs, 1)
        print(f'total size {outputs.shape}')

        total += labels.size(0)
        # print(f'total size {total}')
        print(f'predicted {predicted.shape} labels {labels.shape}')
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())


accuracy = correct / total
print(f"Test Accuracy: {accuracy}")