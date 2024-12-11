import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms, io
from skimage.metrics import structural_similarity as ssim
from sklearn.mixture import GaussianMixture
from torch.utils.data import random_split
from EncDec import *

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()


])

# Adding Noise to make more augmented images from the given clean images

def add_gaussian_noise(image, mean=0, std=0.1):
    image = image.float()

    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):

    image = image.float()

    salt_mask = torch.rand_like(image) < salt_prob
    pepper_mask = torch.rand_like(image) < pepper_prob

    # Add salt noise
    image[salt_mask] = 1.0

    # Add pepper noise
    image[pepper_mask] = 0.0

    return image

def add_speckle_noise(image, std=0.1):
    image = image.float()

    noise = torch.randn_like(image) * std
    noisy_image = image + image * noise
    return noisy_image


# Creating mapping between clean and augmented images
from sklearn.decomposition import PCA

def create_mapping(aug, clean):
    k = {i: 0 for i in range(10)}
    aug_image = {i: [] for i in range(10)}
    clean_image = {i: [] for i in range(10)}

    for (_, label) in clean:
        k[label] += 1

    for (img_path, label) in clean:
        for query in aug_image:
            if label == query:
                img = io.read_image(img_path)
                img = transforms.functional.to_pil_image(img)
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor()
                ])
                img = transform(img)
                
                # Flatten the image to a one-dimensional array
                img_flat = img.flatten().numpy()
                
                aug_image[query].append(img_flat)
                clean_image[query].append(img_path)

    gmms = {i: GaussianMixture(n_components=50) for i in range(10)}
    pca_images={i: [] for i in range(10)}

    for label in aug_image:

        x = np.stack(aug_image[label])    
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=128)  # or any other desired number of components
        x_pca = pca.fit_transform(x)
        pca_images[label].append(x_pca)
        gmms[label].fit(x_pca)
        # print(np.sum(pca.explained_variance_ratio_))

    perfect_clean={i: [] for i in range(10)}

    for label in pca_images:
        gmm=gmms[label]
        for i in range(50):
            max=0
            best_image=None
            for img_list in pca_images[label]:
                like=gmm.predict_proba(img_list)
                # print(like.shape)
                for j in range(len(img_list)):
                    
                   
                    if like[j][i]>max:
                        max=like[j][i]
                        best_image=clean_image[label][j]
            perfect_clean[label].append(best_image)



    mapping = {}
    # print("start mapping")
    for (img_path, label) in aug:
        img = io.read_image(img_path)
        img = transforms.functional.to_pil_image(img)
        img = transforms.functional.rgb_to_grayscale(img)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        img = transform(img)
        
        # Flatten the image and apply PCA
        img_flat = img.flatten().numpy()
        img_pca = pca.transform([img_flat])
        
        probabilities = gmms[label].predict_proba(img_pca)
        max_cluster = np.argmax(probabilities, axis=1)
  
        mapping[img_path] = perfect_clean[label][max_cluster[0]]

    # with open("mapping.pickle", 'wb') as file:
    #     pickle.dump(mapping, file)


    return mapping

class AlteredMNIST(Dataset):
    """
    Dataset description:
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Dataset stored in folder structure: DLA3/Data/aug and DLA3/Data/clean
    """

    def __init__(self, root_dir='../DLA3/Data', train=True):
        """
        Initialize the dataset
        :param root_dir: Path to the root directory containing 'aug' and 'clean' subdirectories
        :param train: Boolean indicating whether to load the training set (True) or test set (False)
        """
        self.root_dir = root_dir
        self.train = train
        
        # Determine the subdirectory based on train flag
        
        sub_dir_aug = os.path.join(self.root_dir, "aug")
        sub_dir_clean = os.path.join(self.root_dir, "clean")

        self.aug_data = []  # List to store (image, label) tuples
        self.clean_data = []  # List to store (image, label) tuples
        self.extra_aug = []
        # Iterate over the images in the subdirectory
        for filename in os.listdir(sub_dir_aug):
            if filename.endswith('.png'):
                # Extract label from filename
                try:
                    label = int(filename.split('_')[-1].split('.')[0])
                    img_path = os.path.join(sub_dir_aug, filename)
                    # Store image path and label
                    self.aug_data.append((img_path, label))
                except:
                    print('new image')


        
        for filename in os.listdir(sub_dir_clean):
            if filename.endswith('.png'):
                # Extract label from filename
                label = int(filename.split('_')[-1].split('.')[0])
                img_path = os.path.join(sub_dir_clean, filename)
                # Store image path and label
                self.clean_data.append((img_path, label))
        
        self.augmented_to_clean_mapping=create_mapping(self.aug_data,self.clean_data)
        mapping = self.augmented_to_clean_mapping 
        # with open("mapping.pickle", 'rb') as file:
            # mapping = pickle.load(file)
        
        for (img_path,label) in self.clean_data:
            # print("start new mapping")
            image = io.read_image(img_path)
           
            augmented_image1 = add_gaussian_noise(image)
            augmented_image1 = (torch.clamp(augmented_image1, 0, 1) * 255).byte()
            file_name, file_extension = os.path.splitext(os.path.basename(img_path))
            new_file_name = f"{file_name}_new{1}_{label}{file_extension}"

            augmented_path = os.path.join(sub_dir_aug, new_file_name)
            self.extra_aug.append((augmented_path, label))
     
            io.write_png(augmented_image1,augmented_path)
            mapping[augmented_path]=img_path

            augmented_image2 = add_salt_and_pepper_noise(image)
            augmented_image2 = (torch.clamp(augmented_image2, 0, 1) * 255).byte()
            file_name, file_extension = os.path.splitext(os.path.basename(img_path))
            new_file_name = f"{file_name}_new{2}_{label}{file_extension}"

            augmented_path = os.path.join(sub_dir_aug, new_file_name)
            self.extra_aug.append((augmented_path, label))

            io.write_png(augmented_image2,augmented_path)
            mapping[augmented_path]=img_path


            augmented_image3 = add_speckle_noise(image)
            augmented_image3 = (torch.clamp(augmented_image3, 0, 1) * 255).byte()
            file_name, file_extension = os.path.splitext(os.path.basename(img_path))
            new_file_name = f"{file_name}_new{3}_{label}{file_extension}"
            augmented_path = os.path.join(sub_dir_aug, new_file_name)
            self.extra_aug.append((augmented_path, label))

            io.write_png(augmented_image3,augmented_path )
            mapping[augmented_path]=img_path




        # print(len(list(mapping.keys())))
        # print(len(self.aug_data))


        self.augmented_to_clean_mapping=mapping
    
    def __len__(self):
     
        return len(self.aug_data)

    def __getitem__(self, idx):
      
        img_path, label = self.aug_data[idx]
        
        clean_image_path=self.augmented_to_clean_mapping[img_path]
       
        img = io.read_image(img_path)
        
        img = transforms.functional.to_pil_image(img)
        # Apply transformations
        img = transforms.functional.rgb_to_grayscale(img)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        img = transform(img)

        clean = io.read_image(clean_image_path)

        # Convert to PIL Image
        clean = transforms.functional.to_pil_image(clean)
        clean = transforms.functional.rgb_to_grayscale(clean)

        # Apply transformations
        
        clean = transform(clean)

        return img, label, clean

def custom_collate(batch):
    """
    Custom collate function to handle resizing of images in the batch
    """
    # Separate images and labels
    images, labels, clean = zip(*batch)

    # Convert RGB images to grayscale
    images = [transforms.functional.rgb_to_grayscale(img) if img.size(0) == 3 else img for img in images]
    clean = [transforms.functional.rgb_to_grayscale(img) if img.size(0) == 3 else img for img in clean]

    images = torch.stack(images)
    clean = torch.stack(clean)

    return images, torch.tensor(labels), clean

    # Resize images to a fixed size and convert to tensors
    images = torch.stack(images)

    return images, torch.tensor(labels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )



    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out   


class Encoder(nn.Module):
    
    """
    Encoder class
    """
    def __init__(self, num_classes=10, type="AE"):
        self.type=type
        if type=="AE":
            super(Encoder, self).__init__()
            self.resnet1 = ResidualBlock(1, 16)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet2 = ResidualBlock(16, 32)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet3 = ResidualBlock(32, 64)  # Assuming input channels are 1 (grayscale MNIST)
        elif type=="VAE":
            super(Encoder, self).__init__()
            self.resnet1 = ResidualBlock(1, 16)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet2 = ResidualBlock(16,32)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet3 = ResidualBlock(32,64)  # Assuming input channels are 1 (graytscale MNIST)
            self.fc_mu = nn.Linear(64,64)
            self.fc_logvar = nn.Linear(64,64)
        elif type=="CVAE":
            super(Encoder, self).__init__()
            self.resnet1 = ResidualBlock(1, 16)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet2 = ResidualBlock(16,32)  # Assuming input channels are 1 (grayscale MNIST)
            self.resnet3 = ResidualBlock(32,64)  # Assuming input channels are 1 (graytscale MNIST)
            self.fc_mu = nn.Linear(64,num_classes + 64 * 2)
            self.fc_logvar = nn.Linear(64,num_classes + 64 * 2)
            # self.fc = nn.Linear(64, num_classes + 64 * 2)

            
    def forward(self, x):
        if self.type=="AE":
            x = self.resnet1(x)
            x = self.resnet2(x)
            x = self.resnet3(x)
            x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
            return x
        else:
            x = self.resnet1(x)
            x = self.resnet2(x)
            x = self.resnet3(x)
            x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
           
            mu = self.fc_mu(x)
            log_var = self.fc_logvar(x)

            return mu,log_var
        

class Decoder(nn.Module):
    """
    Decoder class
    """
    def __init__(self, type="AE",num_classes=10):
        if type == "AE":
            super(Decoder, self).__init__()
            latent_dim=64
            self.latent_dim = latent_dim
            self.fc = nn.Linear(latent_dim, 64*7*7)
            self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(1)
        elif type == "VAE":
            super(Decoder, self).__init__()
            latent_dim=64
            self.latent_dim = latent_dim
            self.fc = nn.Linear(latent_dim, 64*7*7)
            self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(1)
        elif type == "CVAE":
            super(Decoder, self).__init__()
            latent_dim=num_classes + 64 * 2
            self.latent_dim = latent_dim
            self.fc = nn.Linear(latent_dim, 64*7*7)

            self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(1)


    def forward(self, x):
        x=self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        
        return x


class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def __call__(self, outputs, targets):
        return self.loss_fn(outputs, targets)


def ParameterSelector(E,D):
    
    return list(E.parameters()) + list(D.parameters())


# Define AETrainer class for training the autoencoder
class AETrainer:
    def __init__(self, data_loader, encoder, decoder, criterion, optimizer, proc="gpu"):
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader=data_loader
        dataset=self.dataloader.dataset
        
        BATCH_SIZE=64

        extra=dataset.extra_aug
        aug=dataset.aug_data
        for x in extra:
            aug.append(x)
        dataset.aug_data=aug
        # Initialize the DataLoader for train and test sets
        
   
        self.proc = proc
        self.train(EPOCH)
        

    def train(self, num_epochs, checkpoint_path=None):
        device = next(self.encoder.parameters()).device
        tsne_data = []
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0
            train_ssim = 0.0
            num=0
            for minibatch, data in enumerate(self.dataloader):
                inputs, _, clean_image = data
                inputs = inputs.to(device)
                self.optimizer.zero_grad()
                latent = self.encoder(inputs)
                outputs = self.decoder(latent)
                loss = self.criterion(outputs, clean_image)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
                # Compute SSIM score after every 10th minibatch
                
                if minibatch % 10 == 0:
                    ssim_score = structure_similarity_index(outputs, clean_image)
                    train_ssim += ssim_score
                    num+=1
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, minibatch, loss.item(), ssim_score))

            train_loss /= len(self.dataloader.dataset)
            train_ssim /= num
            # Compute SSIM score for the whole epoch
            if self.proc == "gpu":
                inputs = inputs.cpu()
                outputs = outputs.cpu()
            print("----- Epoch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, train_loss, train_ssim))

           
            # Plot TSNE embedding after every 10 epochs
            if ((epoch + 1) % 10 == 0):
                data,label=self.get_latent_embeddings(device)
                tsne_data.append(data)
                self.plot_tsne(tsne_data[-1],label,save_path=f"AE_epoch_{epoch+1}.png")
        encoder_path = 'encoder_AE.pth'
        decoder_path = 'decoder_AE.pth'

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)



    def get_latent_embeddings(self, device):
        self.encoder.eval()
        latent_embeddings = []
        labels=[]
        with torch.no_grad():
            for data in self.dataloader:
                inputs, lab, clean = data
                labels.append(lab)
                inputs = inputs.to(device)
                latent = self.encoder(inputs)
                latent_embeddings.append(latent.cpu().numpy())
        return np.concatenate(latent_embeddings),np.concatenate(labels)

    def plot_tsne(self, tsne_embedding, labels, save_path=None):
        tsne = TSNE(n_components=3, n_iter=300, verbose=0, perplexity=40)
        tsne_embedding = tsne.fit_transform(tsne_embedding)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define a colormap for better visualization
        colormap = plt.cm.jet(np.linspace(0, 1, len(np.unique(labels))))

        # Scatter plot with different colors for different classes
        for i, label in enumerate(np.unique(labels)):
            indices = np.where(labels == label)
            ax.scatter(tsne_embedding[indices, 0], tsne_embedding[indices, 1], tsne_embedding[indices, 2], color=colormap[i], label=str(label))

        ax.set_title('TSNE Embedding')
        ax.legend()

        # Save the plot if save_path is provided
        if save_path is not None:
            plt.savefig(save_path)




class AE_TRAINED:
    """
    Load trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm
    """
    def __init__(self,gpu):
        self.gpu=gpu
        self.encoder = Encoder()  # Instantiate your encoder class
        self.decoder = Decoder()  # Instantiate your decoder class
        # if self.gpu == True:


    def from_path(self,sample, original, type):
        encoder_path="encoder_AE.pth"
        decoder_path="decoder_AE.pth"
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
                
        # Load and preprocess the first image
        img1 = io.read_image(sample)
        transform1 = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((28, 28))
        ])
        img1 = transform1(img1).unsqueeze(0)

        # Load and preprocess the second image
        img2 = io.read_image(original)
        transform2 = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((28, 28))
        ])
        img2 = transform2(img2).unsqueeze(0)
        # print(img1.shape)
        # print(img2.shape)
        # img1 = transform(img1).unsqueeze(0)  
        # img2 = transform(img2).unsqueeze(0)  
        with torch.no_grad():
            img1 = img1.to(torch.float)
            img2 = img2.to(torch.float)
            latent = self.encoder(img1)
            output = self.decoder(latent)



        if type == "SSIM":
            return structure_similarity_index(output, img2)

        if type == "PSNR":
            return peak_signal_to_noise_ratio(output, img2)


class VAELossFn:

    def __init__(self):
        super(VAELossFn, self).__init__()

    def __call__(self, recon_x, x, mu, logvar):
        ELBO = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return ELBO + KL_divergence



class VAETrainer:
    def __init__(self, data_loader, encoder, decoder, criterion, optimizer, proc="gpu"):
        encoder = Encoder(type="VAE")
        decoder = Decoder(type="VAE")
        LEARNING_RATE=0.001
        optimizer= torch.optim.Adam(ParameterSelector(encoder, decoder), lr=LEARNING_RATE)
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader=data_loader
        dataset=self.dataloader.dataset
       
        BATCH_SIZE=64

        extra=dataset.extra_aug
        aug=dataset.aug_data
        for x in extra:
            aug.append(x)
        dataset.aug_data=aug
        
   
        self.proc = proc
        self.train(EPOCH)
        

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def train(self, num_epochs, checkpoint_path=None):
        device = next(self.encoder.parameters()).device
        tsne_data = []
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0
            train_ssim = 0.0
            num=0
            for minibatch, data in enumerate(self.dataloader):
                inputs, _, clean_image = data
                inputs = inputs.to(device)
                self.optimizer.zero_grad()
                # print(inputs.shape)
                mu,log_var = self.encoder(inputs)
                latent = self.reparameterize(mu,log_var)
                outputs = self.decoder(latent)
                loss = self.criterion(outputs, clean_image, mu,log_var)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
                # Compute SSIM score after every 10th minibatch
                
                if minibatch % 10 == 0:
                    ssim_score = structure_similarity_index(outputs, clean_image)
                    train_ssim += ssim_score
                    num+=1
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, minibatch, loss.item(), ssim_score))

            train_loss /= len(self.dataloader.dataset)
            train_ssim /= num
            # Compute SSIM score for the whole epoch
            if self.proc == "gpu":
                inputs = inputs.cpu()
                outputs = outputs.cpu()
            print("----- Epoch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, train_loss, train_ssim))


            # Plot TSNE embedding after every 10 epochs
            if ((epoch + 1) % 10 == 0):
                data,label=self.get_latent_embeddings(device)
                tsne_data.append(data)
                self.plot_tsne(tsne_data[-1],label,save_path=f"VAE_epoch_{epoch+1}.png")
        
        encoder_path = 'encoder_VAE.pth'
        decoder_path = 'decoder_VAE.pth'

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)



    def get_latent_embeddings(self, device):
        self.encoder.eval()
        latent_embeddings = []
        labels=[]
        with torch.no_grad():
            for data in self.dataloader:
                inputs, lab, clean = data
                inputs = inputs.to(device)
                mu,log_var = self.encoder(inputs)
                
                latent_embeddings.append(mu.cpu().numpy())
                labels.append(lab)
        return np.concatenate(latent_embeddings),np.concatenate(labels)

    def plot_tsne(self, tsne_embedding, labels, save_path=None):
        tsne = TSNE(n_components=3, n_iter=300, verbose=0, perplexity=40)
        tsne_embedding = tsne.fit_transform(tsne_embedding)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define a colormap for better visualization
        colormap = plt.cm.jet(np.linspace(0, 1, len(np.unique(labels))))

        # Scatter plot with different colors for different classes
        for i, label in enumerate(np.unique(labels)):
            indices = np.where(labels == label)
            ax.scatter(tsne_embedding[indices, 0], tsne_embedding[indices, 1], tsne_embedding[indices, 2], color=colormap[i], label=str(label))

        ax.set_title('TSNE Embedding')
        ax.legend()

        # Save the plot if save_path is provided
        if save_path is not None:
            plt.savefig(save_path)



def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """

    def __init__(self,gpu):
        self.encoder = Encoder(type="VAE")  # Instantiate your encoder class
        self.decoder = Decoder(type="VAE")  # Instantiate your decoder class
        self.gpu=gpu
        if self.gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


    def from_path(self,sample, original, type):
        encoder_path="encoder_VAE.pth"
        decoder_path="decoder_VAE.pth"
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
                
        # Load and preprocess the first image
        img1 = io.read_image(sample)
        transform1 = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((28, 28))
        ])
        img1 = transform1(img1).unsqueeze(0)

        # Load and preprocess the second image
        img2 = io.read_image(original)
        transform2 = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((28, 28))
        ])
        img2 = transform2(img2).unsqueeze(0)

        # print(img1.shape)
        # print(img2.shape)
        # img1 = transform(img1).unsqueeze(0)  
        # img2 = transform(img2).unsqueeze(0)  
        if self.gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            img1 = img1.to(torch.float)
            img2 = img2.to(torch.float)
            img1 = img1.to(device)
            mu,logvar = self.encoder(img1)
            latent=reparam(mu,logvar)
            output = self.decoder(latent)



        if type == "SSIM":
            return structure_similarity_index(output, img2)

        if type == "PSNR":
            return peak_signal_to_noise_ratio(output, img2)

class CVAELossFn():
    def __init__(self):
        super(CVAELossFn, self).__init__()

    def __call__(self, recon_x, x, mu, logvar):
        ELBO = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return ELBO + KL_divergence

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    def __init__(self, data_loader, encoder, decoder, criterion, optimizer, proc="gpu"):
        encoder = Encoder(type="CVAE")
        decoder = Decoder(type="CVAE")
        LEARNING_RATE=0.001
        optimizer= torch.optim.Adam(ParameterSelector(encoder, decoder), lr=LEARNING_RATE)
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader=data_loader
        dataset=self.dataloader.dataset
       
        BATCH_SIZE=64

        extra=dataset.extra_aug
        aug=dataset.aug_data
        for x in extra:
            aug.append(x)
        dataset.aug_data=aug
        
        self.proc = proc
        self.train(EPOCH)
        


    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def train(self, num_epochs, checkpoint_path=None):
        device = next(self.encoder.parameters()).device
        tsne_data = []
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0
            train_ssim = 0.0
            num=0
            for minibatch, data in enumerate(self.dataloader):
                inputs,label, clean_image = data
                inputs = inputs.to(device)
                self.optimizer.zero_grad()
                
                mu,log_var = self.encoder(inputs)
                latent = self.reparameterize(mu,log_var)
                outputs = self.decoder(latent)
                loss = self.criterion(outputs,clean_image,mu,log_var)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
                # Compute SSIM score after every 10th minibatch
                
                if minibatch % 10 == 0:
                    ssim_score = structure_similarity_index(outputs, clean_image)
                    train_ssim += ssim_score
                    num+=1
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, minibatch, loss.item(), ssim_score))

            train_loss /= len(self.dataloader.dataset)
            train_ssim /= num
            # Compute SSIM score for the whole epoch
            if self.proc == "gpu":
                inputs = inputs.cpu()
                outputs = outputs.cpu()
            print("----- Epoch:{}, Loss:{}, Similarity:{:.4f}".format(epoch + 1, train_loss, train_ssim))

            # Plot TSNE embedding after every 10 epochs
            if ((epoch + 1) % 10 == 0):
                data,label=self.get_latent_embeddings(device)
                tsne_data.append(data)
                self.plot_tsne(tsne_data[-1],label,save_path=f"CVAE_epoch_{epoch+1}.png")
        encoder_path = 'encoder_CVAE.pth'
        decoder_path = 'decoder_CVAE.pth'

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)



    def get_latent_embeddings(self, device):
        self.encoder.eval()
        latent_embeddings = []
        label=[]
        with torch.no_grad():
            for data in self.dataloader:
                inputs, lab, clean = data
                inputs = inputs.to(device)
                label.append(lab)
                mu,log_var = self.encoder(inputs)
                latent = self.reparameterize(mu,log_var)
                latent_embeddings.append(latent.cpu().numpy())
        return np.concatenate(latent_embeddings), np.concatenate(label)

    def plot_tsne(self, tsne_embedding, labels, save_path=None):
        tsne = TSNE(n_components=3, n_iter=300, verbose=0, perplexity=40)
        tsne_embedding = tsne.fit_transform(tsne_embedding)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define a colormap for better visualization
        colormap = plt.cm.jet(np.linspace(0, 1, len(np.unique(labels))))

        # Scatter plot with different colors for different classes
        for i, label in enumerate(np.unique(labels)):
            indices = np.where(labels == label)
            ax.scatter(tsne_embedding[indices, 0], tsne_embedding[indices, 1], tsne_embedding[indices, 2], color=colormap[i], label=str(label))

        ax.set_title('TSNE Embedding')
        ax.legend()

        # Save the plot if save_path is provided
        if save_path is not None:
            plt.savefig(save_path)

    


class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    def __init__(self):
        # self.encoder = Encoder()  # Instantiate your encoder class
        self.decoder = Decoder(type="CVAE")  # Instantiate your decoder class

    def save_image(self,digit, save_path):
        # encoder_path="encoder_CVAE.pth"
        decoder_path="decoder_CVAE.pth"
        
        self.decoder.load_state_dict(torch.load(decoder_path))
        one_hot = torch.zeros(10)
        one_hot[digit] = 1
        latent_dim = 128
        latent_sample = torch.randn(1, latent_dim)  
        generated_image = self.decoder(torch.cat((latent_sample, one_hot.unsqueeze(0)),dim=1))
        pil_image=transforms.ToPILImage()(generated_image.squeeze())
        filename=os.path.join(save_path,f"generated_image_{digit}.png")
        pil_image.save(filename)



def structure_similarity_index(img1, img2):
    # if gpu:
    img1 = img1.cpu().detach()
    img2 = img2.cpu().detach()
    ssim_score = ssim(img1.squeeze().cpu().numpy(), img2.squeeze().cpu().numpy(),data_range=(img1.max()-img1.min()).item())
    return ssim_score


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()
