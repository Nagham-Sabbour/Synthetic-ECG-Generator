import torch
import torch.nn as nn


SIGNAL_LENGTH = 500
BOTTLENECK_LENGTH = 32
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_NUM_CLASSES = 11


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels1=16, hidden_channels2=32, hidden_channels3=64, hidden_channels4=128, embedding_dim=DEFAULT_EMBEDDING_DIM, num_classes=DEFAULT_NUM_CLASSES):
        '''
        Encode a conditional ECG segment into mean and log-variance vectors
        
        Args:
            in_channels: is 1 for preprocessed ECG dataset (1 lead)
            hidden_channels1: number of channels after first conv layer
            hidden_channels2: number of channels after second conv layer
            hidden_channels3: number of channels after third conv layer
            hidden_channels4: number of channels after fourth conv layer
            embedding_dim = number of output channels
            num_classes: number of diagnostic classes for label conditioning
        '''
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels + num_classes, hidden_channels1, stride=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels1)

        self.conv2 = nn.Conv1d(hidden_channels1, hidden_channels2, stride=2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels2)

        self.conv3 = nn.Conv1d(hidden_channels2, hidden_channels3, stride=2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels3)

        self.conv4 = nn.Conv1d(hidden_channels3, hidden_channels4, stride=2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_channels4)

        self.flatten = nn.Flatten()

        # Layer lengths: 500 -> 250 -> 125 -> 63 -> 32 after the four encoder layers
        flattened_size = hidden_channels4 * BOTTLENECK_LENGTH
        self.fc_mean = nn.Linear(flattened_size, embedding_dim)
        self.fc_logvar = nn.Linear(flattened_size, embedding_dim)

    def forward(self, signals, labels):
        '''
        Encode signals with one-hot labels attached across time
        
        Args:
            signals: input signal, shape (batch, in_channels, 500)
            labels: one-hot class label, shape (batch, num_classes)
        '''
        labels = labels.unsqueeze(-1).expand(-1, -1, signals.size(-1))
        x = torch.cat([signals, labels], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.flatten(x)

        return self.fc_mean(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=DEFAULT_EMBEDDING_DIM, hidden_channels1=128, hidden_channels2=64, hidden_channels3=32, hidden_channels4=16, out_channels=1, num_classes=DEFAULT_NUM_CLASSES):
        '''
        Decode a latent vector and one-hot label into an ECG segment
    
        Args: 
            embedding_dim = number of input channels
            hidden_channels1: number of channels before first conv layer
            hidden_channels2: number of channels before second conv layer
            hidden_channels3: number of channels before third conv layer
            hidden_channels4: number of channels before fourth conv layer
            out_channels: is 1 for preprocessed ECG dataset (1 lead)
            num_classes: number of diagnostic classes for label conditioning
        '''
        super().__init__()

        self.bottleneck_length = BOTTLENECK_LENGTH
        self.hidden_channels1 = hidden_channels1
        layer_lengths = [32, 63, 125, 250, SIGNAL_LENGTH]

        self.fc = nn.Linear(
            embedding_dim + num_classes,
            hidden_channels1 * self.bottleneck_length,
        )
        self.relu = nn.ReLU()

        self.up1 = nn.Upsample(size=layer_lengths[1])
        self.conv1 = nn.Conv1d(hidden_channels1, hidden_channels2, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels2)

        self.up2 = nn.Upsample(size=layer_lengths[2])
        self.conv2 = nn.Conv1d(hidden_channels2, hidden_channels3, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels3)

        self.up3 = nn.Upsample(size=layer_lengths[3])
        self.conv3 = nn.Conv1d(hidden_channels3, hidden_channels4, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels4)

        self.up4 = nn.Upsample(size=layer_lengths[4])
        self.conv4 = nn.Conv1d(hidden_channels4, out_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(out_channels)

    def forward(self, latent, labels):
        '''
        Decode one latent vector for each supplied label
        
        Args:
            latent: latent vector, shape (batch, embedding_dim)
            labels: one-hot class label, shape (batch, num_classes)
        '''
        x = torch.cat([latent, labels], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels1, self.bottleneck_length)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.up3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.up4(x)
        x = self.conv4(x)

        return x


class VAE(nn.Module):

    def __init__(self, in_channels=1, hidden_channels1=16, hidden_channels2=32, hidden_channels3=64, hidden_channels4=128, embedding_dim=DEFAULT_EMBEDDING_DIM, num_classes=DEFAULT_NUM_CLASSES):
        '''
        Conditional VAE model containing encoder and decoder for 500-sample Lead II ECG segments

        Args:
            in_channels: number of input channels for encoder / output channels for decoder - is 1 for preprocessed ECG dataset (1 lead)
            hidden_channels1: number of channels after first conv layer of encoder / before fourth (last) conv layer of decoder
            hidden_channels2: number of channels after second conv layer / before third conv layer of decoder
            hidden_channels3: number of channels after third conv layer / before second conv layer of decoder
            hidden_channels4: number of channels after fourth conv layer / before first conv layer of decoder
            embedding_dim = number of output channels for encoder / input channels for decoder
            num_classes: number of diagnostic classes for label conditioning
        '''
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, hidden_channels1=hidden_channels1, hidden_channels2=hidden_channels2, hidden_channels3=hidden_channels3, hidden_channels4=hidden_channels4, embedding_dim=embedding_dim, num_classes=num_classes)
        self.decoder = Decoder(embedding_dim=embedding_dim, hidden_channels1=hidden_channels4, hidden_channels2=hidden_channels3, hidden_channels3=hidden_channels2, hidden_channels4=hidden_channels1, out_channels=in_channels, num_classes=num_classes)

    @staticmethod
    def reparameterize(mean, logvar):
        '''
        Reparameterization trick.
        Sample from the latent distribution during the forward pass

        Args:
            mean: mean of latent representation as returned by encoder
            logvar: log variance of latent representation as returned by encoder
        '''

        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + eps * std


    def forward(self, signals, labels):
        '''
        Return the latent parameters and reconstructed ECG signals
        
        Args:
            signals: input signal, shape (batch, in_channels, 500)
            labels: one-hot class label, shape (batch, num_classes)
        '''
        mean, logvar = self.encoder(signals, labels)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(latent, labels)

        return mean, logvar, reconstruction


class Discriminator(nn.Module):

    def __init__(self, in_channels=1, hidden_channels1=16, hidden_channels2=32, hidden_channels3=64, hidden_channels4=128, num_classes=DEFAULT_NUM_CLASSES):
        '''
        Class-conditional discriminator for VAE-GAN fine-tuning  to classify real vs fake samples
        
        Args:
            in_channels: is 1 for preprocessed ECG dataset (1 lead)
            hidden_channels1: number of channels after first conv layer
            hidden_channels2: number of channels after second conv layer
            hidden_channels3: number of channels after third conv layer
            hidden_channels4: number of channels after fourth conv layer
            num_classes: number of diagnostic classes for label conditioning
        '''
        super().__init__()

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv1d(in_channels + num_classes, hidden_channels1, stride=2, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(hidden_channels1, hidden_channels2, stride=2, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(hidden_channels2, hidden_channels3, stride=2, kernel_size=3, padding=1)

        self.conv4 = nn.Conv1d(hidden_channels3, hidden_channels4, stride=2, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        flattened_size = hidden_channels4 * BOTTLENECK_LENGTH
        self.fc = nn.Linear(flattened_size, 1)

    def forward(self, signals, labels):
        '''
        Return one real-versus-fake logit for each ECG signal
        
        Args:
            x: input signal, shape (batch, in_channels, 500)
            y: one-hot class label, shape (batch, num_classes)
        '''
        labels = labels.unsqueeze(-1).expand(-1, -1, signals.size(-1))
        x = torch.cat([signals, labels], dim=1)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.fc(x)

        return x