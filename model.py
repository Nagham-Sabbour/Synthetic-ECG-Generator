import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels1=16, hidden_channels2=32, hidden_channels3=64, hidden_channels4=128, embedding_dim=32, num_classes=15):
        '''
        Encoder for the VAE model that converts data samples to latent representation.

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
        self.num_classes = num_classes

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

        # For input_length=500, kernel=3, stride=2, padding=1, 4 conv layers:
        # Layer lengths: 500 -> 250 -> 125 -> 63 -> 32
        # **Note: Recompute if architecture is changed
        flattened_size = hidden_channels4 * 32

        self.fc_mean = nn.Linear(flattened_size, embedding_dim)
        self.fc_logvar = nn.Linear(flattened_size, embedding_dim)


    def forward(self, x, y):
        '''
        Args:
            x: input signal, shape (batch, in_channels, 500)
            y: one-hot class label, shape (batch, num_classes)
        '''

        y_broadcast = y.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, y_broadcast], dim=1)

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

        x_mean = self.fc_mean(x)
        x_logvar = self.fc_logvar(x)

        return x_mean, x_logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim=32, hidden_channels1=128, hidden_channels2=64, hidden_channels3=32, hidden_channels4=16, out_channels=1, num_classes=15):
        '''
        Decoder for the VAE model that converts latent representation to data samples.

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


        # From Encoder: for input_length=500, kernel=3, stride=2, padding=1, 4 conv layers:
        # Layer lengths: 500 -> 250 -> 125 -> 63 -> 32
        # **Note: Recompute if architecture is changed
        self.bottleneck_length = 32
        layer_lengths = [32, 63, 125, 250, 500]

        self.hidden_channels1 = hidden_channels1

        self.fc = nn.Linear(in_features=embedding_dim + num_classes, out_features=hidden_channels1 * self.bottleneck_length)

        self.relu = nn.ReLU()

        self.up1 = nn.Upsample(size=layer_lengths[1])
        self.conv1 = nn.Conv1d(hidden_channels1, hidden_channels2, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels2)

        self.up2 = nn.Upsample(size=layer_lengths[2])
        self.conv2 = nn.Conv1d(hidden_channels2, hidden_channels3, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels3)

        self.up3 = nn.Upsample(size=layer_lengths[3])
        self.conv3 = nn.Conv1d(hidden_channels3, hidden_channels4, stride=1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels4)

        self.up4 = nn.Upsample(size=layer_lengths[4])
        self.conv4 = nn.Conv1d(hidden_channels4, out_channels, stride=1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(out_channels)

    def forward(self, x, y):
        '''
        Args:
            x: latent vector, shape (batch, embedding_dim)
            y: one-hot class label, shape (batch, num_classes)
        '''

        x = torch.cat([x, y], dim=1)
        
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
    def __init__(self, in_channels=1, hidden_channels1=16, hidden_channels2=32, hidden_channels3=64, hidden_channels4=128, embedding_dim=32, num_classes=15):
        '''
        Wrapper class for VAE model containing encoder and decoder.

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


    def reparameterize(self, mean, logvar): 
        '''
        Reparameterization trick.

        Args:
            mean: mean of latent representation as returned by encoder
            logvar: log variance of latent representation as returned by encoder
        '''

        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + eps * std


    def forward(self, x, y):
        '''
        Args:
            x: input signal, shape (batch, in_channels, 500)
            y: one-hot class label, shape (batch, num_classes)
        '''

        mean, logvar = self.encoder(x, y)
        reparam = self.reparameterize(mean, logvar)
        out = self.decoder(reparam, y)

        return mean, logvar, out

    