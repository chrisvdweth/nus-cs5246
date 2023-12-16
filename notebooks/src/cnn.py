import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnSentenceClassifier(nn.Module):
    
    def __init__(self, vocab_size, output_size, embed_size):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Convolutional Layers for filter sizes 2, 3, and 4
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, embed_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, embed_size))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, embed_size))
            
        # Linear layer to map to output (#conv_layers*#filters)
        self.out = nn.Linear(3*2, output_size)
            
    def forward(self, inputs):
        # Push through embedding layer and add "in_channel" dimensions
        X = self.embedding(inputs).unsqueeze(1) # Shape: (N x C_in=1 x S x E)
        # Push through all 3 convolutional layers
        X2 = self.conv2(F.relu(X)).squeeze(-1)
        X3 = self.conv3(F.relu(X)).squeeze(-1)
        X4 = self.conv4(F.relu(X)).squeeze(-1)
        # Perform 1-max pooling
        X2, _ = torch.max(X2, dim=-1)
        X3, _ = torch.max(X3, dim=-1)
        X4, _ = torch.max(X4, dim=-1)
        # Concatenate all outputs from the different conv layers
        X = torch.cat([X2, X3, X4], -1)
        # Push through last linear layer
        X = self.out(X)
        # Return softmax probabilities
        return F.log_softmax(X, dim=-1)
    
    
    
    
    
class CnnTextClassifier(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size, self.params.embed_size)

        self.flatten_size = 0

        self.conv_layers = nn.ModuleDict()
        for ks in self.params.conv_kernel_sizes:
            self.conv_layers['conv_{}'.format(ks)] = nn.Conv2d(in_channels=1,
                                                               out_channels=self.params.out_channels,
                                                               kernel_size=(ks, self.params.embed_size),
                                                               stride=self.params.conv_stride,
                                                               padding=(self.params.conv_padding, 0))
            # Calculate the length of the conv output
            conv_out_size = self._calc_conv_output_size(self.params.seq_len,
                                                        ks,
                                                        self.params.conv_stride,
                                                        self.params.conv_padding)
            # Calculate the length of the maxpool output
            maxpool_out_size = self._calc_maxpool_output_size(conv_out_size,
                                                              self.params.maxpool_kernel_size,
                                                              self.params.maxpool_padding,
                                                              self.params.maxpool_kernel_size,
                                                              1)
            # Add all lengths together
            self.flatten_size += maxpool_out_size
            
        self.flatten_size *= self.params.out_channels

        self.maxpool_layers = nn.ModuleDict()
        for ks in self.params.conv_kernel_sizes:
            #self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool2d(kernel_size=(self.params.maxpool_kernel_size, self.params.embed_size))
            self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool2d(kernel_size=(1, self.params.maxpool_kernel_size))
            #self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool1d(kernel_size=self.params.maxpool_kernel_size)

        self.linear_sizes = [self.flatten_size] + self.params.linear_sizes

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_sizes)-1):
            self.linears.append(nn.Linear(self.linear_sizes[i], self.linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            if self.params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=self.params.linear_dropout))


        self.out = nn.Linear(self.linear_sizes[-1], self.params.output_size)


    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        X = self.embedding(inputs)
        # Embedding output shape: N x S x E
        # Turn (N x S x E) into (N x C_in=1 x S x E) for CNN
        # (note: embedding dimension = input channels)
        X = X.unsqueeze(1)
        # Conv1d input shape: batch size x input channels x input length
        all_outs = []
        for ks in self.params.conv_kernel_sizes:
            out = self.conv_layers['conv_{}'.format(ks)](F.relu(X))
            out = self.maxpool_layers['maxpool_{}'.format(ks)](out.squeeze(-1))
            out = out.view(batch_size, -1)
            all_outs.append(out)
        # Concatenate all outputs from the different conv layers
        X = torch.cat(all_outs, 1)
        # Go through all layers (dropout, fully connected + activation function)
        for l in self.linears:
            X = l(X)    
        # Push through last linear layer
        X = self.out(X)
        # Return log probabilities
        return F.log_softmax(X, dim=1)


    def _calc_conv_output_size(self, seq_len, kernel_size, stride, padding):
        return int(((seq_len - kernel_size + 2*padding) / stride) + 1)

    def _calc_maxpool_output_size(self, seq_len, kernel_size, padding, stride, dilation):
        return int(math.floor( ( (seq_len + 2*padding - dilation*(kernel_size-1) - 1) / stride ) + 1 ))        