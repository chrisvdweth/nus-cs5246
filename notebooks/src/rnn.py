import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnType:
    RNN  = 1
    GRU  = 2
    LSTM = 3


class RnnTextClassifier(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        # We have to memorize this for initializing the hidden state
        self.params = params
        
        # Calculate number of directions
        self.rnn_num_directions = 2 if params.rnn_bidirectional == True else 1
        
        # Calculate scaling factor for first linear (2x the size if attention is used)
        self.scaling_factor = 2 if params.dot_attention == True else 1
        
        #################################################################################
        ### Create layers
        #################################################################################
        
        ## Embedding layer
        self.embedding = nn.Embedding(params.vocab_size, params.embed_size)
        
        ## Recurrent Layer
        if params.rnn_type == RnnType.GRU:
            rnn = nn.GRU
        elif params.rnn_type == RnnType.LSTM:
            rnn = nn.LSTM
        else:
            rnn = nn.RNN
        self.rnn = rnn(params.embed_size,
                       params.rnn_hidden_size,
                       num_layers=params.rnn_num_layers,
                       bidirectional=params.rnn_bidirectional,
                       dropout=params.rnn_dropout,
                       batch_first=True)
        
        ## Linear layers (incl. Dropout and Activation)
        linear_sizes = [params.rnn_hidden_size * self.rnn_num_directions * self.scaling_factor] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(p=params.linear_dropout))
        
        if self.params.dot_attention == True:
            self.attention = DotAttentionClassification()
            
        
        self.out = nn.Linear(linear_sizes[-1], params.output_size)
        
        #################################################################################
        
        
    def forward(self, inputs, hidden):
        
        batch_size, seq_len = inputs.shape

        # Push through embedding layer
        X = self.embedding(inputs)

        # Push through RNN layer
        rnn_outputs, hidden = self.rnn(X, hidden)
        
        # Extract last hidden state
        if self.params.rnn_type == RnnType.LSTM:
            last_hidden = hidden[0].view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)[-1]
        else:
            last_hidden = hidden.view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)[-1]

        # Handle directions
        if self.rnn_num_directions == 1:
            final_hidden = last_hidden.squeeze(0)
        elif self.rnn_num_directions == 2:
            h_1, h_2 = last_hidden[0], last_hidden[1]
            final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states
            
        X = final_hidden
        
        # Push through attention layer
        
        if self.params.dot_attention == True:
            #rnn_outputs = rnn_outputs.permute(1, 0, 2)  #
            X, attention_weights = self.attention(rnn_outputs, final_hidden)
        else:
            X, attention_weights = final_hidden, None
        
        # Push through linear layers (incl. Dropout & Activation layers)
        for l in self.linears:
            X = l(X)

        X = self.out(X)
            
        return F.log_softmax(X, dim=1)

    
    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.LSTM:
            return (torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size),
                    torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size))
        else:
            return torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)
        
        
        
        

class DotAttentionClassification(nn.Module):
    
    def __init__(self):
        super(DotAttentionClassification, self).__init__()

    def forward(self, rnn_outputs, final_hidden_state):
        # Shapes of tensors:
        # rnn_outputs.shape: (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape:  (batch_size, hidden_size)

        # Calculate attention weights                          
        attention_weights = torch.bmm(rnn_outputs, final_hidden_state.unsqueeze(2))
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=1)

        # Calculate context vector
        context = torch.bmm(rnn_outputs.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        
        # Concatenate context vector and final hidden state
        concat_vector = torch.cat((context, final_hidden_state), dim=1)
        
        # Return concatenated vector and attention weights
        return concat_vector, attention_weights        
    
    
    
    
    
class VanillaRnnNER(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params      
        # Define Embedding Layer
        self.embedding = nn.Embedding(self.params.vocab_size_words, self.params.embed_size)
        # Define Bi-LSTM layer
        self.bilstm = nn.LSTM(params.embed_size,
                              params.bilstm_hidden_size,
                              num_layers=params.bilstm_num_layers,
                              dropout=params.bilstm_dropout,
                              bidirectional=True,                              
                              batch_first=True)        
        # Define list of Linear Layers (with activation and dropout)
        self.linears = nn.ModuleList()
        linear_sizes = [params.bilstm_hidden_size] + params.linear_hidden_sizes
        for i in range(len(linear_sizes)-1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            # Add Dropout layer if probality > 0
            if params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=params.linear_dropout))                    
        # Define output layer
        self.out = nn.Linear(linear_sizes[-1], params.vocab_size_label)
        
    def forward(self, X):
        batch_size, seq_len = X.shape
        hidden = self._init_hidden(batch_size)
        # Push through embedding layer
        X = self.embedding(X)
        # Push through Bi-LSTM layer
        outputs, hidden = self.bilstm(X, hidden)
        # Handling forward and backward direction by adding both directions
        outputs = outputs.reshape(batch_size, seq_len, 2, self.params.bilstm_hidden_size)
        outputs = outputs[:,:,0,:] + outputs[:,:,1,:]
        # Push through
        for l in self.linears:
            outputs = l(outputs)
        # Push through output layer and return logits
        return self.out(outputs)
        
    def _init_hidden(self, batch_size):
        return (torch.zeros(self.params.bilstm_num_layers * 2, batch_size, self.params.bilstm_hidden_size).to(self.params.device),
                torch.zeros(self.params.bilstm_num_layers * 2, batch_size, self.params.bilstm_hidden_size).to(self.params.device)       
        )        
        
        

class PosRnnNER(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.embed_words = nn.Embedding(self.params.vocab_size_words, self.params.embed_size_words)
        self.embed_pos = nn.Embedding(self.params.vocab_size_pos, self.params.embed_size_pos)
        
        self.total_embed_size = 0
        if self.params.embed_size_words > 0:
            self.total_embed_size += self.params.embed_size_words
        if self.params.embed_size_pos > 0:
            self.total_embed_size += self.params.embed_size_pos
        
        self.bilstm = nn.LSTM(params.embed_size_words+self.params.embed_size_pos,
                              self.params.bilstm_hidden_size,
                              num_layers=self.params.bilstm_num_layers,
                              dropout=self.params.bilstm_dropout,
                              bidirectional=True,                              
                              batch_first=True)
        
        # Fully connected layers (incl. Dropout and Activation)
        linear_sizes = [params.bilstm_hidden_size] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            # Add Dropout layer if probality > 0
            if params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=params.linear_dropout))            
        
        self.out = nn.Linear(linear_sizes[-1], params.vocab_size_tag)
        
        
    def forward(self, X):
        batch_size, seq_len = X.shape
        hidden = self._init_hidden(batch_size)
        
        # Split input sequence into words section and POS tags section
        X_words, X_pos = torch.split(X, seq_len//2, dim=1)
        
        X_words = self.embed_words(X_words)
        X_pos = self.embed_pos(X_pos)
        
        X = torch.cat([X_words, X_pos], dim=2)
        
        outputs, hidden = self.bilstm(X, hidden)

        # Handling forward and backward direction by adding both directions        
        outputs = outputs.reshape(batch_size, seq_len//2, 2, self.params.bilstm_hidden_size)
        outputs = outputs[:,:,0,:] + outputs[:,:,1,:]
        
        for l in self.linears:
            outputs = l(outputs)
        
        # Return outputs
        return self.out(outputs)
        
        
    def _init_hidden(self, batch_size):
        return (torch.zeros(self.params.bilstm_num_layers * 2, batch_size, self.params.bilstm_hidden_size).to(self.params.device),
                torch.zeros(self.params.bilstm_num_layers * 2, batch_size, self.params.bilstm_hidden_size).to(self.params.device)       
        )        
    