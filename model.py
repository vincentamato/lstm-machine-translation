import torch
import torch.nn as nn
import random
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        hidden = self.fc(hidden)
        
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        cell = self.fc(cell)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        
        self.rnn = nn.LSTM(embedding_size, hidden_size * 2, num_layers, dropout=0, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        x = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = trg[:, t] if teacher_force else top1

        return outputs

def load_glove_embeddings(vocab, embedding_dim):
    glove = GloVe(name='6B', dim=embedding_dim)
    embeddings = torch.zeros(len(vocab), embedding_dim)
    for i in range(len(vocab)):
        token = vocab.lookup_token(i)
        if token in glove.stoi:
            embeddings[i] = glove[token]
        else:
            embeddings[i] = torch.randn(embedding_dim)
    return embeddings

def create_seq2seq_model(src_vocab, trg_vocab, embedding_dim, hidden_size, num_layers, dropout, device):
    src_embeddings = load_glove_embeddings(src_vocab, embedding_dim)
    trg_embeddings = load_glove_embeddings(trg_vocab, embedding_dim)

    encoder = Encoder(len(src_vocab), embedding_dim, hidden_size, num_layers, dropout, src_embeddings)
    decoder = Decoder(len(trg_vocab), embedding_dim, hidden_size, len(trg_vocab), num_layers, dropout, trg_embeddings)    
    model = Seq2Seq(encoder, decoder, device)
    
    # Initialize parameters
    for name, param in model.named_parameters():
        if 'weight' in name and 'embedding' not in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model

def summarize_model(model, src_vocab_size, trg_vocab_size):
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_params = encoder_params + decoder_params
    
    print(f"{'=' * 50}")
    print(f"Model Summary:")
    print(f"{'=' * 50}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {trg_vocab_size}")
    print(f"{'=' * 50}")