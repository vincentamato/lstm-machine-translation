import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
import spacy
from model import Encoder, Decoder, Seq2Seq
from data import SRC, TRG, get_vocab_sizes

# Load spacy language models
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

# Tokenizers
tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenize_fr = get_tokenizer('spacy', language='fr_core_news_sm')

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (make sure these match your training script)
ENC_EMB_SIZE = 256
DEC_EMB_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

def load_model():
    # Get vocabulary sizes
    INPUT_SIZE_ENC, INPUT_SIZE_DEC = get_vocab_sizes()
    OUTPUT_SIZE = INPUT_SIZE_DEC

    # Instantiate the model
    encoder = Encoder(INPUT_SIZE_ENC, ENC_EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS, ENC_DROPOUT)
    decoder = Decoder(INPUT_SIZE_DEC, DEC_EMB_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    return model

def translate_sentence(model, sentence, max_length=50):
    model.eval()
    
    # Tokenize the sentence
    tokens = tokenize_en(sentence)
    print(f"Tokenized input: {tokens}")
    
    # Convert tokens to indices
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [SRC[token] for token in tokens]
    print(f"Source indices: {src_indexes}")
    
    # Convert to tensor and move to device
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    # Get encoder outputs
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    # Start with '<sos>' token
    trg_indexes = [TRG['<sos>']]
    
    for i in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == TRG['<eos>']:
            break
    
    trg_tokens = [TRG.get_itos()[i] for i in trg_indexes]
    print(f"Target indices: {trg_indexes}")
    print(f"Target tokens: {trg_tokens}")
    
    return trg_tokens[1:-1]  # Remove <sos> and <eos>

def main():
    model = load_model()
    
    while True:
        sentence = input("Enter an English sentence (or 'q' to quit): ")
        if sentence.lower() == 'q':
            break
        
        translation = translate_sentence(model, sentence)
        print("Translation:", ' '.join(translation))
        print()

if __name__ == "__main__":
    main()