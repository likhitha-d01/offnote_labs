import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
import pytorch_lightning as pl
from easydict import EasyDict as ED

class TranslationModel(pl.LightningModule):

    def __init__(self, encoder, decoder):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits

    def save(self, tokenizers, output_dirs):
   
        save_model(self.encoder, output_dirs.encoder)
        save_model(self.decoder, output_dirs.decoder)

    def flat_accuracy(preds, labels):

        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        print (f'preds: {pred_flat}')
        print (f'labels: {labels_flat}')

        return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)

    def training_step(self, batch,batch_idx):  
        source = batch[0].cuda()
        target = batch[1].cuda()

        loss, logits = self(source, target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        self.logger.log_metrics({'loss':loss},self.current_epoch+1)
    
        return {'loss':loss}

    """
    def validation_step(self, batch,batch_idx):  
        source = batch[0]
        target = batch[1]

        loss, logits = self(source, target)

        return {'val_loss':loss}

    """
    def test_step(self,batch,batch_idx):
        source = batch[0]
        target = batch[1]

        loss, logits = self(source, target)

        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()


        acc = self.flat_accuracy(logits, label_ids)

        return {'test_acc':acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def build_model(config):
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'
    
    #hidden_size and intermediate_size are both wrt all the attention heads. 
    #Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    #Create encoder and decoder embedding layers.
    encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    encoder.set_input_embeddings(encoder_embeddings.cuda())
    
    decoder = BertForMaskedLM(decoder_config)
    decoder.set_input_embeddings(decoder_embeddings.cuda())

    model = TranslationModel(encoder, decoder)
    model.cuda()
    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    return model, tokenizers
