import pytorch_lightning as pl
from transformers import  get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import CHRFScore
import torch 
import os
import torch.nn as nn
import pandas as pd
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Error loading .*: <urlopen error [Errno 111] Connection refused>")



class DoctorPatientDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_input_length = 10000, max_output_length =  2000,is_train=True,is_split=None):
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.is_train = is_train
        self.is_split = is_split
        prompt1 = "The conversation of doctor-patient is about:"
        prompt2 = "Summarize:"
        ## TODO ## 
        # self.inputs = self.data.apply(lambda x: prompt1+x['section_header']+'.'+prompt2+ x['dialogue'], axis=1)
        self.inputs = self.data.apply(lambda x: prompt2+ x['dialogue'], axis=1)
        self.inputs = self.inputs.apply(lambda x: x[:self.max_input_length])
        
        if self.is_train:
            if self.is_split:
                print('TaskB data loading....',self.is_split)
                self.data[self.is_split] = self.data[self.is_split].astype(str)
                self.outputs = self.data.apply(lambda x: x[self.is_split], axis=1)
                self.outputs = self.outputs.apply(lambda x: x[:self.max_output_length])
            else:
                if 'section_text' in self.data.columns:
                    print('TaskA data loading....')
                    self.outputs = self.data.apply(lambda x: x['section_text'], axis=1)
                    self.outputs = self.outputs.apply(lambda x: x[:self.max_output_length])
                    # TODO truncate

                else:
                    print('TaskB or TaskC data loading....')
                    self.outputs = self.data['note'].apply(lambda x: x[:self.max_output_length]) 
        else:
            print("---------------------test-----------------")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.is_train:
            input_sequence = self.inputs[index]
            output_sequence = self.outputs[index]
            return input_sequence, output_sequence
        else:
            input_sequence = self.inputs[index]
            return input_sequence

class Long_Summarizer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.r_factor = 0.5
        # PREPARE DATA
        
        self.train_dataset = DoctorPatientDialogueDataset(self.params.train_file,is_train=True,is_split=self.params.is_split,max_input_length=self.params.max_input_length,max_output_length=self.params.max_output_length)
        self.val_dataset = DoctorPatientDialogueDataset(self.params.val_file, is_train=True,is_split=self.params.is_split,max_input_length=self.params.max_input_length,max_output_length=self.params.max_output_length)
        self.test_dataset = DoctorPatientDialogueDataset(self.params.test_file, is_train=False,is_split=self.params.is_split,max_input_length=self.params.max_input_length,max_output_length=self.params.max_output_length)

        
        pretrained_model_name = self.params.pretrained_model
        self.model = LongT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.rouge = ROUGEScore()
        
        # self.automatic_optimization = False
        # self.learning_rate = learning_rate
        if self.params.chrf_score:
            self.chrf = CHRFScore()
        
        self.warm_up=params.warm
        self.auto_lr = self.params.auto_lr
        
        self.learning_rate = params.learning_rate

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask,labels=decoder_input_ids,output_hidden_states=True,return_dict=True)
        return outputs.loss,outputs.logits

    # def compute_kl_loss(self, p, q, pad_mask=None):
    
    #     p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    #     q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
    #     # pad_mask is for seq-level tasks
    #     if pad_mask is not None:
    #         p_loss.masked_fill_(pad_mask, 0.)
    #         q_loss.masked_fill_(pad_mask, 0.)

    #     # You can choose whether to use function "sum" and "mean" depending on your task
    #     p_loss = p_loss.sum()
    #     q_loss = q_loss.sum()

    #     loss = (p_loss + q_loss) / 2
    #     return loss

    def training_step(self, batch, batch_idx):
        

        input_ids, attention_mask, labels, labels_mask = batch
        loss,logits = self.forward(input_ids, attention_mask,labels)
        if self.params.use_r_drop:
            loss2,logits2 = self.forward(input_ids, attention_mask,labels)
            ce_loss = 0.5*(loss+loss2)
            kl_loss= self.compute_kl_loss(logits, logits2,pad_mask = labels_mask)
            loss = ce_loss + self.r_factor*kl_loss

        
        self.log("train_loss", loss,on_step=True, prog_bar=True)
        return loss
    
    

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, labels_mask = batch
        loss,_ = self.forward(input_ids, attention_mask,labels)
        # loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        # Add the ROUGE evaluation
        
        if self.params.rouge_score:
            # outputs = self.model.generate(input_ids, attention_mask=attention_mask,top_p=0.95,min_length=100,max_length=4000,top_k=100,repetition_penalty=2.0)
            outputs = self.model.generate(input_ids, attention_mask=attention_mask,top_p=0.95,min_length=1,max_new_tokens=self.params.max_output_length,top_k=100,repetition_penalty=3.0)
            generated_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_summaries = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print("generated_summaries:",generated_summaries[:100])
            # print("target_summaries:",target_summaries[:100])
            rouge_score = self.rouge(generated_summaries, target_summaries)
            self.log("rouge_score", rouge_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)
            # self.log("rougeLsum_f", rouge_score['rougeLsum_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
            # self.log("rouge1_f", rouge_score['rouge1_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
            # self.log("rouge2_f", rouge_score['rouge2_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
            return {'val_loss': loss, "rougeLsum_fmeasure":rouge_score['rougeLsum_fmeasure'],"rouge1_f":rouge_score['rouge1_fmeasure'],"rouge2_f":rouge_score['rouge2_fmeasure']}
        if self.params.chrf_score:
            chrf_score = self.chrf(generated_summaries, target_summaries)
            self.log("chrf_score", chrf_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)
            return {'val_loss': loss, "chrf_score":chrf_score, "rougeLsum_fmeasure":rouge_score['rougeLsum_fmeasure'],"rouge1_f":rouge_score['rouge1_fmeasure'],"rouge2_f":rouge_score['rouge2_fmeasure']}
        
        
        

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        avg_rouge1 = torch.stack([x['rouge1_f'] for x in outputs]).mean()
        self.log("avg_rouge_1", avg_rouge1,sync_dist=True,prog_bar=True)
        avg_rouge2 = torch.stack([x['rouge2_f'] for x in outputs]).mean()
        self.log("avg_rouge_2", avg_rouge2,sync_dist=True,prog_bar=True)
        avg_rougeLsum_fmeasure = torch.stack([x['rougeLsum_fmeasure'] for x in outputs]).mean()
        self.log("avg_rougeLsum_fmeasure", avg_rougeLsum_fmeasure,sync_dist=True,prog_bar=True)
        # return {'avg_val_loss': avg_loss}
        print("avg_rouge_1", avg_rouge1,"avg_rouge_2", avg_rouge2,"avg_rougeLsum_fmeasure", avg_rougeLsum_fmeasure)
    
    def predict_step(self, batch, batch_idx) :
        input_ids, attention_mask = batch
        
        outputs = self.model.generate(input_ids, attention_mask=attention_mask,top_p=0.95,min_length=1,max_new_tokens=self.params.max_output_length,top_k=100,repetition_penalty=3.0)
        
        # print(outputs)
        
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(preds)
        return preds
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn_test,shuffle=False,num_workers=8)
    
    
    def train_dataloader(self):
        dataload= torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=True,num_workers=8)
        self.total_train_batches = len(dataload)
        return dataload
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=False,num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn_test,shuffle=False,num_workers=8)
    

    def collate_fn_test(self, batch):
        input_sequences = [x for x in batch]
        # print(input_sequences)
        input_ids = self.tokenizer.batch_encode_plus(input_sequences, padding='longest', return_tensors='pt')['input_ids']
        attention_mask = input_ids.ne(0).long()
        return input_ids, attention_mask

    def collate_fn(self, batch):

        ## 数据增强 ##
        input_sequences = [x[0] for x in batch]
        output_sequences = [x[1] for x in batch]
        input_ids = self.tokenizer.batch_encode_plus(input_sequences, padding='longest', return_tensors='pt')['input_ids']
        decoder_input_ids = self.tokenizer.batch_encode_plus(output_sequences, padding='longest', return_tensors='pt')['input_ids']

        attention_mask = input_ids.ne(0).long()
        decoder_attention_mask = decoder_input_ids.ne(0).long()
        
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

    def configure_optimizers(self):
        #optimizer = optim.Adafactor(self.model.parameters(),lr= self.learning_rate)
        
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.learning_rate)


        
        
        if self.warm_up:

            # scheduler = get_cosine_schedule_with_warmup(
            # optimizer,  num_training_steps=self.params.epochs, num_warmup_steps=5
            # )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,  num_training_steps=self.params.epochs, num_warmup_steps=5
            )
            return {'optimizer': optimizer,'lr_scheduler': scheduler}

            
        else:
            return optimizer
    
        

