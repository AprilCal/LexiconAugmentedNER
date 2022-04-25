# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crf import CRF
from model.layers import NERmodel
from transformers.modeling_bert import BertModel

class GazLSTM(nn.Module):
    def __init__(self, data):
        super(GazLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert

        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim)

        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))


        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))
        if self.use_biword:
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
            else:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), self.word_emb_dim)))


        # char_feature_dim = self.word_emb_dim + 4*self.gaz_emb_dim
        char_feature_dim = self.word_emb_dim
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        ## lstm model
        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden, num_layer=self.lstm_layer, biflag=self.bilstm_flag)

        ## cnn model
        if self.model_type == 'cnn':
            self.NERmodel = NERmodel(model_type='cnn', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout, gpu=self.gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer', input_dim=char_feature_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer, dropout=data.HP_dropout)

        self.drop = nn.Dropout(p=data.HP_dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size+2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.NERmodel = self.NERmodel.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

    def get_tags(self,gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []

        word_embs = self.word_embedding(word_inputs)
        # print(word_embs)
        # print(word_embs.size())

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs,biword_embs],dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)   #(b,l,we)
        else:
            word_inputs_d = word_embs

        # print(word_inputs_d.size())

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1,1,1,1,1,self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  #(b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:  #use gaz embedding
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.gaz_emb_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  #(b,l,4,g,ge)  ge:gaz_embed_dim
            # print(gaz_embeds)

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  #(b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  #(b,l,1,1)

            weights = gaz_count.div(count_sum)  #(b,l,4,g)
            weights = weights*4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights*gaz_embeds  #(b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  #(b,l,4,e)

        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  #(b,l,4,ge)/(b,l,4,1)

        gaz_embeds_cat = gaz_embeds.view(batch_size,seq_len,-1)  #(b,l,4*ge)
        # word_input_cat = torch.cat([word_inputs_d, gaz_embeds_cat], dim=-1)  #(b,l,we+4*ge)
        word_input_cat = word_inputs_d

        # print(word_input_cat.size())

        ### cat bert feature
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:,1:-1,:]
            word_input_cat = torch.cat([word_input_cat, outputs], dim=-1)

        feature_out_d = self.NERmodel(word_input_cat)
        tags = self.hidden2tag(feature_out_d)
        return tags, gaz_match

    def contrastive(self, gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask):
        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs,biword_embs],dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)   #(b,l,we)
        else:
            word_inputs_d = word_embs

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1,1,1,1,1,self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  #(b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:  #use gaz embedding
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.gaz_emb_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  #(b,l,4,g,ge)  ge:gaz_embed_dim
            # print(gaz_embeds)

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  #(b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  #(b,l,1,1)

            weights = gaz_count.div(count_sum)  #(b,l,4,g)
            weights = weights*4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights*gaz_embeds  #(b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  #(b,l,4,e)
        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  #(b,l,4,ge)/(b,l,4,1)


        rand_batch = np.random.randint(0,batch_size)
        neg_batch = np.random.randint(0,batch_size)
        while neg_batch == rand_batch:
            neg_batch = np.random.randint(0,batch_size)
        # print(rand_batch, neg_batch)
        bmes_samples = []
        for i in range(4):
            bmes_samples.append(gaz_embeds[rand_batch,:,i,:].unsqueeze(0))

        anchor_sample = word_inputs_d[rand_batch,:,:].unsqueeze(0)
        negative_sample = word_inputs_d[neg_batch,:,:].unsqueeze(0)

        bmes_tags = []
        for i in range(4):
            feature_out_b = self.NERmodel(bmes_samples[i])
            b_tags = self.hidden2tag(feature_out_b)
            bmes_tags.append(b_tags)

        feature_out_anchor = self.NERmodel(anchor_sample)
        anchor_tags = self.hidden2tag(feature_out_anchor)
        
        feature_out_neg = self.NERmodel(negative_sample)
        neg_tags = self.hidden2tag(feature_out_neg)

        for i in range(4):
            bmes_tags[i] = torch.mean(bmes_tags[i].squeeze(0),0)
        anchor_tags = torch.mean(anchor_tags.squeeze(0),0)
        neg_tags = torch.mean(neg_tags.squeeze(0),0)
        # print(b_tags.size(),anchor_tags.size(),neg_tags.size())
        
        loss_pos = 0
        for i in range(4):
            loss_pos += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                anchor_tags.view(1, 1, 20),
                bmes_tags[i].view(1, 20, 1)
            )))

        loss_neg = -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
            anchor_tags.view(1, 1, 20),
            neg_tags.view(1, 20, 1)
        )))
        
        return (loss_pos+loss_neg)

    def contrastive2(self, gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask):
        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs,biword_embs],dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)   #(b,l,we)
        else:
            word_inputs_d = word_embs

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1,1,1,1,1,self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  #(b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:  #use gaz embedding
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1,1,1,1,self.gaz_emb_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  #(b,l,4,g,ge)  ge:gaz_embed_dim
            # print(gaz_embeds)

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  #(b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  #(b,l,1,1)

            weights = gaz_count.div(count_sum)  #(b,l,4,g)
            weights = weights*4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights*gaz_embeds  #(b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  #(b,l,4,e)
        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  #(b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  #(b,l,4,ge)/(b,l,4,1)

        anchor_tag_list = []
        loss_pos=0
        for pos_idx in range(batch_size):
            anchor_sample = word_inputs_d[pos_idx,:,:].unsqueeze(0)
            # anchor_sample_list.append(anchor_sample)
            bmes_samples = []
            for i in range(3):
                bmes_samples.append(gaz_embeds[pos_idx,:,i,:].unsqueeze(0))
        
            bmes_tags = []
            for i in range(3):
                feature_out_b = self.NERmodel(bmes_samples[i])
                b_tags = self.hidden2tag(feature_out_b)
                bmes_tags.append(b_tags)

            for i in range(3):
                bmes_tags[i] = torch.mean(bmes_tags[i].squeeze(0),0)

            feature_out_anchor = self.NERmodel(anchor_sample)
            anchor_tags = self.hidden2tag(feature_out_anchor)
            anchor_tags = torch.mean(anchor_tags.squeeze(0),0)
            # print(anchor_tags.shape)
            anchor_tag_list.append(anchor_tags)
        
            for i in range(3):
                loss_pos += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                anchor_tags.view(1, 1, 20),
                bmes_tags[i].view(1, 20, 1)
                )))

        loss_neg = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if j>=i:
                    continue
                loss_neg += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                    anchor_tag_list[i].view(1, 1, 20),
                    anchor_tag_list[j].view(1, 20, 1)
                )))
        
        return (loss_pos/(batch_size*4)+2*loss_neg/(batch_size*(batch_size-1)))/100
        
    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask):
        tags, _ = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)
        contrastive_loss = self.contrastive(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)
        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)
        return total_loss+contrastive_loss, tag_seq


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,layer_gaz, gaz_count,gaz_chars, gaz_mask,gazchar_mask, mask, batch_bert, bert_mask):
        tags, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)
        return tag_seq, gaz_match