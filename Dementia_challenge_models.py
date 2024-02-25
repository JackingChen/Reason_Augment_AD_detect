import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime
from torch.utils.data.sampler import WeightedRandomSampler

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, LinearLR ,CosineAnnealingLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.metrics import mean_absolute_error,r2_score,median_absolute_error,mean_squared_error,max_error,explained_variance_score
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertConfig, BertModel,XLMTokenizer, XLMModel
from transformers import AutoTokenizer, AutoModel
from prompts import assesmentPrompt_template, Instruction_templates, Psychology_template,\
    Sensitive_replace_dict, generate_psychology_prompt
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from addict import Dict
import librosa
from sklearn import preprocessing
No_PAR_people=['S009', 'S011', 'S017', 'S027', 'S029', 'S052', 'S080', 'S084', 'S086', 'S118', 'S124', 'S138', 'S145']

def regression_report(y_true, y_pred):
    
    # error = y_true - y_pred
    # percentil = [5,25,50,75,95]
    # percentil_value = np.percentile(error, percentil)
    
    metrics_dict = {
        'mean absolute error': mean_absolute_error(y_true, y_pred),
        'median absolute error': median_absolute_error(y_true, y_pred),
        'mean squared error': mean_squared_error(y_true, y_pred),
        'max error': max_error(y_true, y_pred),
        'r2 score': r2_score(y_true, y_pred),
        'explained variance score': explained_variance_score(y_true, y_pred)
    }

    return metrics_dict

class BertPooler(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
Embsize_map={}
Embsize_map['t_hidden_size']={'xlm':1280,'mbert':768,'Semb':1536,'Embedding':350}

Audio_pretrain=set(['en','gr','multi','wv'])
xlm_mlm_100_1280=set(['xlm_sentence','xlm_session'])
mbert_series=set(['mbert_sentence','mbert_session',])
bert_library=set(["bert-base-uncased",
                "xlm-roberta-base",
                "albert-base-v1",
                "xlnet-base-cased",
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.2",
                "YituTech/conv-bert-base",])
Text_Summary=set(['anomia'])

Text_pretrain = xlm_mlm_100_1280 | mbert_series | bert_library


# Define common settings
Summary_Embeddings_common_settings = {
    'inp_hidden_size': 1536,
}
file_in_pattern = '/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-{attribute_name}_aug'

# List of Summary_Embeddings
Summary_Embeddings_cols = ['max_emb', 'mean_emb', 'Summary', 'max_emb_Summary', 'Max(max_emb,Summary)','mean_emb_Summary']
Psych_attributes = ['psych_ver1','Positive_attributes','psych_ver_1.1','psych_ver_1.1_highcorr','psych_ver_1.1_highcorr2','psych_ver_1.1Fusion','psych_ver_1.1_catpool','psych_ver_2.1','psych_ver_3','psych_ver_3.1.1','psych_ver_3.1.2','RAG3']

Text_Summary_Embedding=set([f'{attribute_name}__{emb_col}' for attribute_name in Psych_attributes for emb_col in Summary_Embeddings_cols])

Model_settings_dict=Dict()
# Populate Model_settings_dict using a loop
for attribute_name in Psych_attributes:
    for emb_col in Summary_Embeddings_cols:
        key = f'{attribute_name}__{emb_col}'
        value = {'inp_col_name': emb_col, 'file_in': file_in_pattern.format(attribute_name=attribute_name), **Summary_Embeddings_common_settings}
        Model_settings_dict[key] = value
for feat in ['max_emb_Summary','mean_emb_Summary']:
    Model_settings_dict[f'psych_ver_1.1Fusion__{feat}']['inp_hidden_size']=1536+1536
    Model_settings_dict[f'psych_ver_1.1_catpool__{feat}']['inp_hidden_size']=1536+1536

# ,'Semb','Embedding'
Model_settings_dict['mbert_sentence']={
    'inp_col_name': 'text',
    'inp_hidden_size': 768,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['mbert_session']={
    'inp_col_name': 'text',
    'inp_hidden_size': 768,
    'file_in':"/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia",
}
Model_settings_dict['xlm_sentence']={
    'inp_col_name': 'text',
    'inp_hidden_size': 1280,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['xlm_session']={
    'inp_col_name': 'text',
    'inp_hidden_size': 1280,
    'file_in':"/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia",
}
Model_settings_dict['en']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['gr']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['multi']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['wv']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['anomia']={
    'inp_col_name': 'Psych_Summary',
    'inp_hidden_size': Embsize_map['t_hidden_size']['mbert'], # use mbert to tokenize and model it
    'file_in':'/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia',
}

model_settings_dict = {
    'bert-base-uncased': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'xlm-roberta-base': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'albert-base-v1': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'xlnet-base-cased': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'emilyalsentzer/Bio_ClinicalBERT': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'dmis-lab/biobert-base-cased-v1.2': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'YituTech/conv-bert-base': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
}

Model_settings_dict.update(model_settings_dict)


def check_keys_matching(*sets, model_settings_dict):
    # Combine all input sets using union
    all_pretrain = set().union(*sets)
    
    # Get the keys from Model_settings_dict
    model_keys = set(model_settings_dict.keys())
    
    # Check if all keys in Model_settings_dict match the combined pretrain set
    return model_keys == all_pretrain

# Example usage
assert check_keys_matching(Audio_pretrain, Text_pretrain, Text_Summary,Text_Summary_Embedding, model_settings_dict=Model_settings_dict)



class ModelArg:
    version = 1
    # data
    epochs: int = 5  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    batch_size: int = 8
    # batch_size: int = 1

class SingleForwardModel(LightningModule):
    # """
    # Is the model of 0207_DM_SentenceLvl1input.py
    # """
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.mdlArg = args.mdlArg
        self.args=args
        self.config = config
        self.batch_size = self.mdlArg.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'dementia_labels'
        self.label_names = ['Control','ProbableAD']
        self.num_labels = 2
        
        
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.test_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch

        self.ID_col='ID   '
        self.le=preprocessing.LabelEncoder()
        
        self.sampler=None
        self.stage=1
        self.No_PAR_people=No_PAR_people
        
        self.label_Calibrate=False
        # Variables that may vary
        # self.inpArg = args.inpArg
        # self.inp_embed_type = self.config['inp_embed']
        # self.inp_col_name = self.inpArg.inp_col_name
        # self.inp_hidden_size = self.inpArg.inp_hidden_size
        # self.hidden = int(self.inpArg.linear_hidden_size)
        # self.inp_tokenizer, self.inp_model, self.pooler=self._setup_embedding(self.inp_embed_type, self.inp_hidden_size)
        # self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        # self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)
    def freeze_model_weight(self,model):
        for param in model.parameters():
            param.requires_grad = False
        print(f'setting inp model {model} freezed')
    def unfreeze_model_weight(self,model):
        for param in model.parameters():
            param.requires_grad = True
        print(f'setting inp model {model} unfreezed')
    def _setup_embedding(self,inp_embed_type, inp_hidden_size):
        if inp_embed_type == "en":
            inp_pretrained = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "gr":
            inp_pretrained = "lighteternal/wav2vec2-large-xlsr-53-greek"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "multi":
            inp_pretrained = "voidful/wav2vec2-xlsr-multilingual-56"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "wv":
            inp_pretrained = 'facebook/wav2vec2-base'
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type in mbert_series:
            inp_pretrained = 'bert-base-multilingual-uncased'
            inp_tokenizer = BertTokenizer.from_pretrained(inp_pretrained)
            inp_model = BertModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in xlm_mlm_100_1280:
            inp_pretrained = 'xlm-mlm-100-1280'
            inp_tokenizer = XLMTokenizer.from_pretrained(inp_pretrained)
            inp_model = XLMModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in Text_Summary:
            inp_pretrained = 'bert-base-multilingual-uncased'
            inp_tokenizer = BertTokenizer.from_pretrained(inp_pretrained)
            inp_model = BertModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in bert_library:
            inp_pretrained = inp_embed_type
            inp_tokenizer = AutoTokenizer.from_pretrained(inp_pretrained)
            inp_model = AutoModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in Text_Summary_Embedding:   
            inp_tokenizer = None
            inp_model = None
        else:
            raise ValueError(f"{inp_embed_type} seems not in Model_settings_dict")
        pooler = BertPooler(inp_hidden_size)
        return inp_tokenizer, inp_model, pooler
    def _get_embedding(self,inp,inp_embed_type, inp_model, pooler):
        if inp_embed_type in mbert_series:
            out = inp_model(inp)[1]
        elif inp_embed_type in xlm_mlm_100_1280:
            out = inp_model(inp)[0]
            out = pooler(out)
        elif inp_embed_type in bert_library:
            if inp_embed_type=='xlnet-base-cased':
                out = inp_model(inp)[0]
                out = pooler(out)
            else:
                out = inp_model(inp)[1]
        elif inp_embed_type in Audio_pretrain:
            out = inp_model(inp)['extract_features']  # [2] # last_hidden_state, feature extraction
            out = out[:, 0, :]
        elif inp_embed_type in Text_Summary:
            out = inp_model(inp)[1]
        elif inp_embed_type in Text_Summary_Embedding:  
            out=inp
        else:
            raise ValueError("Invalid inp_embed_type specified.")
        
        return out
    def _get_embedding_seq(self,inp,inp_embed_type, inp_model):
        if inp_embed_type in mbert_series:
            out = inp_model(inp)[0]
        elif inp_embed_type in xlm_mlm_100_1280:
            out = inp_model(inp)[0]
        elif inp_embed_type in bert_library:
            if inp_embed_type=='xlnet-base-cased':
                out = inp_model(inp)[0]
            else:
                out = inp_model(inp)[0]
        elif inp_embed_type in Audio_pretrain:
            out = inp_model(inp)['extract_features']  # [2] # last_hidden_state, feature extraction
            # out = out[:, 0, :]
        elif inp_embed_type in Text_Summary:
            out = inp_model(inp)[0]
        else:
            raise ValueError("Invalid inp_embed_type specified.")
        return out
    def forward(self, inp):
        output = self._get_embedding(inp,self.inp_embed_type, self.inp_model, self.pooler)
        
        logits = self.clf2(self.clf1(output))
    
        return logits
    def configure_optimizers(self):
        if self.stage==1:
            return self._configure_optim_stage1()
        elif self.stage==2:
            return self._configure_optim_stage2()
        else:
            raise ValueError()

    def _configure_optim_stage1(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler_dict={
            'exp':ExponentialLR(optimizer, gamma=0.5),
            'lin':LinearLR(optimizer),
            'cos':CosineAnnealingLR(optimizer,T_max=5)
        }
        scheduler = scheduler_dict[self.config['lr_scheduler']]
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    def _configure_optim_stage2(self):
        # return optimizers and scheduler for fine-tine
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5)
        scheduler_dict={
            'exp':ExponentialLR(optimizer, gamma=0.5),
            'lin':LinearLR(optimizer),
            'cos':CosineAnnealingLR(optimizer,T_max=5)
        }
        scheduler = scheduler_dict[self.config['lr_scheduler']]
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    def _Tokenize(self, df, inp_embed_type, inp_col_name, inp_tokenizer,tg_sr=16000):
        if inp_embed_type in Text_pretrain:
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer.encode(
                str(x),
                padding='max_length',
                max_length=self.mdlArg.max_length,
                truncation=True,
            ))
        elif inp_embed_type in Text_Summary:
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer.encode(
                str(x),
                padding='max_length',
                max_length=self.mdlArg.max_length,
                truncation=True,
            ))
        elif inp_embed_type in Audio_pretrain:
            audio_root = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips"
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer(
                librosa.load(f"{audio_root}/{x}")[0],
                padding='max_length',
                sampling_rate=tg_sr,
                max_length=100000,
                truncation=True
            )['input_values'][0])
        elif inp_embed_type in Text_Summary_Embedding:   
            a=1 #No operation 
        else:
            raise ValueError("Invalid inp_embed_type specified.")
        return df
    def preprocess_dataframe(self):
        #.iloc[:40]
        df_train = pd.read_csv(f"{self.inpArg.file_in}/train.csv")
        df_dev = pd.read_csv(f"{self.inpArg.file_in}/dev.csv")
        df_test = pd.read_csv(f"{self.inpArg.file_in}/test.csv")
        self.df_train=self._Tokenize(df_train, self.inp_embed_type, self.inpArg.inp_col_name,self.inp_tokenizer)
        self.df_dev=self._Tokenize(df_dev, self.inp_embed_type,self.inpArg.inp_col_name,self.inp_tokenizer)
        self.df_test=self._Tokenize(df_test, self.inp_embed_type,self.inpArg.inp_col_name,self.inp_tokenizer)

        concatenated_column = pd.concat([self.df_train[self.ID_col], self.df_dev[self.ID_col], self.df_test[self.ID_col]], ignore_index=True)        
        self.le.fit(concatenated_column)
        
        print(f'# of train:{len(df_train)}, val:{len(df_dev)}, test:{len(df_test)}')
        self._df2Dataset()
        if 'samplebalance' in self.config and self.config['samplebalance']==True:
            self._buid_BalancedSampler()
        self.labels_train_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_train,tag='train')
        self.labels_dev_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_dev,tag='dev')
        self.labels_test_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_test,tag='test')
    def _Get_person_Lvl_Groundtruth_label(self,df_data, tag=""):
        # Outputs: padded_data, labels_all_df
        # the ID of padded_data will be synced with labels_all_df
        # Group sentence level data to person level
        df_data[['st', 'ed']] = df_data['path'].apply(lambda x: pd.Series(x.replace(".wav","").split("_")[-2:]))
        df_data[['role']] = df_data['path'].apply(lambda x: pd.Series(x.replace(".wav","").split("_")[1]))
        grouped_dftrain = df_data.groupby('ID   ')

        dataframes_dict = {name:group for name, group in grouped_dftrain}
        # Output: -> Dict: dataframes_dict

        def are_all_elements_same(lst):
            return all(x == lst[0] for x in lst)

        # Create session level inputs and pad
        labels_all_df=pd.DataFrame()
        for key in dataframes_dict.keys():
            if dataframes_dict[key]['role'].str.contains('PAR').any():
                txt_all=[]
                label_all=[]
                for i,row in dataframes_dict[key].iterrows():
                    if row['role'] == 'PAR':
                        txt_all.append(row['text'])
                        label_all.append(row['dementia_labels'])
                assert are_all_elements_same(label_all)

                labels_all_df.loc[key,'dementia_labels']=int(label_all[0])
            else:
                print(f"Session {tag}_{key} have no PAR")
                txt_all=[]
                label_all=[]
                for i,row in dataframes_dict[key].iterrows():
                    txt_all.append(row['text'])
                labels_all_df.loc[key,'dementia_labels']=int(0)

        return labels_all_df, list(labels_all_df.index)
    def _df2Dataset(self):
        def DecideDtype(inp_embed_type):
            if inp_embed_type in Text_pretrain:
                dtype=torch.long
            elif inp_embed_type in Audio_pretrain:
                dtype=torch.float
            return dtype
        dtype=DecideDtype(self.inp_embed_type)
        self.train_data = TensorDataset(
            torch.tensor(self.df_train[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_train[self.label_cols].tolist(), dtype=torch.long),
            torch.tensor(self.le.transform(self.df_train[self.ID_col]).tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(self.df_dev[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_dev[self.label_cols].tolist(), dtype=torch.long),
            torch.tensor(self.le.transform(self.df_dev[self.ID_col]).tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(self.df_test[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_test[self.label_cols].tolist(), dtype=torch.long),
             torch.tensor(self.df_test.index.tolist(), dtype=torch.long),
             torch.tensor(self.le.transform(self.df_test[self.ID_col]).tolist(), dtype=torch.long),
        )
    def _buid_BalancedSampler(self):
        train_labels=torch.tensor(self.df_train[self.label_cols].tolist(), dtype=torch.long)
        # Calculate class weights
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts

        # Assign weights to each sample based on its class
        sample_weights = class_weights[train_labels]

        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"The sampler is initiated as:\n WeightedRandomSampler(weights={sample_weights}, num_samples={len(sample_weights)}, replacement=True)")

    # def train_dataloader(self):
    #     print(f"Loading batch size: {self.batch_size}")
    #     if self.sampler is not None:
    #         return DataLoader(
    #         self.train_data,
    #         batch_size=self.batch_size,
    #         sampler=self.sampler,
    #         num_workers=self.mdlArg.cpu_workers,
    #     )
    #     else:
    #         return DataLoader(
    #             self.train_data,
    #             batch_size=self.batch_size,
    #             shuffle=True,
    #             num_workers=self.mdlArg.cpu_workers,
    #         )
    def train_dataloader(self):
        print(f"Loading batch size: {self.batch_size}")
        if self.sampler is not None:
            return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.mdlArg.cpu_workers,
        )
        else:
            return DataLoader(
                self.train_data,
                batch_size=3,
                shuffle=True,
                num_workers=3,
            )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mdlArg.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mdlArg.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        # token, audio, labels = batch  
        # token,  labels = batch  
        token,  labels, ID = batch  
        # logits = self(token, audio) 
        logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def Maj_vote(self,df_data, clses)->pd.DataFrame:
        # Count the number of 1's and 0's for each ID
        counts_all = pd.DataFrame()
        counts_tmp = pd.DataFrame([],columns=clses)

        # Count the number of 1's and 0's for each ID in y_pred
        counts_pred = df_data.groupby('ID')['y_pred'].value_counts().unstack().fillna(0)
        counts_tmp[counts_pred.columns]=counts_pred
        counts_tmp=counts_tmp.fillna(0)
        counts_all['y_pred'] = counts_tmp[clses].idxmax(axis=1).values

        # Count the number of 1's and 0's for each ID in y_true
        counts_true = df_data.groupby('ID')['y_true'].value_counts().unstack().fillna(0)
        counts_all['y_true'] = counts_true[clses].idxmax(axis=1).values

        assert (counts_true.index == counts_pred.index).all()
        counts_all.index=counts_pred.index
        # pred	true
        # ID		
        # S169	1	0
        # S173	0	1
        # S176	1	1
        # S178	1	1
        # S183	0	1
        # S194	1	0
        return counts_all
    
    def validation_step(self, batch, batch_idx):
        # token, audio, labels = batch  
        token, labels, ID = batch  
        # logits = self(token, audio) 
        logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)     
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
        ID=list(self.le.inverse_transform(ID.cpu().numpy()))

        # --> HERE STEP 2 <--
        self.val_step_outputs.append({
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
            'ID': ID,
        })
        # self.val_step_targets.append(y_true)
        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }
        
            
    def test_step(self, batch, batch_idx):
        # token, audio, labels,id_ = batch 
        token, labels,id_, ID = batch 
        # print('id', id_)
        # logits = self(token, audio) 
        logits = self(token) 
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
        ID=list(self.le.inverse_transform(ID.cpu().numpy()))

        # --> HERE STEP 2 <--
        self.test_step_outputs.append({
            'y_true': y_true,
            'y_pred': y_pred,
            'ID': ID,
        })
        # self.test_step_targets.append(y_true)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
        }
    
    def on_validation_epoch_end(self):
        loss = torch.tensor(0, dtype=torch.float)
        # print("Value= ",self.val_step_outputs)
        # print("type(self.val_step_outputs)=",type(self.val_step_outputs))
        # print("type(self.val_step_outputs[0])=",type(self.val_step_outputs[0]))
        # print("type(self.val_step_outputs[0] loss)=",type(self.val_step_outputs[0]['loss']))
        for i in self.val_step_outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(self.val_step_outputs)
        loss = float(_loss)
        y_true = []
        y_pred = []
        y_id = []
        for i in self.val_step_outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            y_id   += i['ID']
        df_data=pd.DataFrame([y_true,y_pred,y_id],index=['y_true','y_pred','ID']).T
        clses=list(set(y_true))
        df_Mj_vot=self.Maj_vote(df_data,clses)
        if self.label_Calibrate==True:
            df_Mj_vot['y_true']=self.labels_dev_df['dementia_labels']
            df_Mj_vot=df_Mj_vot.drop(self.No_PAR_people)
        y_pred, y_true, y_id = df_Mj_vot['y_pred'], df_Mj_vot['y_true'], df_Mj_vot.index
        
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        pred_dict['ID']= y_id
        
        val_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        
        self.log("val_acc", val_acc)
        # print("y_pred= ", y_pred)
        # print('\n\n\n')
        # print("y_true= ", y_true)
        # print('\n\n\n')
        print("-------val_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        pprint(df_result)
        

        # df_result.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_val.csv')

        # pred_df = pd.DataFrame(pred_dict)
        # pred_df.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_val_pred.csv')
        self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_val')
        self.val_step_outputs.clear()
        # self.val_step_targets.clear()
        return {'loss': _loss}

    def on_test_epoch_end(self):

        y_true = []
        y_pred = []
        y_id = []
        
        for i in self.test_step_outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            y_id   += i['ID']
        df_data=pd.DataFrame([y_true,y_pred,y_id],index=['y_true','y_pred','ID']).T
        clses=list(set(y_true))
        df_Mj_vot=self.Maj_vote(df_data,clses)
        if self.label_Calibrate==True:
            df_Mj_vot['y_true']=self.labels_test_df['dementia_labels']
        y_pred, y_true, y_id = df_Mj_vot['y_pred'], df_Mj_vot['y_true'], df_Mj_vot.index
        
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        pred_dict['ID']= y_id
   
        print("-------test_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        self.test_step_outputs.clear()
        # self.test_step_targets.clear()
        pprint(df_result)
        

        # df_result.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_test.csv')

        # pred_df = pd.DataFrame(pred_dict)
        # pred_df.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_test_pred.csv')
        self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_test')
    def _DecideDtype(self,inp_embed_type):
        if inp_embed_type in Text_pretrain:
            dtype=torch.long
        elif inp_embed_type in Audio_pretrain:
            dtype=torch.float
        elif inp_embed_type in Text_Summary:
            dtype=torch.long
        elif inp_embed_type in Text_Summary_Embedding:   
            dtype=torch.float
        return dtype
    def _safe_output(self):
        self.outStr=self.inp_embed_type.replace("/","__")
    def _save_results_to_csv(self, df_result, pred_dict, args, suffix):
        # Save df_result to CSV
        self._safe_output()
        df_result.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}.csv')

        # Save pred_df to CSV
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}_pred.csv')

class Heterogeneous2InpModel(SingleForwardModel):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.inp1Arg = args.inp1Arg
        self.inp2Arg = args.inp2Arg
        self.inp1_embed_type = self.config['inp1_embed']
        self.inp2_embed_type = self.config['inp2_embed']
        self.inp1_col_name = self.inp1Arg.inp_col_name
        self.inp2_col_name = self.inp2Arg.inp_col_name
        

        self.inp1_hidden_size = self.inp1Arg.inp_hidden_size
        self.inp2_hidden_size = self.inp2Arg.inp_hidden_size
        self.hidden = int(self.inp1_hidden_size + self.config['hidden_size'])
        self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)
        self.inp1_tokenizer, self.inp1_model, self.pooler1=self._setup_embedding(self.inp1_embed_type, self.inp1_hidden_size)
        self.inp2_tokenizer, self.inp2_model, self.pooler2=self._setup_embedding(self.inp2_embed_type, self.inp2_hidden_size)

        self.SummaryPool=nn.Linear(self.inp2Arg.inp_hidden_size,self.config['hidden_size'])
    def forward(self, inp1, inp2):
        # Add or modify the forward method for NewModel2
        # You can still use the functionality from the parent class by calling super().forward(inp)
        # ...
        out1 = self._get_embedding(inp1,self.inp1_embed_type, self.inp1_model, self.pooler1)
        out2 = self._get_embedding(inp2,self.inp2_embed_type, self.inp2_model, self.pooler2)
        out2_shrink=self.SummaryPool(out2)
        output = torch.cat((out1,out2_shrink),axis=1)  
        # output = torch.cat((out1,out2),axis=1)  
        logits = self.clf2(self.clf1(output))
    
        return logits
    def preprocess_dataframe(self):
        
        df_train = pd.read_csv(f"{self.inp1Arg.file_in}/train.csv")
        df_dev = pd.read_csv(f"{self.inp1Arg.file_in}/dev.csv")
        df_test = pd.read_csv(f"{self.inp1Arg.file_in}/test.csv")
        self.df_train=self._Tokenize(df_train, self.inp1_embed_type, self.inp1Arg.inp_col_name, self.inp1_tokenizer)
        self.df_dev=self._Tokenize(df_dev, self.inp1_embed_type, self.inp1Arg.inp_col_name, self.inp1_tokenizer)
        self.df_test=self._Tokenize(df_test, self.inp1_embed_type, self.inp1Arg.inp_col_name, self.inp1_tokenizer)

        concatenated_column = pd.concat([self.df_train[self.ID_col], self.df_dev[self.ID_col], self.df_test[self.ID_col]], ignore_index=True)        
        self.le.fit(concatenated_column)
        
        self._preprocess_loaded_summaries(self.inp2_embed_type,self.inp2Arg.inp_col_name, self.inp2_tokenizer)
        self._merge_DataAug2Data()
        print(f'# of train:{len(df_train)}, val:{len(df_dev)}, test:{len(df_test)}')
        self._df2Dataset()
        if 'samplebalance' in self.config and self.config['samplebalance']==True:
            self._buid_BalancedSampler()

    def _preprocess_loaded_summaries(self,inp2_embed_type,inp_col_name,inp2_tokenizer):
        df_train = pd.read_pickle(f"{self.inp2Arg.file_in}/train.pkl")
        df_dev = pd.read_pickle(f"{self.inp2Arg.file_in}/dev.pkl")
        df_test = pd.read_pickle(f"{self.inp2Arg.file_in}/test.pkl")


        df_train=self._Tokenize(df_train, inp2_embed_type,inp_col_name, inp2_tokenizer)
        df_dev=self._Tokenize(df_dev, inp2_embed_type,inp_col_name, inp2_tokenizer)
        df_test=self._Tokenize(df_test, inp2_embed_type,inp_col_name, inp2_tokenizer)

        
        df_test = df_test.reset_index(drop=True)
        self.df_train_aug=df_train
        self.df_dev_aug=df_dev
        self.df_test_aug=df_test
        self.Aug_col_name=self.inp2Arg.inp_col_name

    def _merge_DataAug2Data(self):
        pname_col_name='ID   '
        similar_col_name='session'
        def AppendID(df_data):
            if pname_col_name not in df_data.columns:
                df_data[pname_col_name]=df_data[similar_col_name]
        AppendID(self.df_train_aug)
        AppendID(self.df_dev_aug)
        AppendID(self.df_test_aug)

        self.df_train = pd.merge(self.df_train, self.df_train_aug, on='ID   ', how='left', suffixes=('', '_aug'))
        self.df_dev = pd.merge(self.df_dev, self.df_dev_aug, on='ID   ', how='left', suffixes=('', '_aug'))
        self.df_test = pd.merge(self.df_test, self.df_test_aug, on='ID   ', how='left', suffixes=('', '_aug'))
 
    def _df2Dataset(self):
        dtype1=self._DecideDtype(self.inp1_embed_type)
        dtype2=self._DecideDtype(self.inp2_embed_type)
        
        self.train_data = TensorDataset(
            torch.tensor(self.df_train[self.inp1Arg.inp_col_name].tolist(), dtype=dtype1),
            torch.tensor(self.df_train[self.inp2Arg.inp_col_name].tolist(), dtype=dtype2),
            torch.tensor(self.df_train[self.label_cols].tolist(), dtype=torch.long),
            torch.tensor(self.le.transform(self.df_train[self.ID_col]).tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(self.df_dev[self.inp1Arg.inp_col_name].tolist(), dtype=dtype1),
             torch.tensor(self.df_dev[self.inp2Arg.inp_col_name].tolist(), dtype=dtype2),
            torch.tensor(self.df_dev[self.label_cols].tolist(), dtype=torch.long),
            torch.tensor(self.le.transform(self.df_dev[self.ID_col]).tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(self.df_test[self.inp1Arg.inp_col_name].tolist(), dtype=dtype1),
             torch.tensor(self.df_test[self.inp2Arg.inp_col_name].tolist(), dtype=dtype2),
            torch.tensor(self.df_test[self.label_cols].tolist(), dtype=torch.long),
             torch.tensor(self.df_test.index.tolist(), dtype=torch.long),
             torch.tensor(self.le.transform(self.df_test[self.ID_col]).tolist(), dtype=torch.long),
        )




    def training_step(self, batch, batch_idx):
        # pprint(f"The sampler is initiated as:\n {self.sampler}")
        inp1, inp2, labels, ID = batch  
        # token,  labels = batch  
        logits = self(inp1, inp2) 
        # logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inp1, inp2, labels, ID = batch  
        # token, labels = batch  
        logits = self(inp1, inp2) 
        # logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)     
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
        ID=list(self.le.inverse_transform(ID.cpu().numpy()))

        # --> HERE STEP 2 <--
        self.val_step_outputs.append({
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
            'ID': ID,
        })
        # self.val_step_targets.append(y_true)
        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def test_step(self, batch, batch_idx):
        inp1, inp2, labels,id_, ID = batch 
        # token, labels,id_ = batch 
        logits = self(inp1, inp2) 
        # logits = self(token) 
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())
        ID=list(self.le.inverse_transform(ID.cpu().numpy()))

        # --> HERE STEP 2 <--
        self.test_step_outputs.append({
            'y_true': y_true,
            'y_pred': y_pred,
            'ID': ID,
        })
        # self.test_step_targets.append(y_true)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def _safe_outputStr_gen(self):
        self.outStr=self.config['params_tuning_str'].replace("/","_")
    def _save_results_to_csv(self, df_result, pred_dict, args, suffix):
        # Save df_result to CSV
        self._safe_outputStr_gen()
        df_result.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}.csv')

        # Save pred_df to CSV
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}_pred.csv')
    
# class SingleForwardModelRegression(SingleForwardModel):
#     # """
#     # Is the model of 0207_DM_SentenceLvl1input.py
#     # """
#     def __init__(self, args,config):
#         super().__init__(args, config)
#         # config:
        

#         self.label_names = ['mmse']
#         self.num_labels = 1
#     def _df2Dataset(self):
#         def DecideDtype(inp_embed_type):
#             if inp_embed_type in Text_pretrain:
#                 dtype=torch.long
#             elif inp_embed_type in Audio_pretrain:
#                 dtype=torch.float
#             return dtype
#         dtype=DecideDtype(self.inp_embed_type)
#         self.train_data = TensorDataset(
#             torch.tensor(self.df_train[self.inpArg.inp_col_name].tolist(), dtype=dtype),
#             torch.tensor(self.df_train[self.label_cols].tolist(), dtype=torch.float),
#         )
        
#         self.val_data = TensorDataset(
#              torch.tensor(self.df_dev[self.inpArg.inp_col_name].tolist(), dtype=dtype),
#             torch.tensor(self.df_dev[self.label_cols].tolist(), dtype=torch.float),
#         )

#         self.test_data = TensorDataset(
#              torch.tensor(self.df_test[self.inpArg.inp_col_name].tolist(), dtype=dtype),
#             torch.tensor(self.df_test[self.label_cols].tolist(), dtype=torch.float),
#              torch.tensor(self.df_test.index.tolist(), dtype=torch.long),
#         )
#     def training_step(self, batch, batch_idx):
#         # token, audio, labels = batch  
#         token,  labels = batch  
#         # logits = self(token, audio) 
#         logits = self(token) 
#         loss = nn.MSELoss()(logits, labels)   
        
#         return {'loss': loss}
#     def validation_step(self, batch, batch_idx):
#         # token, audio, labels = batch  
#         token, labels = batch  
#         # logits = self(token, audio) 
#         logits = self(token) 
#         loss = nn.MSELoss()(logits, labels)     
        
#         preds = logits.argmax(dim=-1)

#         y_true = list(labels.cpu().numpy())
#         y_pred = list(preds.cpu().numpy())

#         # --> HERE STEP 2 <--
#         self.val_step_outputs.append({
#             'loss': loss,
#             'y_true': y_true,
#             'y_pred': y_pred,
#         })
#         # self.val_step_targets.append(y_true)
#         return {
#             'loss': loss,
#             'y_true': y_true,
#             'y_pred': y_pred,
#         }
#     def test_step(self, batch, batch_idx):
#         # token, audio, labels,id_ = batch 
#         token, labels,id_ = batch 
#         # print('id', id_)
#         # logits = self(token, audio) 
#         logits = self(token) 
        
#         preds = logits.argmax(dim=-1)

#         y_true = list(labels.cpu().numpy())
#         y_pred = list(preds.cpu().numpy())

#         # --> HERE STEP 2 <--
#         self.test_step_outputs.append({
#             'y_true': y_true,
#             'y_pred': y_pred,
#         })
#         # self.test_step_targets.append(y_true)
#         return {
#             'y_true': y_true,
#             'y_pred': y_pred,
#         }
#     def on_validation_epoch_end(self):
#         loss = torch.tensor(0, dtype=torch.float)
#         for i in self.val_step_outputs:
#             loss += i['loss'].cpu().detach()
#         _loss = loss / len(self.val_step_outputs)
#         loss = float(_loss)
#         y_true = []
#         y_pred = []

#         for i in self.val_step_outputs:
#             y_true += i['y_true']
#             y_pred += i['y_pred']
            
#         y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
#         y_true = np.asanyarray(y_true)
        
#         pred_dict = {}
#         pred_dict['y_pred']= y_pred
#         pred_dict['y_true']= y_true
        
#         val_acc = r2_score(y_true=y_true, y_pred=y_pred)
        
#         self.log("val_acc", val_acc)
#         # print("y_pred= ", y_pred)
#         # print('\n\n\n')
#         # print("y_true= ", y_true)
#         # print('\n\n\n')
#         print("-------val_report-------")
#         metrics_dict = regression_report(y_true, y_pred)
#         df_result = pd.DataFrame(list(metrics_dict.values()), index=metrics_dict.keys(), columns=['corrs']).transpose()
#         pprint(df_result)
        

#         # df_result.to_csv(
#         #     f'{self.args.Output_dir}/{self.inp_embed_type}_val.csv')

#         # pred_df = pd.DataFrame(pred_dict)
#         # pred_df.to_csv(
#         #     f'{self.args.Output_dir}/{self.inp_embed_type}_val_pred.csv')
#         self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_val')
#         self.val_step_outputs.clear()
#         # self.val_step_targets.clear()
#         return {'loss': _loss}
#     def on_test_epoch_end(self):

#         y_true = []
#         y_pred = []

#         for i in self.test_step_outputs:
#             y_true += i['y_true']
#             y_pred += i['y_pred']
            
#         y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
#         y_true = np.asanyarray(y_true)
        
#         pred_dict = {}
#         pred_dict['y_pred']= y_pred
#         pred_dict['y_true']= y_true
        
        
#         print("-------test_report-------")
#         metrics_dict = regression_report(y_true, y_pred)
#         df_result = pd.DataFrame(list(metrics_dict.values()), index=metrics_dict.keys(), columns=['corrs']).transpose()
#         self.test_step_outputs.clear()
#         # self.test_step_targets.clear()
#         pprint(df_result)
        

#         # df_result.to_csv(
#         #     f'{self.args.Output_dir}/{self.inp_embed_type}_test.csv')

#         # pred_df = pd.DataFrame(pred_dict)
#         # pred_df.to_csv(
#         #     f'{self.args.Output_dir}/{self.inp_embed_type}_test_pred.csv')
#         self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_test')
#     def _save_results_to_csv(self, df_result, pred_dict, args, suffix):
#         # Save df_result to CSV
#         self._safe_output()
#         df_result.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}.csv')

#         # Save pred_df to CSV
#         pred_df = pd.DataFrame(pred_dict)
#         pred_df.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}_pred.csv')
    