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

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertConfig, BertModel,XLMTokenizer, XLMModel

from Dementia_challenge_models import SingleForwardModel, BertPooler, Audio_pretrain, ModelArg, Model_settings_dict, Text_pretrain, Text_Summary
import librosa

class Model(SingleForwardModel):
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
        self.save_hyperparameters()
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
        self.labels_train_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_train,tag='train')
        self.labels_dev_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_dev,tag='dev')
        self.labels_test_df, _=self._Get_person_Lvl_Groundtruth_label(self.df_test,tag='test')
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
    

def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
        
    if config['task']=='regression':
        model=ModelRegression(args,config)
    elif config['task']=='classification':
        model = Model(args,config) 
    else:
        raise ValueError()
    
    model.preprocess_dataframe()
    pprint(f"The sampler is initiated as:\n {model.sampler}")
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{SaveRoot}/Model/{config['params_tuning_str']}/checkpoints",
        monitor='val_acc',
        auto_insert_metric_name=True,
        verbose=True,
        mode='max', 
        save_top_k=1,
      )    

    print(":: Start Training ::")
    #     
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback,checkpoint_callback],
        # callbacks=[early_stop_callback],
        enable_checkpointing = True,
        max_epochs=args.mdlArg.epochs,
        fast_dev_run=args.mdlArg.test_mode,
        num_sanity_val_steps=None if args.mdlArg.test_mode else 0,
        # deterministic=True, # True會有bug，先false
        deterministic=False,
        # For GPU Setup
        # gpus=[config['gpu']] if torch.cuda.is_available() else None,
        strategy='ddp_find_unused_parameters_true',
        precision=16 if args.mdlArg.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader(),ckpt_path="best")
    

if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='exp', help="learning rate")
    parser.add_argument("--hidden_size", type=int, default=16) 

    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--inp1_embed", type=str, default="mbert_sentence", help="should only be raw text or raw audio. It has to be sentence level stuff") 
    parser.add_argument("--inp2_embed", type=str, default="Summary", help="") 
    parser.add_argument("--SaveRoot", type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs') 
    parser.add_argument("--task", type=str, default="classification") 

    parser.add_argument("--samplebalance", action='store_true', help="Specify to enable sample balancing.")
    parser.add_argument("--sampleImbalance", action='store_false', help="Specify to enable sample balancing.")

    
    config = parser.parse_args()
    SaveRoot=config.SaveRoot
    


    script_path, file_extension = os.path.splitext(__file__)


    samplebalance_str='SampleBal' if config.samplebalance else 'SampleImBal'

    config.params_tuning_str='__'.join([config.inp1_embed,
                                        config.inp2_embed,
                                        str(config.epochs),
                                        str(config.hidden_size),
                                        samplebalance_str])
    # 使用os.path模組取得檔案名稱
    script_name = os.path.basename(script_path)
    task_str='result_regression' if config.task=='regression' else 'result_classification'
    Output_dir=f"{SaveRoot}/{task_str}/{script_name}/" 
    os.makedirs(Output_dir, exist_ok=True)
    print(config)

    
    class Inp1Arg:
        inp_hidden_size = Model_settings_dict[config.inp1_embed]['inp_hidden_size']
        pool_hidden_size = inp_hidden_size # BERT-base: 768, BERT-large: 1024, BERT paper setting
        linear_hidden_size = inp_hidden_size
        inp_col_name = Model_settings_dict[config.inp1_embed]['inp_col_name']
        file_in = Model_settings_dict[config.inp1_embed]['file_in']
    class Inp2Arg:
        inp_hidden_size = Model_settings_dict[config.inp2_embed]['inp_hidden_size']
        pool_hidden_size = inp_hidden_size # BERT-base: 768, BERT-large: 1024, BERT paper setting
        linear_hidden_size = inp_hidden_size
        inp_col_name = Model_settings_dict[config.inp2_embed]['inp_col_name']
        file_in = Model_settings_dict[config.inp2_embed]['file_in']
    class Arg:
        mdlArg=ModelArg()
        inp1Arg=Inp1Arg()
        inp2Arg=Inp2Arg()
        Output_dir=Output_dir

    args = Arg()
    args.mdlArg.epochs=config.epochs
    main(args,config.__dict__)       


"""

python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed en
python 0207_DM_multi.py --gpu 1 --t_embed xlm --a_embed en

# don
python 0207_DM_multi.py --gpu 0 --t_embed xlm --a_embed gr
python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed gr

"""