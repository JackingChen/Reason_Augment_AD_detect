import pickle
import pandas as pd
import os
import argparse
import pandas as pd
from io import StringIO

import openai
#importing the necessary dependencies
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub
from rouge_score import rouge_scorer
import subprocess
from langchain.embeddings import AzureOpenAIEmbeddings
import numpy as np

from Analysis_utils import RAG_chatbot, process_sessions, Summary2Attribute, binarize_df

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings', help="")
parser.add_argument('--prompt_dir', type=str, default='text_data2vec-audio-large-960h_Phych-Positive_attributes', help="")
args = parser.parse_args()

file_path=args.file_path
prompt_dir=args.prompt_dir
filename=os.path.basename(file_path)



train_file=f"{file_path}/{prompt_dir}/train.pkl"
dev_file=f"{file_path}/{prompt_dir}/dev.pkl"
test_file=f"{file_path}/{prompt_dir}/test.pkl"

df_train = pd.read_pickle(train_file)
df_dev = pd.read_pickle(dev_file)
df_test = pd.read_pickle(test_file)
df_train['ex']='train'
df_dev['ex']='dev'
df_test['ex']='test'

df_total=pd.concat([df_train,df_dev,df_test],axis=0)









def Attribute2ConfusionMatrix(df_augmented: pd.DataFrame, attributes_cols_lst: list)->pd.DataFrame:
    HC_idxes=df_augmented['dementia_labels']==0
    AD_idxes=df_augmented['dementia_labels']==1
    AD_attribute_df=df_augmented.loc[AD_idxes,attributes_cols_lst]
    HC_attribute_df=df_augmented.loc[HC_idxes,attributes_cols_lst]
    AD_bin_df=binarize_df(AD_attribute_df)
    HC_bin_df=binarize_df(HC_attribute_df)
    AD_Sum=AD_bin_df.sum()
    HC_Sum=HC_bin_df.sum()
    AD_HC__cm=pd.DataFrame([AD_Sum,HC_Sum],index=['AD','HC'])
    return AD_HC__cm


df_augmented, attributes_cols_lst=Summary2Attribute(df_total)
AD_HC__cm=Attribute2ConfusionMatrix(df_augmented,attributes_cols_lst)
AD_HC__cm


RAG_bot=RAG_chatbot()
RAG_bot.chatopenai=RAG_bot.Initialize_openai()
RAG_bot.Embedder=RAG_bot.Initialize_Embedder()



df_train, attributes_cols_lst=Summary2Attribute(df_train)
df_dev, attributes_cols_lst=Summary2Attribute(df_dev)
df_test, attributes_cols_lst=Summary2Attribute(df_test)





def generate_embeddings(df_data, sel_col, Embedder)->pd.core.series.Series:
    Embedding_dict = {}

    for session, row in df_data.iterrows():
        text = row[sel_col]
        if pd.isna(text) or len(str(text).strip()) == 0:
            text = 'None'
        embeddings = Embedder.embed_query(text)
        Embedding_dict[session] = embeddings

    out_df = df_data.index.to_series().apply(lambda x: Embedding_dict.get(x, []))
    
    return out_df

def Emb_pool(concatenated_embeddings):
    Max_dict, Mean_dict = {}, {}
    for i, row in concatenated_embeddings.iterrows():
        col_bag = []
        for c in row:
            col_bag.append(np.array(c))
        col_array = np.vstack(col_bag)  # (5, emb_size)
        mean_array = np.mean(col_array, axis=0)
        max_array = np.max(col_array, axis=0)
        Max_dict[i] = max_array
        Mean_dict[i] = mean_array

    concatenated_embeddings['max_emb'] = concatenated_embeddings.index.to_series().apply(lambda x: Max_dict.get(x, []))
    concatenated_embeddings['mean_emb'] = concatenated_embeddings.index.to_series().apply(lambda x: Mean_dict.get(x, []))

    return concatenated_embeddings[['max_emb','mean_emb']]

# df_data=df_test
# concatenated_embeddings = pd.DataFrame()
# for sel_col in select_cols:
#     result_embeddings = generate_embeddings(df_data, sel_col, Embedder)
#     # Concatenate the result_embeddings on columns
#     concatenated_embeddings = pd.concat([concatenated_embeddings, result_embeddings], axis=1)

# attribute_embedding_df=Emb_pool(concatenated_embeddings) # col=['max_emb','mean_emb']

def process_AttributeEmb(df_data, select_cols, Embedder):
    concatenated_embeddings = pd.DataFrame()

    for sel_col in select_cols:
        result_embeddings = generate_embeddings(df_data, sel_col, Embedder)
        concatenated_embeddings = pd.concat([concatenated_embeddings, result_embeddings], axis=1)

    attribute_embedding_df = Emb_pool(concatenated_embeddings)
    merged_df = pd.merge(df_data, attribute_embedding_df, left_index=True, right_index=True)
    return merged_df

def process_AttributeEmb4Analysis(df_data, select_cols, Embedder):
    # Usage:
    # df_train = process_AttributeEmb4Analysis(df_train, select_cols, Embedder)

    df_attriEmbs=pd.DataFrame()
    for sel_col in select_cols:
        result_embeddings = generate_embeddings(df_data, sel_col, Embedder)
        df_attriEmbs[sel_col]=result_embeddings
    merged_df = pd.merge(df_data, df_attriEmbs, left_index=True, right_index=True)
    return merged_df


# manual_select_cols=['Hesitation and Pauses',
# 'Limited Recall of Details',
# 'Difficulty Organizing Description',
# 'Lack of Narrative Coherence',
# 'trailing off speech']
manual_select_cols=[]

select_cols=manual_select_cols if len(manual_select_cols)>0 else attributes_cols_lst

df_train = process_AttributeEmb(df_train, select_cols, RAG_bot.Embedder)
df_dev = process_AttributeEmb(df_dev, select_cols, RAG_bot.Embedder)
df_test = process_AttributeEmb(df_test, select_cols, RAG_bot.Embedder)

def process_SummaryEmb(df_data, Embedder, sel_col='Summary'):
    summary_embeddings = generate_embeddings(df_data, sel_col, Embedder)
    df_data['Summary']=summary_embeddings
    # merged_df = pd.merge(df_data, summary_embeddings, left_index=True, right_index=True)
    return df_data

df_train = process_SummaryEmb(df_train, RAG_bot.Embedder)
df_dev = process_SummaryEmb(df_dev, RAG_bot.Embedder)
df_test = process_SummaryEmb(df_test, RAG_bot.Embedder)

outpath=f'{file_path}/{args.prompt_dir}_aug'
os.makedirs(outpath, exist_ok=True)
df_train.to_pickle(f"{outpath}/train.pkl")
df_dev.to_pickle(f"{outpath}/dev.pkl")
df_test.to_pickle(f"{outpath}/test.pkl")