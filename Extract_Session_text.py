import os
import argparse
import pandas as pd
from addict import Dict

from rouge_score import rouge_scorer
import pandas as pd
import argparse 
import os
from bleu import list_bleu
from fastchat.model import load_model, get_conversation_template
from langchain.prompts import PromptTemplate
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
from addict import Dict

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
from tqdm import tqdm
import re

from Analysis_utils import RAG_chatbot, process_sessions,process_sessions_ver3


parser = argparse.ArgumentParser()
parser.add_argument('--using_text_mode', type=str, default='text', help="[text | pred_str]")
parser.add_argument('--Psych_prefix', type=str, default='text_data2vec-audio-large-960h_Phych-', help="")
parser.add_argument('--psych_summary_root', type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings', help="")

parser.add_argument('--input_file', type=str, default='./saves/results/data2vec-audio-large-960h.csv', help="[./saves/results/data2vec-audio-large-960h.csv | data2vec-audio-large-960h.train]")

parser.add_argument('--selected_psych', type=str, default='Positive_attributes', help="")
parser.add_argument('--neg_attr_prompt', type=str, default='', help="")
parser.add_argument('--pos_attr_prompt', type=str, default='', help="")

args = parser.parse_args()
using_text_mode=args.using_text_mode
input_file=args.input_file
psych_summary_root=args.psych_summary_root
Psych_prefix=args.Psych_prefix
pos_attr_prompt=args.pos_attr_prompt
neg_attr_prompt=args.neg_attr_prompt

def check_string_match(s):
    pattern = re.compile(r'^psych_ver_3.*$')
    return bool(pattern.match(s))

if check_string_match(args.selected_psych):
    assert len(pos_attr_prompt) >0
    assert len(neg_attr_prompt) >0

positive_file=f'{psych_summary_root}/{Psych_prefix}{pos_attr_prompt}'
negative_file=f'{psych_summary_root}/{Psych_prefix}{neg_attr_prompt}'
ASR_name=os.path.basename(os.path.dirname(input_file))
ds_tag=os.path.basename(input_file).replace(".csv","")
df_test=pd.read_csv(input_file)


# 假設你已經使用 pd.read_csv 讀取了 CSV 檔案到 df_test 中
df_test['path'] = df_test['path'].str.rstrip('.wav')
# 使用 str.split 拆分 'path' 欄位
df_test[['session', 'role', 'number', 'start_time', 'end_time']] = df_test['path'].str.split('_', expand=True)

# 如果 'number' 欄位的末尾包含 '.wav'，你可能需要再進行一次拆分
df_test['number'] = df_test['number'].str.rstrip('.wav')

# 將 'start_time' 和 'end_time' 欄位轉換為數值型別
df_test[['start_time', 'end_time']] = df_test[['start_time', 'end_time']].astype(int)



# Packer
def Packer(df_test) -> dict:
    People_dict = dict(tuple(df_test.groupby('session')))
    return People_dict
People_dict=Packer(df_test)
# Filterer



def filter_people_dict(People_dict, mode="INV+PAR", verbose=False) -> dict:# 'INV' , 'PAR', 'INV+PAR'
    filtered_people_dict = {}

    for session, data_frame in People_dict.items():
        # 使用 query 過濾 'role' 為 'INV' 或 'PAR'
        if mode == "PAR":
            filtered_data_frame = data_frame.query("role == 'PAR'")
            filtered_people_dict[session] = filtered_data_frame
        elif mode == "INV":
            filtered_data_frame = data_frame.query("role == 'INV'")
            filtered_people_dict[session] = filtered_data_frame
        elif mode == "INV+PAR":
            filtered_people_dict[session] = data_frame
        else:
            raise OSError
    
    if verbose:
        # 印出過濾後的 People_dict 中每個 session 的 DataFrame
        for session, data_frame in filtered_people_dict.items():
            print(f"Session: {session}")
            print(data_frame)
            print("\n")

    return filtered_people_dict
role_mode="INV+PAR"
People_dict = filter_people_dict(People_dict, mode=role_mode, verbose=False)

def Dialogueturn2corpus(data_frame, mode='text'): #mode can be 'pred_str' or 'text'
    # 按 'start_time' 列進行排序
    sorted_data_frame = data_frame.sort_values(by='start_time')

    # 添加前綴並使用 '\n' 進行拼接
    processed_text = sorted_data_frame.apply(lambda row: f"{row['role']}: {row[mode]}", axis=1).str.cat(sep='\n')

    return processed_text

# Dialogue Formatter
def Dialogue_Formatter(People_dict, sep="\n",role_mode='PAR')->dict:
    session_df=pd.DataFrame()
    for session, data_frame in People_dict.items():
        total_info=data_frame.iloc[0].copy()
        sessional_text = Dialogueturn2corpus(data_frame,mode='text')
        sessional_predStr = Dialogueturn2corpus(data_frame,mode='pred_str')
        
        # total_info,'text']=sessional_text
        # session_df.loc[session,'pred_str']=sessional_predStr
        # session_df.loc[session,'role']=role_mode
        # session_df.loc[session,'start_time']=data_frame['start_time'].min()
        # session_df.loc[session,'end_time']=data_frame['end_time'].max()

        total_info['text']=sessional_text
        total_info['pred_str']=sessional_predStr
        total_info['role']=role_mode
        total_info['start_time']=data_frame['start_time'].min()
        total_info['end_time']=data_frame['end_time'].max()
        session_df = pd.concat([session_df, pd.DataFrame([total_info], index=[session])])
    return session_df

session_df=Dialogue_Formatter(People_dict,role_mode=role_mode)

RAG_bot=RAG_chatbot()
RAG_bot.chatopenai=RAG_bot.Initialize_openai()
RAG_bot.Embedder=RAG_bot.Initialize_Embedder()
# Transform to Emb
# def Transform_to_Emb(dict: data, prompt, chatopenai, )
    
#     return Emb


from prompts import assesmentPrompt_template, Instruction_templates, Psychology_template,\
    Sensitive_replace_dict, generate_psychology_prompt, Direct_template, generate_direct_prompt



if args.selected_psych in Direct_template.keys():
   prompts_dict  = generate_direct_prompt()
else:
    prompts_dict = generate_psychology_prompt(assessment_prompt_template=assesmentPrompt_template,
                                                instruction_template=Instruction_templates[args.selected_psych],
                                                psychology_template=Psychology_template,
                                                )



result_prompts={k:v for k,v in prompts_dict.items() if k == args.selected_psych}
# result_prompts={k:v for k,v in  prompts_dict.items()}
for key, prompt_template in tqdm(result_prompts.items()):
    # session_df = process_sessions(session_df, prompt_template, chatopenai, Sensitive_replace_dict, Embedder, use_text=using_text_mode,
    #                               selected_people=mmse_analyze_selected_people)
    if check_string_match(args.selected_psych):
        session_positive=pd.read_pickle(f'{positive_file}/{ds_tag}.pkl')
        session_negative=pd.read_pickle(f'{negative_file}/{ds_tag}.pkl')
        session_df = process_sessions_ver3(session_df,session_positive,session_negative, prompt_template, RAG_bot, Sensitive_replace_dict, use_text=using_text_mode,verbose=False)
    else:    
        session_df = process_sessions(session_df, prompt_template, RAG_bot, Sensitive_replace_dict, use_text=using_text_mode)
    prompt_name=f"Phych-{key}"

    

    # OutFile_path=f"dacs/centralized/EmbFeats/Lexical/TextSummarize_Emb.pkl"
    # Output_Root="EmbFeats/Lexical"
    Output_Root=f"/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/{using_text_mode}_{ASR_name}_{prompt_name}"
    if not os.path.exists(Output_Root):
        os.makedirs(Output_Root)


    OutFile_path=f"{Output_Root}/{ds_tag}.pkl"
    # Save the DataFrame as a pickle file
    print(f"File saved at {OutFile_path}")
    session_df.to_pickle(OutFile_path)
