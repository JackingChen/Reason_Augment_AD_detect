import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
from tqdm import tqdm

Combination_RAG13={'FakeSelect':['Simplified Sentence Structure', 'Word/phrase revision',
       'Telegraphic speech']}



Positive_attributes=[
    "Verbose Eloquence",
    "Fluent Continuity",
    "Concise Clarity",
    "Precise Expression",
    "Diverse Lexicon",
    "Articulate Detailing",
    "Pronoun Precision",
    "Grammatical Proficiency",
    "Seamless Fluency",
    "Narrative Fluidity",
    "Detailed Recall",
    "Complex Sentence Structure",
    "Organized Articulation"
]
Negative_attributes=[
    "Trailing off speech",
    "Circumlocution in speech",
    "Word/phrase revision",
    "Word/phrase repetition",
    "Telegraphic speech",
    "Misuse of pronouns",
    "Poor grammar",
    "Hesitation and pauses",
    "Lack of narrative coherence",
    "Limited recall of details",
    "Simplified Sentence Structure",
    "Difficulty Organizing Description"
]




RAG13_columns=['Poor grammar',
       'Simplified Sentence Structure', 'Word/phrase revision',
       'Telegraphic speech', 'Word/phrase repetition',
       'Lack of narrative coherence', 'Limited recall of details',
       'Difficulty Organizing Description', 'Hesitation and pauses',
       'Trailing off speech', 'Circumlocution in speech',
       'Detecting problem results', 'Empty speech', 'Misuse of pronouns']

Incorrect2Correct_indexes=  ['S167', 'S185', 'S187', 'S192', 'S202']
Correct2Incorrect_indexes=  ['S161', 'S179']
Incorrect2Correct_indexes_and_predictedAD=  ['S167', 'S185', 'S187', 'S192']

high_corr_cols=['Circumlocution in speech', 'Difficulty Organizing Description',
       'Hesitation and pauses', 'Lack of narrative coherence',
       'Limited recall of details', 'Trailing off speech',
       'Word/phrase repetition']


#########====================end Retreiver area==============================
class RAG_chatbot:
       # usage:
       # RAG_bot=RAG_chatbot()
       # RAG_bot.chatopenai=RAG_bot.Initialize_openai()
       # RAG_bot.Embedder=RAG_bot.Initialize_Embedder()
    def __init__(self):
        self.retreiver = None
        self.stepback_model = None
        self.answer_model = None
        self.few_shot_prompt=None
        self.MultiTurn_prompt={}
        self.MultiTurn_prompt[0]=HumanMessage(content="please answer the sheet")
        
    def Initialize_openai(self,env_script_path = '/home/FedASR/dacs/centralized/env.sh'):
        #activate environment
        # 读取 env.sh 文件内容
        with open(env_script_path, 'r') as file:
            env_content = file.read()

        # 将 env.sh 文件内容以换行符分割，并逐行执行
        for line in env_content.split('\n'):
            # 跳过注释和空行
            if line.strip() and not line.startswith('#'):
                # 使用 split 等号来分割键值对
                key, value = line.split('=', 1)
                # 设置环境变量
                os.environ[key.replace("export ","")] = value.strip()
        #Initialize openai
        openai.api_type = "azure"
        openai.api_version = "2023-05-15" 
        openai.api_base = os.getenv('OPENAI_API_BASE')  # Your Azure OpenAI resource's endpoint value.
        openai.api_key = os.getenv('OPENAI_API_KEY')
        chatopenai=ChatOpenAI(api_key=openai.api_key,model_kwargs={"engine": "gpt-35-turbo"})
        return chatopenai
    def Prompt2Conversation(self, prompt, chatopenai,verbose=False):
       
       conversation = [
              SystemMessage(content="You're a clinical doctor of alzheimer's disease"),
              HumanMessage(content=f"{prompt}"),
              ]
       for i in range(len(self.MultiTurn_prompt)+1):
              # Generate a response from the assistant
              if verbose:
                     print(conversation)
              response = chatopenai(conversation)

              output_parser = StrOutputParser()
              summary = output_parser.parse(response).content
              conversation.append(AIMessage(content=f"{summary}"))
              if i in self.MultiTurn_prompt:
                     conversation.append(self.MultiTurn_prompt[i])
              # Extract the assistant's reply from the response
              # assistant_reply = response.choices[0].text.strip()

              # Print the assistant's reply
              if verbose:
                     print(f"{i}'st response")
                     print("Assistant:", summary)
       return summary
    def Initialize_Embedder(self):
        os.environ["AZURE_OPENAI_API_KEY"] = openai.api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = openai.api_base


        embedder = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
        )
        return embedder

    def Initialize_fewshot_prompt(self, user_input):
        # 在知識庫中搜尋與使用者輸入相關的資訊
        # 這裡假設 knowledge_base 是一個包含資訊的字典或其他數據結構
        if user_input in self.knowledge_base:
            return self.knowledge_base[user_input]
        else:
            return None

def process_sessions(session_df, prompt_template, RAG_bot, Sensitive_replace_dict, use_text='text',\
    selected_people=[], verbose=True):
    Embedding_dict, Summary_dict, Prompt_dict = {}, {}, {}
    for session, row in tqdm(session_df.iterrows()):
        if len(selected_people)>0:
            if session not in selected_people:
                if verbose:
                    print(f"Skipped {session}")
                continue
        
        if session in Sensitive_replace_dict.keys():
            dialogue_content = row[use_text]
            for values in Sensitive_replace_dict[session]:
                dialogue_content = dialogue_content.replace(values[0], values[1])
            
        else:
            dialogue_content = row[use_text]

        prompt=prompt_template.format(dialogue_content=dialogue_content)
        # 1 turn asking
        # ans_middle = chatopenai(conversation)
        # output_parser = StrOutputParser()
        # summary = output_parser.parse(ans_middle).content
        
        summary=RAG_bot.Prompt2Conversation(prompt,RAG_bot.chatopenai)
        embeddings = RAG_bot.Embedder.embed_query(summary)
        Embedding_dict[session] = embeddings
        Summary_dict[session] = summary
        Prompt_dict[session] = prompt

    session_df['Embedding'] = session_df.index.to_series().apply(lambda x: Embedding_dict.get(x, []))
    session_df['Psych_Summary'] = session_df.index.to_series().apply(lambda x: Summary_dict.get(x, []))
    session_df['Psych_Prompt'] = session_df.index.to_series().apply(lambda x: Prompt_dict.get(x, []))
    return session_df

def process_sessions_ver3(session_df,session_positive,session_negative, prompt_template, RAG_bot, Sensitive_replace_dict, use_text='text',\
    use_Psych_summary='Psych_Summary',selected_people=[], verbose=True):
    Embedding_dict, Summary_dict, Prompt_dict = {}, {}, {}
    for session, row in tqdm(session_df.iterrows()):
        if len(selected_people)>0:
            if session not in selected_people:
                if verbose:
                    print(f"Skipped {session}")
                continue
        
        if session in Sensitive_replace_dict.keys():
            dialogue_content = row[use_text]
            for values in Sensitive_replace_dict[session]:
                dialogue_content = dialogue_content.replace(values[0], values[1])
            
        else:
            dialogue_content = row[use_text]

        negative_summary=session_negative[session_negative['session']==session].loc[session,use_Psych_summary]
        positive_summary=session_positive[session_negative['session']==session].loc[session,use_Psych_summary]
        prompt=prompt_template.format(dialogue_content=dialogue_content,
                                    negative_summary=negative_summary,
                                    positive_summary=positive_summary)
        # 1 turn asking
        # ans_middle = chatopenai(conversation)
        # output_parser = StrOutputParser()
        # summary = output_parser.parse(ans_middle).content
        
        summary=RAG_bot.Prompt2Conversation(prompt,RAG_bot.chatopenai)
        embeddings = RAG_bot.Embedder.embed_query(summary)
        Embedding_dict[session] = embeddings
        Summary_dict[session] = summary
        Prompt_dict[session] = prompt

    session_df['Embedding'] = session_df.index.to_series().apply(lambda x: Embedding_dict.get(x, []))
    session_df['Psych_Summary'] = session_df.index.to_series().apply(lambda x: Summary_dict.get(x, []))
    session_df['Psych_Prompt'] = session_df.index.to_series().apply(lambda x: Prompt_dict.get(x, []))
    return session_df

def Summary2Attribute(df_total: pd.DataFrame, col='Psych_Summary')->pd.DataFrame:

    def merge_df_row(df, row_df):
        # Create a copy of the df DataFrame to avoid modifying the original
        row_df_ = row_df.copy()

        # Iterate through the rows of df and add corresponding values from row
        for i, r in df.iterrows():
            row_df_[i] = r['Description']
            # print(r['circumlocution'])
            # print(r.columns)
        # print(row_df_)
        return row_df_
    df_augmented = pd.DataFrame()
    for i, row in df_total.iterrows():
        text=row[col]

        # df = pd.read_csv(StringIO(text), sep=":", names=["Item", "Description"], skipinitialspace=True, skiprows=1)
        lines = text.split('\n')  # Split text into lines
        items_list = []
        current_item = None

        for line in lines:
            line = line.strip()
            if ":" in line:
                current_item, description = map(str.strip, line.split(":", 1))
                items_list.append({'Item': current_item, 'Description': description})
            elif current_item is not None and line:
                # If there is a current item and the line is not empty, consider it as part of the description
                items_list[-1]['Description'] += ' ' + line

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(items_list)
        
        
        df['Item'] = df['Item'].str.replace('\t', '')

        df.set_index('Item', inplace=True)
        # print(df)

        row_df = pd.DataFrame(row).T

        merged_df = merge_df_row(df, row_df)
        df_augmented = pd.concat([df_augmented,merged_df], ignore_index=True)

    attributes_cols_set=set(list(df.index))
    attributes_cols_lst=list(attributes_cols_set)
    return df_augmented, attributes_cols_lst

def binarize_df(df_data):
    result_df = df_data.applymap(lambda x: 1 if pd.notna(x) and len(x) > 0 and x!='No'  else 0)
    return result_df