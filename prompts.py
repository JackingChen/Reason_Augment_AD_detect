from addict import Dict
ori_lst = [
    ["S181", "PAR: THAT'S A GOOD WAY TO BREAK HIS NECK"],
    ["S181", "PAR: BREAK HIS BACK I SHOULDA SAID"],
    ["S205", "PAR: THE LITTLE GIRL IS LAUGHING AT THE BOY FALLING OFF THE CHAIR"]
]

rep_lst = [
    ["S181", "PAR: THAT WILL HURT HIMSELF"],
    ["S181", "PAR: LIKE HURT HIS BACK I SHOULDA SAID"],
    ["S205", "PAR: THE LITTLE GIRL FINDS AMUSEMENT IN THE BOY ACCIDENTALLY FALLING OFF THE CHAIR"]
]

Sensitive_replace_dict = {}

for ori_item, rep_item in zip(ori_lst, rep_lst):
    key = ori_item[0]
    value = (ori_item[1], rep_item[1])
    if key not in Sensitive_replace_dict:
        Sensitive_replace_dict[key] = []
    Sensitive_replace_dict[key].append(value)
#======================================================    
mmse_people_select={}
mmse_people_select['mmse_low']=set(['S111','S090'])
mmse_people_select['mmse_middle']=set(['S110','S114'])
mmse_people_select['mmse_high']=set(['S061','S062'])

mmse_analyze_selected_people = set()

# 合併所有值到 selected_people 集合中
for value_set in mmse_people_select.values():
    mmse_analyze_selected_people = mmse_analyze_selected_people.union(value_set)
#======================================================
Symbol_template={}
Symbol_template["importanceTag"]={
    'st':'[notice]',
    'ed':'[\\notice]'
}
Psychology_template=Dict()


# Psychology_template['inability_to_answer_questions'] = {
#     'definition': "Inability to answer questions about a scene, struggling to provide accurate and relevant answers, indicating impaired comprehension.",
#     'example': [
#         "Inability to Answer Questions: I, um, don't really know what's happening in the picture. You see, there are, you know, things, but I can't say much about them.",
#         "Difficulty Providing Relevant Answers: When you asked about, um, the scene, I'm not sure, you know, what to say. It's a bit confusing for me." 
#         ]
# }

# Psychology_template['anomia']['definition']=f"Empty speech,  trailing off speech, circumlocution in speech"
# Psychology_template['anomia']['example']=[f"Empty speech: He’s trying to get {Symbol_template['importanceTag']['st']} this {Symbol_template['importanceTag']['ed']} and he’s gonna fall off of {Symbol_template['importanceTag']['st']} there {Symbol_template['importanceTag']['ed']}",
#                                           f"trailing off speech: If that little girl {Symbol_template['importanceTag']['st']} don’t xxx {Symbol_template['importanceTag']['ed']}",
#                                           f"circumlocution in speech: The boy hasn’t {Symbol_template['importanceTag']['st']} gotten down to his {Symbol_template['importanceTag']['st']} fall {Symbol_template['importanceTag']['ed']} yet."]
# Psychology_template['disflueny']['definition']=f"Word/phrase revision, word/phrase repetition, phonological fragment"
# Psychology_template['disflueny']['example']=[f"Word/phrase revision: The wife is wiping a {Symbol_template['importanceTag']['st']} dish plate. {Symbol_template['importanceTag']['ed']}",
#                                              f"word/phrase repetition: {Symbol_template['importanceTag']['st']} His his {Symbol_template['importanceTag']['ed']} sister’s asking for one.",
#                                              f"phonological fragment: Here’s a {Symbol_template['importanceTag']['st']} sp {Symbol_template['importanceTag']['ed']} water spigot here."
#                                              ]
Psychology_template['anomia']['definition']=f"Empty speech,  trailing off speech, circumlocution in speech"
Psychology_template['anomia']['example']=[f"Empty speech: Eloquent articulation lacking the expression of meaningful information.",
                                          f"trailing off speech: dropping speech, when the last few words in an utterance become barely audible.",
                                          f"circumlocution in speech: circumlocution of words/concepts within an utterance."]



Psychology_template['disflueny']['definition']=f"Word/phrase revision, word/phrase repetition, phonological fragment"
Psychology_template['disflueny']['example']=[f"Word/phrase revision: reviewing and making changes to individual words or phrases within a piece of written content."
                                             f"word/phrase repetition: same word or phrase is used multiple times within a short span of text",
                                             ]
Psychology_template['Agrammatism']['definition']=f"Telegraphic speech, misuse of pronuns, poor grammar"
Psychology_template['Agrammatism']['example']=[f"Telegraphic speech: a style of communication that is concise and stripped down to the essential words, similar to the simplicity seen in telegrams.",
                                               f"misuse of pronuns: using pronouns incorrectly in a sentence, which can result in confusion or ambiguity about the intended subject or object.",
                                               f"poor grammar:  incorrect or non-standard use of the rules and structures of a language."]

Psychology_template['psych1']['definition'] = Psychology_template['anomia']['definition'] +',' +Psychology_template['disflueny']['definition'] +',' +Psychology_template['Agrammatism']['definition']
Psychology_template['psych1']['example'] = Psychology_template['anomia']['example'] + Psychology_template['disflueny']['example'] + Psychology_template['Agrammatism']['example']

# Word Finding and Vocabulary Impairments
Psychology_template['hesitation_and_pauses'] = {
    'definition': "Hesitation and pauses in speech, experiencing difficulty finding appropriate words, leading to pauses or circumlocutions.",
    'example': [
    "Hesitation and Pauses: The individual tried to describe a common household tool used for tightening screws but hesitated, saying, 'I need that thing, you know, the one with a handle that turns and helps put things together.'"
    ]

}


# Pragmatic Language Deficits
Psychology_template['lack_of_narrative_coherence'] = {
    'definition': "Lack of narrative coherence, struggling to organize description in a logical and coherent manner, disrupting the flow of the narrative.",
    'example': ["Lack of Narrative Coherence: refers to a situation where a story or piece of writing lacks clarity, logical progression, or a cohesive structure.",
 "Simplified Sentence Structure: The movie was, uh, good. It had, you know, action and, um, interesting characters.",
 "Difficulty Organizing Description: I saw, um, a thing, and it was, you know, interesting because of, uh, some reasons that I can't quite, you know, put in order."
 ]
}

# Memory Impairments
Psychology_template['limited_recall_of_details'] = {
    'definition': "Limited recall of details, difficulty recalling specific details from a picture, reflecting memory deficits.",
    'example': [
        "Limited Recall of Details: I remember, you know, seeing something, but I can't recall the, uh, specific details, like colors or, um, what people were doing.",
        ]
}

Psychology_template['Positive_attributes'] = {
    'definition': "meaningful speech, Fluent and continuous speech, concise expression or direct communication, Organized and structured descriptions, Coherent narrative, Enunciating speech, grammarly correct, Confident speech.",
    'example': [
        "meaningful speech: clear, purposeful, and coherent communication that reflects an individual's ability to convey thoughts and ideas in a way that makes sense and demonstrates cognitive function.",
        "Fluent and continuous speech: express thoughts or ideas smoothly, coherently, and without significant interruptions or hesitations, indicating intact language skills and cognitive functioning.",
        "concise expression or direct communication: conveying ideas, information, or messages using a minimal and straightforward number of words, avoiding unnecessary details or elaboration.",
        "Organized and structured descriptions: clear and systematic presentations of information, with a logical arrangement and well-defined order in conveying details or attributes.",
        "Coherent narrative: logically structured story or account with clear connections between its elements.",
        "Enunciating speech: where most of the sentences are complete and clear.",
        "grammarly correct: no syntax error, spelling mistakes.",
        "Confident speech: No Word/phrase revision, word/phrase repetition, phonological fragment"
]
}

Psychology_template['psych2']['definition'] = Psychology_template['hesitation_and_pauses']['definition'] +',' +Psychology_template['lack_of_narrative_coherence']['definition'] +',' +Psychology_template['limited_recall_of_details']['definition']
Psychology_template['psych2']['example'] = Psychology_template['hesitation_and_pauses']['example'] +Psychology_template['lack_of_narrative_coherence']['example'] +Psychology_template['limited_recall_of_details']['example']

Psychology_template['psych_ver1']['definition'] = Psychology_template['psych1']['definition'] +',' +Psychology_template['psych2']['definition']
Psychology_template['psych_ver1']['example'] = Psychology_template['psych1']['example'] +Psychology_template['psych2']['example']

Rule_statement={}
Rule_statement['psych_ver1']=Rule_statement['Positive_attributes']="""Note.
* If you didn't detect any problem please leave it blank. 
* The sumary should be in consistant with the detected problems in the answering sheet.
* You should only analyze what appeared in the dialogue, and do not analyze anything appear in the psycological definition
* Keep the summary short and precise. 
* The answering sheet should be in the format of:"""

Rule_statement['psych1']="""Note.   
* The summary should be consistent with the detected attributes in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the psychological linguistic attribute definition 
* Keep the summary short and precise.  

* Please fill in the following sheet below for the 8 attributes, and paste the exact sentence from the dialog exhibiting the attribute after the attribute name :. Otherwise, do not paste anything after the attribute name.
* When you are generating each attribute, please check the following rules:
* Please check if the sentences for a detected attribute are exactly from the dialog
* Please check attributes not detected have empty results
* Please check the sheet format is correct 


"""

Rule_statement['data_augmentation_attribute']="""Note. 
* the generated description has to be describing the same story. 
* the generated description has to follow the same format as the provided dialog. Change a line if the speaker changes. That is, INV: or PAR: should be the beginning of a sentence."""


Example_templates={}
Example_templates['psych_ver1']="""Before starting, lets see one answer example about other dialogue (not dialogue in this case):
detected problems:
	Empty speech:
	trailing off speech: "S UH OH I CAN'T READ"
	circumlocution in speech:
	Word/phrase revision:
	Telegraphic speech:
	misuse of pronouns:
	poor grammar:
	Hesitation and Pauses: "S UH OH I CAN'T READ"
	Lack of Narrative Coherence: The responses lack a clear and organized narrative structure. The participant struggles to provide a coherent and logical description of the picture.
	Simplified Sentence Structure:
	Difficulty Organizing Description: The participant has difficulty organizing the description, leading to hesitations and pauses in speech.
	Limited Recall of Details: The participant has difficulty recalling specific details from the picture, as seen in statements like "I CAN'T" and "UH THAT'S A LITTLE."

Summary: The participant's speech exhibits hesitation and pauses, a lack of narrative coherence, and difficulty organizing a description of the picture. There are also instances of limited recall of details."""
Example_templates['data_augmentation_attribute']="""


"""


Form_templates={}
Form_templates['psych_ver1']="""detected problems:
	Empty speech:
	trailing off speech:
	circumlocution in speech:
	Word/phrase revision:
	Telegraphic speech:
	misuse of pronuns:
	poor grammar:
	Hesitation and Pauses:
	Lack of Narrative Coherence:
	Simplified Sentence Structure:
	Difficulty Organizing Description:
	Limited Recall of Details:

Summary: 

"""
Form_templates['Positive_attributes']="""detected attributes:
    Fluent and continuous speech:
    Concise expression or direct communication:
    Organized and structured descriptions:
    Coherent narrative:
    Enunciating speech:
    Grammarly correct:
    Confident speech:
    
Summary: 

"""
Background_prompt="""{rule_statement}

{form_statement}
---

{example}"""

Instruction_templates={}
Instruction_templates['psych_ver1']=Background_prompt.format(rule_statement=Rule_statement['psych_ver1'],
                                                             form_statement=Form_templates['psych_ver1'],
                                                             example=Example_templates['psych_ver1']
                                                             )
Instruction_templates['psych1']=Background_prompt.format(rule_statement=Rule_statement['psych1'],
                                                             form_statement=Form_templates['psych_ver1'],
                                                             example=Example_templates['psych_ver1']
                                                             )
Instruction_templates['Positive_attributes']=Background_prompt.format(rule_statement=Rule_statement['Positive_attributes'],
                                                             form_statement=Form_templates['Positive_attributes'],
                                                             example=''
                                                             )
assesmentPrompt_template=Dict()
assesmentPrompt_template['basic']="""Read the psychological definition below. Analyze the dialogue provided and identify if PAR exhibits any of the psychological problems mentioned. If problems are detected, fill in the answering sheet accordingly. Only analyze what appears in the dialogue, not the psychological definition. Keep the summary short and precise.

psycological definition:
{psychology_template}

{Instruction_template}

Dialogue:

{content}
"""

##********************** TBD: external_source 怎麼找要找一下
assesmentPrompt_template['external_source']="""{Instruction_template}

psycological definition:
---

{psychology_template}

{external_source}

Dialogue:

- {content}


"""

assesmentPrompt_template['data_augmentation']="""Based on the following dementia assessment results: 

{content}

can you help keep the speaker's style and Alzheimer's disease status and generate another description results to simulate the patient performing the test again?"""
# assesmentPrompt_template['data_augmentation_attribute']="""Given a participant (PAR) undergoing the Cookie Thief test for Alzheimer's disease measurement, the transcripts of the dialogs are as follows: 
# “””
# {dialogue_content}
# “”” 

# we have an initial diasnosis based on above narrative:
# {diagnosis_content}


# Please generate another description result to simulate the patient performing the Cookie Thief test again by keeping the speaker's style and Alzheimer's disease status.
# {Rule_statement}

# +++++ 
# [generated dialog here]
# +++++ 
# [reason here]"""
assesmentPrompt_template['data_augmentation_attribute']="""Given a participant (PAR) undergoing the Cookie Thief test for Alzheimer's disease measurement, the transcripts of the dialogs are as follows: 
“””
{dialogue_content}
“”” 

we have an initial diasnosis based on above narrative:
{diagnosis_content}


Please generate another description result to simulate the patient performing the Cookie Thief test again by keeping the speaker's style and Alzheimer's disease status.
{Rule_statement}

[generated dialog here]
"""

def generate_psychology_prompt(assessment_prompt_template, instruction_template, psychology_template):
    prompts_dict = {}

    # for instruction_template in instruction_templates:
    for key, values in psychology_template.items():
        definition = psychology_template[key]['definition']
        examples = '\n'.join(psychology_template[key]['example'])

        psychology_temp = f"""
        - definition: {definition}
        - examples: {examples}
        """

        assess_prompt_kargs = {
            'Instruction_template': instruction_template,
            'psychology_template': psychology_temp,
            'content': '{dialogue_content}'
        }
        prompt = assessment_prompt_template['basic'].format(**assess_prompt_kargs)
        prompts_dict[key]=prompt

    return prompts_dict
Direct_template={}
Direct_template['psych_ver_1.1']="""Given a participant (PAR) going through with the Cookie Thief test for Alzheimer's disease measurement as the dialogue where an investigator (INV) also engaged, you need to detect if the subject has the following defined linguistic deficits. Analyze the dialogue provided and identify if PAR exhibits any of the linguistic deficit problems mentioned. If problems are detected, fill in the answering sheet accordingly. If not, keep the sheet Only analyze what appears in the dialogue, not the psychological definition. Keep the summary short and precise. 

Defined 13 linguistic deficit problems: 
empty speech, trailing off speech, circumlocution in speech, word/phrase revision, word/phrase repetition, telegraphic speech, misuse of pronouns, poor grammar, hesitation and pauses, lack of narrative coherence, limited recall of details. 


1. Empty speech:
• Definition: Eloquent articulation lacking the expression of meaningful information.
2. Trailing off speech:
• Definition: dropping speech, when the last few words in an utterance become barely audible.
3. Circumlocution in speech:
• Definition: circumlocution of words/concepts within an utterance. experiencing difficulty finding appropriate words, leading to pauses or circumlocutions.
4. Word/phrase revision:
• Definition: reviewing and making changes to individual words or phrases within a piece of written content.
5. Word/phrase repetition:
• Definition: same word or phrase is used multiple times within a short span of text
6. Telegraphic speech:
• Definition: a style of communication that is concise and stripped down to the essential words, similar to the simplicity seen in telegrams.
7. Misuse of pronouns:
• Definition: using pronouns incorrectly in a sentence, which can result in confusion or ambiguity about the intended subject or object.
8. Poor grammar:
• Definition: incorrect or non-standard use of the rules and structures of a language.
9. Hesitation and pauses:
• Definition: Frequent interruptions in speech with pauses or hesitations.
10. Lack of narrative coherence:
• Definition: refers to a situation where a story lacks clarity, logical progression, or a cohesive structure. struggling to organize descriptions logically and coherently, disrupting the flow of the narrative. 
11. Limited recall of details:
• Definition: Difficulty in remembering and expressing specific details.
12. Simplified Sentence Structure:
• Definition: The use of uncomplicated and straightforward sentence constructions, often with shorter and clearer syntax. This can be a linguistic feature observed in individuals with cognitive impairments, including Alzheimer's disease.
13. Difficulty Organizing Description:
• Definition: Challenges in arranging information or events coherently and logically when providing a narrative or describing an event. This difficulty may lead to disjointed or fragmented storytelling, making it harder for the listener to follow the sequence of events.

Note.  
* The summary should be consistent with the detected problems in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the linguistic deficit definition.
* The generated analysis should be traceable back to the dialogues. You should not hallucinate additional contents not existing in the dialog.
* Keep the summary short and precise.  
* Please fill in the following sheet below for the 13 problems, and paste the sentence from the dialog that you think has the attribute in the following form after :
* If you didn't detect any problem please leave it empty in the following form after :

detecting problem results: 
    Empty speech: 
    Trailing off speech: 
    Circumlocution in speech: 
    Word/phrase revision: 
    Word/phrase repetition: 
    Telegraphic speech: 
    Misuse of pronouns: 
    Poor grammar: 
    Hesitation and pauses:  
    Lack of narrative coherence:  
    Limited recall of details:  
    Simplified Sentence Structure: 
    Difficulty Organizing Description:  

Summary:  

--- 
 

Analyze this Dialogue: 

{dialogue_content}"""

Direct_template['psych_ver_1.1_highcorr']="""Given a participant (PAR) going through with the Cookie Thief test for Alzheimer's disease measurement as the dialogue where an investigator (INV) also engaged, you need to detect if the subject has the following defined linguistic deficits. Analyze the dialogue provided and identify if PAR exhibits any of the linguistic deficit problems mentioned. If problems are detected, fill in the answering sheet accordingly. If not, keep the sheet Only analyze what appears in the dialogue, not the psychological definition. Keep the summary short and precise. 

Defined 13 linguistic deficit problems: 
circumlocution in speech, difficulty Organizing Description, hesitation and pauses, ack of narrative coherence, limited recall of details, simplified Sentence Structure, word/phrase repetition

1. Circumlocution in speech:
• Definition: circumlocution of words/concepts within an utterance. experiencing difficulty finding appropriate words, leading to pauses or circumlocutions.
2. Difficulty Organizing Description:
• Definition: Challenges in arranging information or events coherently and logically when providing a narrative or describing an event. This difficulty may lead to disjointed or fragmented storytelling, making it harder for the listener to follow the sequence of events.
3. Hesitation and pauses:
• Definition: Frequent interruptions in speech with pauses or hesitations.
4. Lack of narrative coherence:
• Definition: refers to a situation where a story lacks clarity, logical progression, or a cohesive structure. struggling to organize 
5. Limited recall of details:
• Definition: Difficulty in remembering and expressing specific details.
6. Simplified Sentence Structure:
• Definition: The use of uncomplicated and straightforward sentence constructions, often with shorter and clearer syntax. This can be a 
7. Word/phrase repetition:
• Definition: same word or phrase is used multiple times within a short span of text

Note.  
* The summary should be consistent with the detected problems in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the linguistic deficit definition.
* The generated analysis should be traceable back to the dialogues. You should not hallucinate additional contents not existing in the dialog.
* Keep the summary short and precise.  
* Please fill in the following sheet below for the 13 problems, and paste the sentence from the dialog that you think has the attribute in the following form after :
* If you didn't detect any problem please leave it empty in the following form after :

detecting problem results: 
    Circumlocution in speech:
    Difficulty Organizing Description:
    Hesitation and pauses:
    Lack of narrative coherence:
    Limited recall of details:
    Simplified Sentence Structure:
    Word/phrase repetition:

Summary:  

--- 
 

Analyze this Dialogue: 

{dialogue_content}"""

Direct_template['psych_ver_1.1_highcorr2']="""Given a participant (PAR) going through with the Cookie Thief test for Alzheimer's disease measurement as the dialogue where an investigator (INV) also engaged, you need to detect if the subject has the following defined linguistic deficits. Analyze the dialogue provided and identify if PAR exhibits any of the linguistic deficit problems mentioned. If problems are detected, fill in the answering sheet accordingly. If not, keep the sheet Only analyze what appears in the dialogue, not the psychological definition. Keep the summary short and precise. 

Defined 6 linguistic deficit problems: 
Circumlocution in speech, Limited recall of details, Trailing off speech


1. Trailing off speech:
• Definition: dropping speech, when the last few words in an utterance become barely audible.
2. Circumlocution in speech:
• Definition: circumlocution of words/concepts within an utterance. experiencing difficulty finding appropriate words, leading to pauses or circumlocutions.
3. Limited recall of details:
• Definition: Difficulty in remembering and expressing specific details.

Note.  
* The summary should be consistent with the detected problems in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the linguistic deficit definition.
* The generated analysis should be traceable back to the dialogues. You should not hallucinate additional contents not existing in the dialog.
* Keep the summary short and precise.  
* Please fill in the following sheet below for the 6 problems, and paste the sentence from the dialog that you think has the attribute in the following form after :
* If you didn't detect any problem please leave it empty in the following form after :

detecting problem results: 
    Circumlocution in speech:
    Limited recall of details:
    Trailing off speech: 


Summary:  

--- 
 

Analyze this Dialogue: 

{dialogue_content}"""

Direct_template['psych_ver_2.1']="""Given a participant (PAR) going through with the Cookie Thief test for Alzheimer's disease measurement as the dialogue where an investigator (INV) also engaged, analyze the dialogue provided to identify positive linguistic attributes exhibited by PAR. If attributes are detected, fill in the answering sheet with extracted sentences from the dialog. If not, keep the sheet empty. 
Defined positive linguistic attributes: Verbose Eloquence, Fluent Continuity, Concise Clarity, Precise Expression, Diverse Lexicon, Articulate Detailing, Pronoun Precision, Grammatical Proficiency, Seamless Fluency, Narrative Fluidity, Detailed Recall, Complex Sentence Structure, Organized Articulation.


1.	Verbose Eloquence:
•	Definition: The ability to express oneself with an extensive vocabulary, providing rich and detailed explanations.
2.	Fluent Continuity:
•	Definition: The capability to articulate thoughts seamlessly, avoiding abrupt interruptions or incomplete sentences.
3.	Concise Clarity:
•	Definition: The skill of expressing ideas clearly and directly without unnecessary elaboration or roundabout language.
4.	Precise Expression:
•	Definition: The capacity to convey thoughts accurately and efficiently, minimizing the need for corrections or revisions.
5.	Diverse Lexicon:
•	Definition: Possessing a varied vocabulary and using different words to convey the same idea, avoiding unnecessary repetition.
6.	Articulate Detailing:
•	Definition: The ability to provide a comprehensive and detailed account using descriptive language and complete sentences.
7.	Pronoun Precision:
•	Definition: Using pronouns accurately and appropriately to enhance clarity and precision in communication.
8.	Grammatical Proficiency:
•	Definition: Demonstrating a strong command of grammar rules, leading to accurate and grammatically sound communication.
9.	Seamless Fluency:
•	Definition: The capability to communicate without frequent interruptions, maintaining a smooth and continuous flow of speech.
10.	Narrative Fluidity:
•	Definition: Crafting a well-structured and coherent narrative that unfolds logically and is easy to follow.
11.	Detailed Recall:
•	Definition: The ability to remember and recount specific details, enhancing the depth and accuracy of the narrative.
12.	Complex Sentence Structure:
•	Definition: Using sophisticated sentence structures to convey nuanced ideas and perspectives, contributing to a more sophisticated form of expression.
13.	Organized Articulation:
•	Definition: The skill of presenting information in a well-organized manner, ensuring clarity and coherence in descriptions.

Note.   
* The summary should be consistent with the detected attributes in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the psychological linguistic attribute definition 
* Keep the summary short and precise.  

* Please fill in the following sheet below for the 13 attributes, and paste the exact sentence from the dialog exhibiting the attribute after the attribute name :. Otherwise, do not paste anything after the attribute name.
* When you are generating each attribute, please check the following rules:
* Please check if the sentences for a detected attribute are exactly from the dialog
* Please check attributes not detected have empty results
* Please check the sheet format is correct 

Detected attributes: 
    Verbose Eloquence: 
    Fluent Continuity:
    Concise Clarity:
    Precise Expression: 
    Diverse Lexicon:
    Articulate Detailing:
    Pronoun Precision:
    Grammatical Proficiency:
    Seamless Fluency:
    Narrative Fluidity:
    Detailed Recall:
    Complex Sentence Structure:
    Organized Articulation:
   
Summary:  

---------
Dialog: 
{dialogue_content}"""


Direct_template['psych_ver_3']="""Given a participant (PAR) undergoing the Cookie Thief test for Alzheimer's disease measurement, the provided dialogue has been analyzed for linguistic deficit (negative attributes) and positive linguistic attributes exhibited by PAR. You need to make a joint assessment based on the detected negative and positive attributes. Please provide key attributes that are most representative of the pattern of this dialogue and provide a summary as well.

Defined linguistic deficit problems: 
empty speech, trailing off speech, circumlocution in speech, word/phrase revision, word/phrase repetition, telegraphic speech, misuse of pronouns, poor grammar, hesitation and pauses, lack of narrative coherence, limited recall of details. 

Defined positive linguistic attributes:
Verbose Eloquence, Fluent Continuity, Concise Clarity, Precise Expression, Diverse Lexicon, Articulate Detailing, Pronoun Precision, Grammatical Proficiency, Seamless Fluency, Narrative Fluidity, Detailed Recall, Complex Sentence Structure, Organized Articulation.


Note. 
* If you do not think the attribute is a key please leave it blank.  
* The summary should be consistent with the detected attributes in the answering sheet. 
* You should double-check the detected key attributes have evidence in the dialogue
* Keep the summary short and precise.  
* Please fill in the following sheet below for the 13 detecting problems and 13 key attributes, and paste the sentence from the dialog that you think has the attribute in the following form after :
* If you didn't detect any problem please leave it empty in the following form after :


detecting problem results: 
    Empty speech: 
    Trailing off speech: 
    Circumlocution in speech: 
    Word/phrase revision: 
    Word/phrase repetition: 
    Telegraphic speech: 
    Misuse of pronouns: 
    Poor grammar: 
    Hesitation and pauses:  
    Lack of narrative coherence:  
    Limited recall of details:  
    Simplified Sentence Structure: 
    Difficulty Organizing Description:  
Key Attributes: 
    Verbose Eloquence: 
    Fluent Continuity: 
    Concise Clarity: 
    Precise Expression: 
    Diverse Lexicon: 
    Articulate Detailing: 
    Pronoun Precision: 
    Grammatical Proficiency: 
    Seamless Fluency: 
    Narrative Fluidity: 
    Detailed Recall: 
    Complex Sentence Structure: 
    Organized Articulation:  


Summary:  

--- 
 

Dialogue: 

{dialogue_content}

{negative_summary}

{positive_summary}"""

Direct_template['psych_ver_3.1.1']=Direct_template['psych_ver_3.1.2']=Direct_template['psych_ver_3']
Direct_template_metadata={}
Direct_template_metadata['psych_ver_3.1.1']={
    'positive_attribute':'Positive_attributes',
    'negative_attribute':'psych_ver_1.1'
    }
Direct_template_metadata['psych_ver_3.1.2']={
    'positive_attribute':'Positive_attributes',
    'negative_attribute':'psych_ver_2.1'
}

Direct_template['RAG3']="""Given a participant (PAR) going through with the Cookie Thief test for Alzheimer's disease measurement as the dialogue where an investigator (INV) also engaged, you need to detect if the subject has the following defined linguistic deficits. Analyze the dialogue provided and identify if PAR exhibits any of the linguistic deficit problems mentioned. If problems are detected, fill in the answering sheet accordingly. If not, keep the sheet Only analyze what appears in the dialogue, not the psychological definition. Keep the summary short and precise. 

Defined 8 linguistic deficit problems: 
Empty speech,  trailing off speech, circumlocution in speech,Word/phrase revision, word/phrase repetition, phonological fragment,Telegraphic speech, misuse of pronuns, poor grammar


1. Empty speech: 
• Definition: Eloquent articulation lacking the expression of meaningful information.
2. Trailing off speech: 
• Definition: dropping speech, when the last few words in an utterance become barely audible.
3. circumlocution in speech: 
• Definition: circumlocution of words/concepts within an utterance.
4. Word/phrase revision: 
• Definition: reviewing and making changes to individual words or phrases within a piece of written content.word/phrase repetition: same word or phrase is used multiple times within a short span of text
5. Telegraphic speech: 
• Definition: a style of communication that is concise and stripped down to the essential words, similar to the simplicity seen in telegrams.
6. Misuse of pronuns: 
• Definition: using pronouns incorrectly in a sentence, which can result in confusion or ambiguity about the intended subject or object.
7. poor grammar: 
• Definition: incorrect or non-standard use of the rules and structures of a language.
        
Note.  
* The summary should be consistent with the detected problems in the answering sheet. 
* You should only analyze what appeared in the dialogue and do not analyze anything that appears in the linguistic deficit definition.
* The generated analysis should be traceable back to the dialogues. You should not hallucinate additional contents not existing in the dialog.
* Keep the summary short and precise.  
* Please fill in the following sheet below for the 8 problems, and paste the sentence from the dialog that you think has the attribute in the following form after :
* If you didn't detect any problem please leave it empty in the following form after :

detected problems:
        Empty speech:
        trailing off speech:
        circumlocution in speech:
        Word/phrase revision:
        Telegraphic speech:
        misuse of pronuns:
        poor grammar:

Summary: 

---


Analyze this Dialogue: 

{dialogue_content}"""

def generate_direct_prompt():
    prompts_dict = {}
    # for instruction_template in instruction_templates:
    for key, values in Direct_template.items():
        prompts_dict[key]=values
    return prompts_dict

# print(assesmentPrompt_template['data_augmentation_attribute'])
targeted_prompt='psych1'

# Usage:
# result_prompts = generate_direct_prompt()
result_prompts = generate_psychology_prompt(assessment_prompt_template=assesmentPrompt_template,
                                            instruction_template=Instruction_templates[targeted_prompt],
                                            psychology_template=Psychology_template,
                                            )
# print(result_prompts[targeted_prompt])

# print(result_prompts['psych_ver_3'])
# print(list(Direct_template.keys()))
