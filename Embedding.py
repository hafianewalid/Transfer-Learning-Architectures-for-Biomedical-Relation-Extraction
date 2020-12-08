"""

This module offering word embedding using BERT model

"""

import torch
import Bert
from Parameters import global_param


bert_type=global_param.model_param['bert']
tokenizer,model=Bert.get_bert(bert_type=bert_type)

def Text2tokens(text):
    """
    This function tokenize the input text
    :param text: input text
    :return: tow list of tokens and segmentation respectively
    """
    marked_text = "[CLS] " + text +" [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    return indexed_tokens,segments_ids

def Token2tonsor(token):
    """
    This function convert the tuple (list tokens , list segmentation) to tuple of torch tensor
    :param token: tuple (list tokens ,list segmentation)
    :return: tuple of tokens and segments tensors respectively
    """
    indexed_tokens,segments_ids = token[0],token[1]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor,segments_tensors

def Bert_Forward(inputs):
    """
    This function forward the input through BERT layers and feedback the activity of each layer
    :param inputs: tuple of Bert inputs (tokens_tensor,segments_tensors)
    :return: list of BERT layers activity
    """
    tokens_tensor,segments_tensors=inputs[0],inputs[1]
    model.eval()
    with torch.no_grad():
        activity_layers, _ = model(tokens_tensor, segments_tensors)
        if(isinstance(activity_layers,torch.Tensor)):
            activity_layers=[activity_layers]
    return activity_layers


def Reshap(activity):
    """
    This function reshape the activity tensor
    layers,batches,tokens,features => tokens,layers,features
    :param activity: the activity tensor
    :return: the activity tensor with shape (tokens,layers,features)
    """
    activity = torch.stack(activity, dim=0)
    activity = torch.squeeze(activity, dim=1)
    activity = activity.permute(1, 0, 2)
    return activity

def Sentence_Embedding(text):
    """
    This function compute the activity of input sentence in Bert network
    :param text: input sentence
    :return: activity tensor corresponding to sentence
    """
    token=Text2tokens(text)
    input=Token2tonsor(token)
    activity=Bert_Forward(input)
    activity=Reshap(activity)
    return activity

def Features_extraction(text_tokens, mode=0):
    """
    This function compute the features corresponding to input tokens using Bert model
    :param text_tokens: list of tokens (indexed_tokens,segments_ids)
    :param mode: strategy used in word embedding
    mode 0 for get last layer activity in bert network as word embedding
    :return: the list of features tensor, each ith item is a features tensor corresponding to ith token
    """
    sentence_embedding=Sentence_Embedding(text_tokens)
    features = []
    if mode==0 :
        for word_embedding in sentence_embedding:
            word_features = word_embedding[-1]
            features.append(word_features)
    return features

def Sentence_Features(text,mode=0):
    """
    This function compute the features corresponding to each word at input sentence
    in order to obtain one tensor features per word the tensor features of tokens belong in the same word are averaged
    :param text: input text
    :param mode: strategy used for word embedding
    mode 0 for get last layer activity in bert network as word embedding
    :return:the list of features tensor, each ith item is a features tensor corresponding to ith word
    """
    text=text.lower()
    tokenized_text = tokenizer.tokenize("[CLS] " + text +" [SEP]")
    token_features = Features_extraction(text, mode)
    sentence_features=[]
    sub_word,b_tokens=False,1
    mean=global_param.corpus_param['token_mean']
    for token,features in zip(tokenized_text,token_features):
        if(mean):
            if(token.startswith("##")):
                sentence_features[-1].add(features)
                nb_tokens+=1
                sub_word = True
            else:
                if(sub_word):
                    sentence_features[-1]/=nb_tokens
                sentence_features.append(features)
                sub_word, nb_tokens = False, 1
        else:
            sentence_features.append(features)


    if(global_param.corpus_param['post_embadding']):
        size = sentence_features[0].size()
        for i in range(global_param.corpus_param['padding_size']-len(sentence_features)):
            sentence_features.append(torch.zeros(size))

    return torch.stack(sentence_features)

def Sentence_1D_Features(text,mode=0):
    """
    this function compute the features corresponding to each word at input sentence
    the output tensor are flattened (the features vector are concatenated into one vector (dim=1) )
    :param text: input text
    :param mode: strategy used for word embedding
    """
    return Sentence_Features(text,mode).view(-1)

def Global_Sentence_Embadding(text,mode):
    tokenized_text = tokenizer.tokenize("[CLS] " + text + " [SEP]")
    token_features = Features_extraction(text, mode)
    return token_features[0]

def get_bert_inputs(text):
    tokenized_text = Text2tokens("[CLS] " + text + " [SEP]")
    inputs=Token2tonsor(tokenized_text)[0].view(-1)
    tokenized_text = tokenizer.tokenize("[CLS] " + text +" [SEP]")
    outind=[0,0,0,0]
    j=0
    v=False
    u=False

    v1=False
    u1=False
     
 
    for i in tokenized_text:
        #print(i)
        if('<'in i and v and outind[0]==0):
            outind[0]=j-1
        if('[' in i and u and outind[2]==0):  
            outind[2]=j-1
        v='<' in i
        u='[' in i

        if('>'in i and v1 and outind[1]==0):
            outind[1]=j

        if(']' in i and u1 and outind[3]==0):  
            outind[3]=j
        v1='>' in i
        u1=']' in i

        j+=1

    if(outind[3]+outind[2]==0):
       outind[3]=outind[1]
       outind[2]=outind[0]

    if(outind[1]+outind[0]==0):
       outind[1]=outind[3]
       outind[0]=outind[2]
    
   
    return inputs,torch.tensor(outind)



