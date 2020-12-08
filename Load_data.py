import glob
import os
from torch.utils import data
import collections
from torch.utils.data import Dataset
from tqdm import tqdm
from Corpus import Corpus
from Corpus import string_normaliz
from Embedding import *
from Parameters import global_param

class Data_set(Dataset):
    '''
    Class of data-set of features and their labels
    '''
    def __init__(self, X, Y):
        '''
        data-set corpus
        :param X: features vector
        :param Y: labels
        '''
        self.X = X
        self.Y = Y
        # print("data", X[0].size())

    def __getitem__(self, index):
        '''
        getter provide the i-th element
        :param index: i index of element
        :return: the i-th first, second input and label
        '''
        return [self.X[index][0], self.X[index][1], self.Y[index]]

    def __len__(self):
        '''
        length function
        :return:size of corpus
        '''
        return len(self.Y)


def collate(batch):
    '''
    Collate function
    :param batch: batch
    :return: tuple of tensors (input1 tensor ,input2 tensor ,labels)
    '''
    input1 = [item[0] for item in batch]
    input1 = torch.nn.utils.rnn.pad_sequence(input1, batch_first=True)

    input2 = torch.stack([item[1] for item in batch])

    targets = [item[2] for item in batch]
    targets = torch.tensor(targets)

    return input1, input2, targets


def torch_loader(X, Y, shuffle=True, batch_size=32):
    '''
    This function provide a torch loader
    :param X: the list of input data
    :param Y: the list of labels
    :param shuffle: shuffling or not
    :param batch_size: size of batch
    :return:torch DataLoader
    '''
    corpus = Data_set(X, Y)
    loader = data.DataLoader(corpus, shuffle=shuffle, batch_size=batch_size, collate_fn=collate, pin_memory=True)

    return loader



def indx_entity(sentence, entity):
    '''
    This function provide a list of index which the entity occurrences
    :param sentence: whole sentence
    :param entity: the entity
    :return: list of index occurrences
    '''
    indx = []
    words = string_normaliz(sentence).split()
    for e in string_normaliz(entity).split():
        for w in words:
            if (w.find(e) > -1):
                indx.append(words.index(w))
    if (len(indx) == 0):
        return None
    else:
        return indx


def remove_sep(s):
    '''
    This function remove all separation
    :param s: string
    :return: input string with out separation
    '''
    s = ''.join(s.split())
    return s


def relative_pos(sentence, entity):
    '''
    This function compute relative positions vector
    (relatives distances between words in sentence and the entity)
    :param sentence: input sentence
    :param entity: input entity
    :return: relative positions vector
    '''
    ref = indx_entity(sentence, entity)[0]
    dis_vector = []
    words = sentence.split()
    sentence_len = len(words)
    for wp in range(sentence_len):
        pos = (ref - wp) / sentence_len
        dis_vector.append(pos)
    padd=global_param.corpus_param['padding_size']
    dis_vector += [0] * (padd-len(dis_vector))
    return dis_vector


def entity_features(entity1, entity2, sentence):
    '''
    This function compute entities features
    :param entity1:the first entity
    :param entity2:the second entity
    :param sentence:the whole sentence
    :return:tensor of entity features
    '''
    entity1, entity2 = string_normaliz(entity1), string_normaliz(entity2)

    pos = relative_pos(sentence, entity1)
    pos.extend(relative_pos(sentence, entity2))

    pos = torch.tensor(pos)
    return pos

def corpus_type(corpora):
    type_cor=1 if global_param.corpus_param['corpus_src']==corpora else 0
    return torch.tensor([type_cor])

def Save_Featurs(X,Y,tag):
    '''
    This function save features data
    :param X: list of inputs tensors
    :param Y: list of labels tensors
    :param tag: the tag according to data set
    '''
    for i in range(len(Y)):
        torch.save(X[i],tag+'/X'+str(i)+'.f')
        torch.save(Y[i],tag+'/Y'+str(i)+'.f')

def Load_Featurs(tag):
    '''
    This function load features data
    :param tag: the tag of data set
    :return: list of inputs and labels
    '''
    X,Y=[],[]
    corpus_size=int(len(glob.glob(tag+"/*.f"))/2)
    pbar = tqdm(total=corpus_size, desc="Features Loading : ")
    for i in range(corpus_size):
        X.append(torch.load(tag+'/X'+str(i)+'.f'))
        Y.append(torch.load(tag+'/Y'+str(i)+'.f'))
        pbar.update(1)
    pbar.close()
    return X,Y

def Corpus_Loading(path, name='snpphena'):
    """
    This function load data-set
    :param path: the path of data-set
    :param name: the name of data set
    :return: list of input features and their labels
    """

    bert=global_param.model_param['bert']
    finetuning ='' if not global_param.model_param['fine_tuning'] else 'fine_tuning'

    Features_dir ="./Features"
    if not os.path.exists(Features_dir):
        os.mkdir(Features_dir)

    corpus = Corpus(path, name)

    Features_corpus_dir = "./Features/"+name
    if not os.path.exists(Features_corpus_dir):
        os.mkdir(Features_corpus_dir)

    tag=path.replace('/','_')+'_'+finetuning+'_'+bert
    if not os.path.exists(Features_corpus_dir+"/"+tag):
        os.mkdir(Features_corpus_dir+"/"+tag)

        dataset_X, dataset_Y_Name = corpus.get_data()

        dataset_XF, dataset_Y = [], []

        pbar = tqdm(total=len(dataset_Y), desc="Features Computing : ")
        for X in dataset_X:
            sentence, entity1, entity2 = X[0], X[1], X[2]

            #FX = Sentence_Features(sentence), entity_featurs(entity1, entity2, sentence)
            ind1, ind2 = indx_entity(sentence, entity1), indx_entity(sentence, entity2)

            sentence_=sentence
            if(global_param.corpus_param['annonimitation']):
                masks=global_param.corpus_param['entitys_masks']
                sentence_=sentence.replace(entity1,masks[0])
                sentence_=sentence_.replace(entity2,masks[1])

            if (global_param.corpus_param['encapculate']):
                items = global_param.corpus_param['encapsulate_items']
                sentence_ = sentence.replace(entity1,items[0]+entity1+items[1])
                sentence_ = sentence_.replace(entity2,items[2]+entity2+items[3])
                print(sentence_)

            if(finetuning==''):
                FX = Sentence_Features(sentence_, remove_e=False, inde1=ind1, inde2=ind2), corpus_type(name)
            else:
                FX = get_bert_inputs(sentence_)#,type_corpora(name)

            dataset_XF.append(FX)

            pbar.update(1)

        pbar.close()

        Association_type = corpus.Association_type
        for e in dataset_Y_Name:
            dataset_Y.append(Association_type[string_normaliz(e)])

        Save_Featurs(dataset_XF,dataset_Y,Features_corpus_dir+"/"+tag)

    else:

        dataset_XF, dataset_Y=Load_Featurs(Features_corpus_dir+"/"+tag)

    Nb_class = corpus.nb_association
    print("Corpus {} loaded ".format(name))
    print(" NB Class : {} \n NB Relation : {}".format(Nb_class, len(dataset_Y)))
    print(" class size ")

    counter = collections.Counter(dataset_Y)
    for i in range(Nb_class):
        print("       C{} [ {} ] ".format(i, counter[i]))

    return dataset_XF, dataset_Y, Nb_class

