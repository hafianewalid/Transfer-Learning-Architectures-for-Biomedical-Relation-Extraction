import glob
import re
import string
import jsonlines
import csv

"""
This module is used to load whole text data of corpus   
"""

Corpus_Association_type = {
    'snpphena_label':{
        "moderateconfidenceassociation":'confidenceassociation',
        "weakconfidenceassociation":'confidenceassociation',
        "strongconfidenceassociation": 'confidenceassociation',
        "negativeassociation":"negativeassociation",
        "neutralassociation":"neutralassociation"
    },
    'snpphena': {
        "moderateconfidenceassociation": 0,
        "weakconfidenceassociation": 0,
        "strongconfidenceassociation": 0,
        "negativeassociation": 1,
        "neutralassociation": 2
    },

    'semeval': {
        "CauseEffect": 0,
        "InstrumentAgency": 1,
        "ProductProducer": 2,
        "ContentContainer": 3,
        "EntityOrigin": 4,
        "EntityDestination": 5,
        "ComponentWhole": 6,
        "MemberCollection": 7,
        "MessageTopic": 8,
        "Other": 9
    },
    'chemprot': {
        'AGONISTINHIBITOR': 0,
        'INHIBITOR': 1,
        'PRODUCTOF': 2,
        'SUBSTRATEPRODUCTOF': 3,
        'INDIRECTDOWNREGULATOR': 4,
        'INDIRECTUPREGULATOR': 5,
        'SUBSTRATE': 6,
        'ANTAGONIST': 7,
        'ACTIVATOR': 8,
        'DOWNREGULATOR': 9,
        'AGONISTACTIVATOR': 10,
        'UPREGULATOR': 11,
        'AGONIST': 12
    },
    'pgx':{
    'increases': 0,
    'isEquivalentTo': 1,
    'treats': 2,
    'decreases': 3,
    'influences': 4,
    'causes': 5,
    'isAssociatedWith': 6
     }


}

Corpus_NB_Association = {
    'snpphena': 3,
    'semeval': 10,
    'chemprot': 13,
    'pgx':7
}


def remove(s,l):
    '''
    :param s: String
    :param l: list of characters
    :return: input String without characters that belong to l
    '''
    for i in l:
        s = s.replace(i, "")
    return s

def chunk(data,k):
    fold_size = len(data[0])/float(k)
    k_X,k_Y = [],[]
    last = 0.0
    while last < len(data[0]):
        k_X.append(data[0][int(last):int(last +fold_size)])
        k_Y.append(data[1][int(last):int(last +fold_size)])
        last += fold_size
    return k_X,k_Y

punctuation = string.punctuation + '\t'
punctuation1=['(',')',':',';',',','?','!','.','%','*','+','=','"','#','~','@','$','0','1','2','3','4','5','6','7','8','9','^','{','}','-','_']

def string_normaliz(s,mode=0):
    '''
    This function remove the punctuation
    :param s: input string
    :return: string without punctuation
    '''
    l = punctuation if mode==0 else punctuation1
    s = s.translate(str.maketrans('', '',l))
    return s

def load_txt(txt_path):
    '''
    :param txt_path: path to file
    :return: the text inside the file
    '''
    txt = open(txt_path).read().replace('\n',"")
    return txt


def load_ann(ann_path):
    '''
    :param ann_path: path to annotation file, this file must be in BRAT format
    :return: tuple contains the first entity, the second one and the label according the relationship
    '''
    ann = open(ann_path).read().split('\n')
    entitys1, entitys2, labels = [], [], []
    T = [i for i in ann if i.startswith('T')]
    R = [i for i in ann if i.startswith('R')]
    T_dic={}
    for i in T:
        indt = int((re.findall('T[0-9]*',i)[0])[1:])
        T_dic[indt]=i


    for e in R:

        eind = re.findall('Arg.:T[0-9]*', e)
        label = e.split(' ')[0][3:]
        e1, e2 = int(re.findall('[0-9]*',eind[0])[-2]),int(re.findall('[0-9]*',eind[1])[-2])

        entity1, entity2 = T_dic[e1].split('\t')[-1], T_dic[e2].split('\t')[-1]
        # print(entity1,entity2,label)
        entitys1.append(entity1)
        entitys2.append(entity2)
        labels.append(string_normaliz(label))
    return entitys1, entitys2, labels


def brat(path):
    '''
    This function load BRAT data
    :param path: path to train or test data
    :return: tuple of two list, the first one is samples list (sentences,entity1,entity2) and the second is labels list
    '''
    ann_txt_files = [(f.split(path)[1], (f.split(path)[1]).split('ann')[0] + "txt") for f in glob.glob(path + "/*.ann")]

    Dataset_X = []
    Dataset_Y_Name = []

    for ann, txt in ann_txt_files:
        sentence = load_txt(path + txt)
        entitys1, entitys2, labels = load_ann(path + ann)
        for entity1, entity2, label in zip(entitys1, entitys2, labels):
            while(entity1 not in sentence):
                entity1=entity1[:-1]
            while(entity2 not in sentence):
                entity2=entity2[:-1]
            Dataset_X.append((sentence, entity1, entity2))
            Dataset_Y_Name.append(label)

    return Dataset_X, Dataset_Y_Name



def semeval(path):
    '''
    This function load SemEval2010 data in text format
    :param path: path to train or test data
    :return: tuple of two list, the first one is samples list (sentences,entity1,entity2) and the second is labels list
    '''
    txt = open(path).read()
    lines = txt.split('\n')
    c = ["<e1>", "<e2>", "</e1>", "</e2>"]

    Dataset_X = []
    Dataset_Y_Name = []
    for i in range(0, len(lines) - 1, 4):
        sentence = remove(lines[i], c).split('\t')[1]
        entity1 = remove(re.findall('<e1>.*</e1>', lines[i])[0], c)
        entity2 = remove(re.findall('<e2>.*</e2>', lines[i])[0], c)
        rel = lines[i + 1].split('(')
        if (len(rel) > 1 and rel[1].index('1') > rel[1].index('2')):
            entity1, entity2 = entity2, entity1
        label = rel[0]
        Dataset_X.append((sentence, entity1, entity2))
        Dataset_Y_Name.append(label)

    return Dataset_X, Dataset_Y_Name


def chemprot(file_path):
    c1 = ['<<', '>>', '[[', ']]']
    Dataset_X = []
    Dataset_Y_Name = []
    with jsonlines.open(file_path) as f_in:
        for json_object in f_in:
            text = json_object.get('text'),
            label = json_object.get('label')
            entity1 = re.findall('<<.*>>', text[0])[0]
            entity2 = re.findall('\[\[.*\]\]', text[0])[0]
            #sentence = remove(text[0], c1)
            sentence = text[0]
            Dataset_X.append((sentence, entity1, entity2))
            Dataset_Y_Name.append(label)
    return Dataset_X, Dataset_Y_Name


class Corpus():
    '''
    Class of corpus contains corpus attributes and allows to load text data
    '''

    def __init__(self, path, name):
        '''
        Corpus constructor
        :param path: path to train or test data
        :param name: name of corpus this param is used to select the text loader corresponding to corpus
        '''
        self.Association_type = None
        self.nb_association = None
        self.path = path
        self.name = name
        self.data = None
        if name in Corpus_Association_type:
            self.Association_type = Corpus_Association_type[name]
        if name in Corpus_NB_Association:
            self.nb_association = Corpus_NB_Association[name]

    def get_data(self):
        """
        this function load text data from a corpus
        :return: text data according to corpus
        """
        if self.name == 'pgx':
            self.data = brat(self.path)

        if self.name == 'snpphena':
            self.data = brat(self.path)

        if self.name == 'snpphena_label':
            self.data = brat(self.path)

        if self.name == 'semeval':
            self.data = semeval(self.path)

        if self.name == 'chemprot':
            self.data = chemprot(self.path)

        return self.data

    def anonymization(self,word1="@GENE$",word2="@DISEASE$"):
        """
        this function apply an entities anonymization
        :param word1 the word witch the first entity be replaced
        :param word2 the word witch the second entity be replaced
        :return: the data with the anonymous entities
        """
        Dataset_X, Dataset_Y_Name= self.data
        for i in range(len(Dataset_X)):
            Dataset_X[i]=(Dataset_X[i][0].replace(Dataset_X[i][1],word1)).replace(Dataset_X[i][2],word2)

        self.data=Dataset_X, Dataset_Y_Name
        return  self.data

    def transform(self,format='tsv',type='train',path=""):
        """
        this function transform corpus data to a specific format
        """
        if(format=='tsv'):
            Dataset_X, Dataset_Y_Name = self.data
            Dataset_Y = [ self.Association_type[string_normaliz(i)] for i in Dataset_Y_Name ]
            with open(path+type+'.tsv','w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                lines=(Dataset_X,Dataset_Y)
                if(type=='test'):
                    lines=(range(len(Dataset_Y)),Dataset_X,Dataset_Y)
                    writer.writerow(('index','sentence','label'))
                for x in zip(*lines):
                    writer.writerow(x)


    def slices(self,k):
        '''
        this function slices corpus on n fold
        '''
        sub_copus=[]
        data=chunk(self.data,k)
        for d in data:
            c=Corpus(self.path,self.name)
            c.data=d
            sub_copus.append(c)
        return sub_copus







