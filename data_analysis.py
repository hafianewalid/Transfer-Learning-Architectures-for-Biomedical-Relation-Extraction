import argparse
import pylab
from Corpus import *
from Load_data import indx_entity
from Analysis_plot import *
import pandas as pd
from stop_words import get_stop_words
from Bert import get_bert


def get_file(path,target):
    ann_txt_files = [(f.split(path)[1], (f.split(path)[1]).split('ann')[0] + "txt") for f in glob.glob(path + "/*.ann")]
    for ann, txt in ann_txt_files:
        sentence = load_txt(path + txt)
        if(sentence==target):
            return txt,ann

def get_rel_class(corpus):
    X, Y = corpus.data
    class_ = {}
    for i in zip(X, Y):
        class_[i[1]] = ''
    return class_.keys()

def get_entitys_types(corpus):
    X, Y = corpus.data
    class_ = {}
    for i in zip(X, Y):
        class_[i[0][1]] = ''
        class_[i[0][2]] = ''
    return class_.keys()

def get_vocab(corpus):
    X, Y = corpus.data
    Voc={}
    Voc_rel={}
    for i in get_rel_class(corpus):
        Voc_rel[i]={}
    for i in zip(X,Y):
        for a in i[0][0].split():
            j=string_normaliz(a)
            if(j.lower() not in Voc):
                Voc[j.lower()]=0
            Voc[j.lower()]+=1
            if(j.lower() not in Voc_rel[i[1]]):
                Voc_rel[i[1]][j.lower()]=0
            Voc_rel[i[1]][j.lower()]+=1
    return Voc,Voc_rel

def Histograme(corpus):
    histo={}
    for i in get_rel_class(corpus):
        histo[i]=0
    X,Y = corpus.data
    for i in zip(X, Y):
        histo[i[1]]+=1
    return histo

def Dist(corpus):
    X,Y = corpus.data
    dist={}
    for i in zip(X,Y):
        ind1,ind2=indx_entity(i[0][0],i[0][1])[-1],indx_entity(i[0][0],i[0][2])[0]
        if(i[1] not in dist):
            dist[i[1]]=[]
        dist[i[1]].append(max(ind1,ind2)-min(ind1,ind2))
    return dist


def multi_label(corpus):
    X,Y = corpus.data
    for i in zip(X,Y):
        s1,e1,e2,l=i[0][0],i[0][1],i[0][2],i[1]
        for j in zip(X,Y):
            s2,e1j, e2j, lj =j[0][0],j[0][1], j[0][2], j[1]
            if(e1==e1j and e2==e2j and l!=lj and s1==s2):
                print("\n\n s1{} \n s2{} \n   e1 : {} e2 : {} \n     =>>>>> l1 : {} l2 : {} ".format(s1,s2,e1,e2,l,lj))
                #print(get_file(path_train,s1))

corpus_path={
'pgx':"data/PGxCorpus/",
'semeval':"data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
'snpphena':"data/SNPPhenA/SNPPhenA_BRAT/Train/",
'chemprot':"data/chemprot/train.txt"
}

def analys(corpus_name):

    corpus = Corpus(corpus_path[corpus_name],corpus_name)
    corpus.get_data()

    path="plot/"
    X,Y = corpus.data
    print("size",len(Y))


    circle_plot(Histograme(corpus),path+"/"+corpus_name+"/",title=corpus_name+" : distribution of relationships")

    st=get_stop_words('en')
    st.extend(string.punctuation)
    st.extend([str(i) for i in range(10)])
    def rm_stop_words(dic):
        for i in st:
            if i in dic:
                dic[i]=0
        return dic


    vocab,vocab_rel=get_vocab(corpus)
    vocab['']=0
    vocab = rm_stop_words(vocab)
    H=pd.DataFrame.from_dict(vocab,orient='index').nlargest(20,0).to_dict()[0]
    histo(H,path+"/"+corpus_name+"/",title=corpus_name+" Histo")

    for i in get_rel_class(corpus):
        vocab=vocab_rel[i]
        vocab[''] = 0
        vocab=rm_stop_words(vocab)
        for k in H:
            if k in vocab:
                vocab[k]=0
        Hi = pd.DataFrame.from_dict(vocab, orient='index').nlargest(20,0).to_dict()[0]
        histo(Hi,path+"/"+corpus_name+"/",title=corpus_name+" relation "+i+" Histo")


    dist=Dist(corpus)
    box(dist,path+"/"+corpus_name+"/",title=corpus_name+" distances")


    dist=Dist(corpus)
    mean_frame,std=[],[]
    for rel in dist.keys():
     df=pd.DataFrame.from_dict({rel:dist[rel]})
     mean_frame.append(df.mean())
     std.append(df.std())

    mean=pd.DataFrame(pd.concat(mean_frame),columns=["mean"])
    std=pd.DataFrame(pd.concat(std),columns=["std"])
    res=pd.concat((mean,std),axis=1)


    data={'sentence length':[],'Vocab':[],'tokenisation length':[]}
    tokenizer_bert,_=get_bert()
    tokenizer_scibert,_=get_bert(bert_type='scibert')

    for x in X:
     data['sentence length'].append(len(x[0].split(' ')))
     data['Vocab'].append('BERT VOCAB')
     data['tokenisation length'].append(len(tokenizer_bert.tokenize(x[0])))

     data['sentence length'].append(len(x[0].split(' ')))
     data['Vocab'].append('SciBERT VOCAB')
     data['tokenisation length'].append(len(tokenizer_scibert.tokenize(x[0])))


    data=pd.DataFrame(data)
    data=data.sort_values(by=['sentence length'])
    print(data)

    title=corpus_name+" tokenisation analysis"


    plt.rcParams["figure.figsize"] = (9,9)

    pylab.mpl.style.use('seaborn')

    g = sns.relplot(x="sentence length", y="tokenisation length",hue="Vocab",style="Vocab",
                    hue_order=['SciBERT VOCAB','BERT VOCAB'],kind="line",data=data
                    ,col_order = ['SciBERT VOCAB', 'BERT VOCAB']
                    ,style_order = ['SciBERT VOCAB', 'BERT VOCAB']
                    )
    sns.despine()
    plt.title(title)

    plt.show()
    plt.savefig(title+".png")

parser = argparse.ArgumentParser()
parser.add_argument('-corpus', default='chemprot', choices=['chemprot', 'pgx'],
                            dest='corpus',
                            help='')
param = parser.parse_args()
analys(param.corpus)