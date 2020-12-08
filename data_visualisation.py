import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Corpus import Corpus_Association_type
from Load_data import Corpus_Loading
from Models import Bert_finetuning
from Analysis_plot import Data_visual
from Train import *

path_train={
'semeval':"data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
'snpphena':"data/SNPPhenA/SNPPhenA_BRAT/Train/",
'chemprot':'data/chemprot/train.txt',
'pgx':'data/PGxCorpus'
}
ep=5

def embedded_visual(X,Y,ep='',bert=global_param.model_param['bert'],corpus=global_param.corpus_param['corpus']):


    X = np.array([i[0][0].numpy() for i in X])

    label = Corpus_Association_type[corpus]
    label = list(label.keys())
    if (corpus == 'snpphena'):
        label = ['confidenceassociation', 'negativeassociation', 'neutralassociation']

    '''
    oneC = 2
    Y = [0 if y == oneC else 1 for y in Y]
    label = [label[oneC], 'other']
    '''

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    df1 = pd.DataFrame(X, columns=[str(i) for i in range(len(X[0]))])
    df1['x1'] = pca_result[:, 0]
    df1['x2'] = pca_result[:, 1]
    df1['y'] = [label[i] for i in Y]
    df1['Methode'] = ['PCA' for i in Y]
    
    #tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=400)
    tsne_results = tsne.fit_transform(X)

    df2 = pd.DataFrame(X, columns=[str(i) for i in range(len(X[0]))])
    df2['x1'] = tsne_results[:, 0]
    df2['x2'] = tsne_results[:, 1]
    df2['y'] = [label[i] for i in Y]
    df2['Methode'] = ['T-SNE' for i in Y]

    #tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=300)
    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000, learning_rate=400)
    tsne_pca_results = tsne.fit_transform(pca_result)

    df3 = pd.DataFrame(X, columns=[str(i) for i in range(len(X[0]))])
    df3['x1'] = tsne_pca_results[:, 0]
    df3['x2'] = tsne_pca_results[:, 1]
    df3['y'] = [label[i] for i in Y]
    df3['Methode'] = ['T-SNE on PCA' for i in Y]


    df = [df1,df2,df3]
    df = pd.concat(df)

    Data_visual(df, corpus+ep, bert, label)

def Visualisation(bert,corpus):

    global_param.model_param['fine_tuning'] = False
    global_param.model_param['bert'] = bert

    X,Y,Nb_class = Corpus_Loading(path_train[corpus],name=corpus)
    embedded_visual(X, Y, ep='0',corpus=corpus)

def Visual_ep(model,X,Y,ep,bert='',corpus=''):
    X_bert=[]
    for x in X:
        input1 = torch.stack([x[0]])
        input1=input1.to(global_param.device)
        with torch.no_grad():
            model.eval()
            activity_layers, _ = model.bert_model(input1)
            X_bert.append(activity_layers)

    embedded_visual(X_bert,Y,str(ep),bert=bert,corpus=corpus)


def Visualisation_ep(bert,corpus):

 print("#########################################")
 print("Visualisation of "+corpus+" Using "+bert)     
 print("#########################################")

 global_param.model_param['fine_tuning'] = True
 global_param.model_param['bert'] = bert

 X,Y,Nb_class = Corpus_Loading(path_train[corpus],name=corpus)

 loader_app = torch_loader(X,Y, batch_size=32)
 model=Bert_finetuning(out=Nb_class,out_src=0,bert_type=bert)
 model.to(global_param.device)

 f_loss = torch.nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.05, amsgrad=False)

 Visual_ep(model,X,Y,0,bert=bert,corpus=corpus)

 for i in range(ep):
    loss ,acc = train(model,loader_app, f_loss, optimizer)
    Visual_ep(model,X,Y,i+1,bert=bert,corpus=corpus)



#############################################

corpus=global_param.corpus_param['corpus']
bert=global_param.model_param['bert']
Visualisation_ep(bert,corpus)

#############################################

