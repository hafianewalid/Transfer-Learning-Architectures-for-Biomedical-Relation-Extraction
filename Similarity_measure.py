from sklearn.cluster import KMeans
from Load_data import Corpus_Loading
from Analysis_plot import *
from Parameters import global_param


path_train={
'semeval':"data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
'snpphena':"data/SNPPhenA/SNPPhenA_BRAT/Train/",
'chemprot':'data/chemprot/train.txt',
'pgx':'data/PGxCorpus'
}

bert_list=['bert','biobert','scibert']

corpus1,corpus2='snpphena','chemprot'

def cleustring(corpus1,corpus2,K,bert):
    path_train1, path_train2 = path_train[corpus1], path_train[corpus2]

    global_param.model_param['fine_tuning'] = False
    global_param.model_param['bert'] = bert

    X_app1, Y_app, Nb_class = Corpus_Loading(path_train1, name=corpus1)
    X_app1 = [i[0][0].numpy() for i in X_app1]

    X_app2, Y_app, Nb_class = Corpus_Loading(path_train2, name=corpus2)
    X_app2 = [i[0][0].numpy() for i in X_app2]

    kmeans = KMeans(n_clusters=K, random_state=0).fit(X_app1)
    prototypes1 = kmeans.cluster_centers_

    kmeans = KMeans(n_clusters=K, random_state=0).fit(X_app2)
    prototypes2 = kmeans.cluster_centers_

    return prototypes1,prototypes2

def box_space(prototypes):
    max_vect=np.max(prototypes,axis=0)
    min_vect=np.min(prototypes,axis=0)
    delta=(max_vect-min_vect)/2
    center=min_vect+delta
    R=np.linalg.norm(center-min_vect)
    return center,R

def dist(V1,V2,R):
    d=np.linalg.norm(V1-V2)
    a=0 if d < R else d
    return a

def similarity(corpus1,corpus2,K,bert):
    p1,p2=cleustring(corpus1,corpus2,K,bert)
    c1,R1=box_space(p1)
    S=np.mean([dist(c1,x,R1) for x in p2])
    return S



corpora_list=path_train.keys()
K=7
for bert in bert_list:
    mat=[]
    for c1 in corpora_list:
        l=[round(np.mean(np.array([similarity(c1,c2,K,bert) for v in range(10)])),3) for c2 in corpora_list ]
        mat.append(l)
    plot_heatmap(corpora_list, bert,np.array(mat))


