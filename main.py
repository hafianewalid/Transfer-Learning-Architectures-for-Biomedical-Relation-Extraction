from sklearn.metrics import classification_report, precision_recall_fscore_support
import Cross_validation
from Load_data import Corpus_Loading
from Models import *
from Train import train_save, prediction
import os
from Parameters import global_param


Recovery = False
id_rec = 0
valid_perc =0
do_valid=True
fold_num=5
do_cross_valid=False

nb_epoch = global_param.traning_param['num_ep']
corpus=global_param.corpus_param['corpus']
corpus_src=global_param.corpus_param['corpus_src']
lr=global_param.traning_param['lr']
bert_type=global_param.model_param['bert']

do_transfer=corpus_src!=''

F_type=global_param.traning_param['F_type']
exp_name=global_param.traning_param['exp_tag']+corpus


machine_name = os.uname()[1]
path_train,path_valid,path_test=None,None,None
X_valid,Y_valid=[],[]
X_test,Y_test=[],[]

if(corpus=='semeval'):
    path_train="data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    path_test="data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
if(corpus=='snpphena'):
    path_train="data/SNPPhenA/SNPPhenA_BRAT/Train/"
    path_test="data/SNPPhenA/SNPPhenA_BRAT/Test/"
if(corpus=='chemprot'):
    path_train='data/chemprot/train.txt'
    path_valid='data/chemprot/dev.txt'
    path_test='data/chemprot/test.txt'
if(corpus=='pgx'):
    path_train='data/PGxCorpus'


path_train_src=''
if(corpus_src=='semeval'):
    path_train_src="data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
if(corpus_src=='snpphena'):
    path_train_src="data/SNPPhenA/SNPPhenA_BRAT/Train/"
if(corpus_src=='chemprot'):
    path_train_src='data/chemprot/train.txt'
if(corpus_src=='pgx'):
    path_train_src='data/PGxCorpus'

X_train_src,Y_train_src,Nb_class_src=[],[],0
if(do_transfer):
    print(" ########### Loading src corpora data ")
    X_train_src, Y_train_src, Nb_class_src = Corpus_Loading(path_train_src, name=corpus_src)

print(" ########### Loading training data ")
X_train, Y_train, Nb_class = Corpus_Loading(path_train, name=corpus) 
#loader_app=torch_loader(X_app,Y_app,batch_size=10)
if(path_valid!= None):
    print(" ########### loading validation data ")
    X_valid,Y_valid, Nb_class = Corpus_Loading(path_valid, name=corpus)
if(path_test!=None):
    print(" ########### loading test data ")
    X_test,Y_test, Nb_class = Corpus_Loading(path_test, name=corpus)

if(valid_perc==0 and X_valid==[]):
    X_valid,Y_valid=X_train,Y_train
    do_valid = False

do_cross_valid=Y_test==[]
path=None

def Experence():

    global Y_test
    print("///////////////////////////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////////////////////////")
    print("/////////////////////////         Model          ///////////////////////////////")

    model=None
    if(global_param.model_param['fine_tuning']):
     if(global_param.finetuning_model=='MLP'):
         model=Bert_finetuning(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
     if (global_param.finetuning_model=='CNN'):
         model=Bert_finetuning_CNN(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
     if (global_param.finetuning_model=='CNN2'):
         model=Bert_finetuning_CNN2(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
     if (global_param.finetuning_model =='CNN_residual'):
         model=Bert_finetuning_CNN_residual(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
     if (global_param.finetuning_model == 'mixed'):
         model=Bert_finetuning_Mixed(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
     if (global_param.finetuning_model == 'CNN_seg'):
         model=Bert_finetuning_CNN_Entity(out=Nb_class,out_src=Nb_class_src,bert_type=bert_type)
    else:
     if(global_param.frozen_model=='RNN'):
         model=RNN(out=Nb_class)
     if(global_param.frozen_model=='CNN'):
         model=CNN(inc=768,out=Nb_class)
     if(global_param.frozen_model=='CNN_RNN'):
         model=RNN_CNN(out=Nb_class)
     if (global_param.frozen_model=='CNN_RNN_parell'):
         model=RNN_CNN(out=Nb_class,seq=False)


    
    print(model)

    model.to(global_param.device)

    train_param = {
        'model': model,
        'X_train': X_train,
        'Y_train': Y_train,
        'X_train_src': X_train_src,
        'Y_train_src': Y_train_src,
        'nb_epoch': nb_epoch,
        'recovery': Recovery,
        'recovery_id': id_rec,
        'percentage': valid_perc,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'F_type': F_type,
        'lr': lr,
        'do_valid': do_valid
    }

    if not do_cross_valid:
        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         Training       ///////////////////////////////")

        best_model,path=train_save(**train_param)

        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         TEST          ///////////////////////////////")

        pred=prediction(best_model,X_test)

    else:

        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         CROSS RESULT          ///////////////////////////////")

        pred,Y_test=Cross_validation.cross_validation(train_param,train_save,fold_num)


    raport = classification_report(y_pred=pred, y_true=Y_test)

    print(raport)

    file = open("result_" + machine_name + "_" + F_type + ".pred", "a+")
    y,p="",""
    for xp,xy in zip(pred,Y_test):
        y+=' '+str(xy)
        p+=' '+str(xp)
    print("Y"+y+'\nP'+p, file=file)
    file.close()


    return precision_recall_fscore_support(y_pred=pred, y_true=Y_test, average=F_type),raport


for k in range(10):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     Experence {}      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(k))

    raport,raport_det=Experence()
    file = open(exp_name+"result_"+machine_name+"_"+F_type+".res","a+")
    print(str(raport[0])+" "+str(raport[1])+" "+str(raport[2]), file=file)
    file.close()

    file = open(exp_name + "result_" + machine_name + "_" + F_type + ".det", "a+")
    print(str(raport_det), file=file)
    file.close()
    #os.rmdir(path)


