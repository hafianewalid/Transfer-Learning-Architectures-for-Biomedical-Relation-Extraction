import copy
import os
import torch
from sklearn.metrics import classification_report, f1_score
from Parameters import global_param


def printf(txt,path,display=True):
    """
    This function display and save input string
    :param txt: input string
    :param path: path of saving file
    :param display: True for displaying
    """
    file = open(path,"a+")
    print(txt,file=file)
    file.close()
    if(display):
        print(txt)


def generate_unique_logpath(logdir, raw_run_name):
    """
    This function get unique key in the input folder
    :param logdir: folder
    :param raw_run_name: run name
    :return: the unique path
    """
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:
    """
    This class is used as check point in training
    """

    def __init__(self,filepath,model,valid_indx=None,F_type='macro',save=False):
        '''
        The Checkpoint's constructor
        :param filepath: path according to running experience
        :param model: the model
        :param valid_indx: list of validation index if we use subpart of training set as validation set
        '''
        #self.min_loss = None
        self.best_f=None
        self.filepath = filepath
        self.model = model
        self.save=save
        if(not save):
            self.best_model=copy.deepcopy(model)
        self.f_type=F_type
        if(valid_indx!=None):
            s=""
            for i in valid_indx:
                s+=str(i)+" "
            printf(s,self.filepath+"/recovery.rec",display=False)

    def update(self,pred,Y,epoch,loss,acc,do_valid=True):
        '''
        The update function
        :param pred: prediction labels
        :param Y:  true ground
        :param epoch: epoch
        :param loss: the training loss
        :param acc: the training accuracy
        '''

        print('|||||||||||| do valid :',do_valid)
        f = f1_score(y_pred=pred, y_true=Y, average=self.f_type)
        #torch.save(self.model, self.filepath +"/last_model.pt")

        F_type = global_param.traning_param['F_type']
        exp_name = global_param.traning_param['exp_tag']
        machine_name = os.uname()[1]
        file = open(exp_name + "result_" + machine_name + "_" + F_type + ".loss_acc", "a+")
        print(str(epoch) + " " + str(loss) + " " + str(acc), file=file)
        file.close()


        printf("---------- epoch : {} ".format(epoch),self.filepath+"/recovery.rec")
        printf(" ep : {} Training Loss : {:.4f}, Acc : {:.4f}".format(epoch,loss, acc),self.filepath+"/log")
        printf(" ep : {} Validation f_mesur : {:.4f}".format(epoch,f), self.filepath +"/log")
        printf(str(epoch)+" " +str(self.best_f)+"\n", self.filepath+"/recovery.rec", display=False)

        if (self.best_f is None) or (f > self.best_f) or (not do_valid):

            if (not self.save):
                self.best_model = copy.deepcopy(self.model)
            else:
                torch.save(self.model,self.filepath+"/best_model.pt")
            report = classification_report(y_pred=pred, y_true=Y)

            self.best_f=f

            printf((epoch,"***********************\n************************\n"),self.filepath+"/log")
            printf((epoch,"**************** Best f-mesure *************** ",f),self.filepath+"/log")
            printf(report,self.filepath+"/log")

    def recovery(self,id):
        '''
        The model recovery function
        :param id: the id of model
        :return: valid index,last epoch
        '''
        self.filepath="logs/Exp_"+str(id)
        #self.model=torch.load(self.filepath+"/last_model.pt")
        lines = open(self.filepath+"/recovery.rec","r").read().split('\n')
        index_str=lines[0].split(' ')
        index_str.pop(-1)
        index=[int(i) for i in index_str]

        for i in range(len(lines)):
            if(lines[len(lines)-1-i].startswith('------')):
                m=lines[len(lines)-1-i+1]
                break

        m=m.split()
        self.best_f=float(m[1])
        epoch=int(m[0])

        return index,epoch-1

    def get_best(self):
        '''
        Getter provide the performances manures
        :return: min loss and best f-mesur
        '''
        return self.min_loss,self.best_f

    def get_best_model(self):
        '''
        Getter provide the best model
        :return: best model
        '''
        if (not self.save):
            return  self.best_model
        else:
            return torch.load(self.filepath+"/best_model.pt")

def save_path():
    '''
    Getter provide the saving path
    :return: saving path
    '''
    logdir = "./logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logdir = generate_unique_logpath(logdir,"Exp")

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    return logdir
