import torch
from tqdm import tqdm
import Models
from Load_data import torch_loader
from Checkpoint import ModelCheckpoint, save_path
from Parameters import global_param
import random


lr_init,num_train_steps,warmup_proportion,global_step=0,0,0,0
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def train(model, loader,f_loss, optimizer,loader_src=None, scheduler=None,grad_norm=False,penalty=False,steps_lr_update=False ):
    '''
    This function train the network model
    :param model:the network model
    :param loader: the data loader
    :param f_loss: the loss function
    :param optimizer: the optimizer
    :param penalty: L2 regularisation
    :return: tuple of training loss and training accuracy
    '''

    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0

    global global_step

    seq=global_param.traning_param['trensfer_type']=='seq'
    src_ep=global_param.traning_param['trensfer_switch']


    nb_batch=len(loader) if loader_src==None else len(loader)+len(loader_src)

    if loader_src!=None and seq and src_ep>0:
     corpora =[i for i in enumerate(loader_src)]
     nb_batch=len(loader_src)
    else:
     corpora=[i for i in enumerate(loader)]

    if loader_src!=None and not seq:
        corpora.extend([i for i in enumerate(loader_src)])

    random.shuffle(corpora)

    pbar = tqdm(total=nb_batch, desc="Training batch : ")

    for i, (inputs1, inputs2, targets) in corpora:

        inputs1,inputs2,targets = inputs1.to(global_param.device),inputs2.to(global_param.device),targets.to(global_param.device)


        if not global_param.model_param['fine_tuning']:
            inputs1=(inputs1.data).permute(0,2,1)


        outputs = model(inputs1,inputs2)
        loss = f_loss(outputs, targets)

        if steps_lr_update:
            lr_this_step = lr_init * warmup_linear(global_step / num_train_steps,warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            global_step += 1

        optimizer.zero_grad()

        loss.backward()

        if penalty :
            model.penalty().backward()
        
        if grad_norm :
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
 
        optimizer.step()

        if scheduler!=None :
            scheduler.step()

        N += targets.shape[0]
        tot_loss += targets.shape[0] * loss.item()
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        pbar.update(1)

    pbar.close()

    return tot_loss / N , correct / N

X_v=None
def prediction(model,X,valid=False):
    '''
    This function compute the predictions associated to the inputs using the network model
    :param model: the model used for prediction
    :param X: the inputs
    :return: the outputs corresponding to the inputs
    '''

    global X_v
    if(Models.Switch):
        if (valid):
          if(X_v==None):
           X_v = [model.bert_model(x) for x in X]
          X=X_v
        else:
          X=[model.bert_model(x) for x in X]


    pbar = tqdm(total=len(X), desc=" Prediction : ")
    Y=[]
    for x in X:
        if not global_param.model_param['fine_tuning']:
            input1 = torch.stack([x[0].data]).permute(0, 2, 1)
        else:
            input1 = torch.stack([x[0]])
        input2 = torch.stack([x[1]])
        input1,input2=input1.to(global_param.device),input2.to(global_param.device)
        with torch.no_grad():
            model.eval()
            output = model(input1,input2)
            predicted_targets = output.argmax(dim=1)
            Y.append(predicted_targets.tolist()[0])
        pbar.update(1)
        continue
    pbar.close()
    return Y



def train_save(model, X_train, Y_train, X_train_src, Y_train_src, nb_epoch=30, recovery=False, recovery_id=None, batch_size=32, percentage=10, X_valid=[], Y_valid=[], F_type='macro', lr= 0.001, do_valid=True):
    '''
    This function train the network model for n epoch with saving and recovery
    :param model: the model to train it
    :param X_train: the inputs of training data
    :param Y_train: the labels of training data
    :param nb_epoch: the number of training epochs
    :param recovery: the recovery mode
    :param recovery_id: model id
    :param batch_sise: the size of batch
    :param percentage: the slicing percentage (this method use sub part of training data as validation data)
    :return: the best model through training epochs
    '''
    start_epoch=0
    indx_valid =[]
    v = 0 if percentage==0 else int(len(Y_train) / (100 / percentage))

    if(recovery):
        checkpoint = ModelCheckpoint(None, model,None)
        indx_valid,start_epoch=checkpoint.recovery(recovery_id)
        model=checkpoint.model
        if (X_valid==[]):
            for ind in indx_valid:
                X, Y = X_train.pop(ind), Y_train.pop(ind)
                X_valid.append(X)
                Y_valid.append(Y)
    else:
        path = save_path()
        if (X_valid==[]):
            for i in range(v):
                ind = 0
                X, Y = X_train.pop(ind), Y_train.pop(ind)
                X_valid.append(X)
                Y_valid.append(Y)
                indx_valid.append(ind)
        checkpoint = ModelCheckpoint(path, model,indx_valid,F_type=F_type)


    loader_app = torch_loader(X_train, Y_train, batch_size=batch_size)
    loader_app_src = torch_loader(X_train_src, Y_train_src, batch_size=batch_size) if X_train_src != [] else None

    f_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.0,amsgrad=False)


    scheduler=None


    for i in range(start_epoch,nb_epoch):
       loss ,acc = train(model,loader_app, f_loss, optimizer,loader_src=loader_app_src,scheduler=scheduler,penalty=False)
       pred=prediction(model,X_valid,valid=True)
       checkpoint.update(pred,Y_valid,i,loss,acc,do_valid=do_valid)

    return checkpoint.get_best_model(),checkpoint.filepath
