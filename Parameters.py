import argparse

import torch

masks_type={
    "simple_bert":["[unused_1]","[unused_1]"],
    "bio_bert":["@GENE$","@DISEASE$"]
}

encapsulate_items={
    "sci_bert":["<<",">>","[[","]]"]
}

class Hyperparameter :
    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('-ft', default=False, type=bool,
                            dest='fine_tuning',
                            help='Transfer learning strategies : True for fine tuning / False for frozen')
        parser.add_argument('-bert', default='bert', choices=['bert', 'biobert', 'scibert'],
                            dest='bert',
                            help='Bert model : bert, biobert, scibert')
        parser.add_argument('-frozen_model', default='RNN', choices=['RNN', 'CNN','CNN_RNN', 'CNN_RNN_parell'],
                            dest='frozen_model',
                            help='The frozen architecture : RNN (LSTM), CNN (MCNN) , CNN_RNN (LSTM_MCNN L.) , CNN_RNN_parell (LSTM_MCNN P.)')
        parser.add_argument('-fine_tuning_model', default='MLP', choices=['MLP','CNN','CNN_seg'],
                            dest='fine_tuning_model',
                            help='The fine tuning architecture : MLP , CNN (MCNN) , CNN_seg')
        parser.add_argument('-F_type', default='macro', choices=['micro','macro'],
                            dest='F_type',
                            help='Type of F-mesure avg (micro,macro)')
        parser.add_argument('-corpus', default='chemprot', choices=['chemprot', 'pgx'],
                            dest='corpus',
                            help='corpus (chemprot, pgx)')
        parser.add_argument('-lr', default=0.001, type=float,
                            dest='lr',
                            help='learning rate ( 0.001 for frozen, 5e-5 / 3e-5  for fine-tune')
        parser.add_argument('-num_ep',default=5, type=int,
                            dest='num_ep',
                            help='number of epochs ( 5/8 : for fine tuning ) / (30/ 60) for frozen ')

        param = parser.parse_args()


        self.model_param={
        'fine_tuning':param.fine_tuning,
        'bert': param.bert
        }

        self.corpus_param={
        'corpus':param.corpus,
        'corpus_src':'',
        'annonimitation':False,
        'encapculate':param.corpus!='chemprot',
        'normalisation':False,
        'entitys_masks':masks_type["bio_bert"],
        'encapsulate_items':encapsulate_items['sci_bert'],
        'padding_size':500,
        'post_embadding':True,
        'post_indication':False,
        'token_mean':False
        }


        self.frozen_model=param.frozen_model


        self.finetuning_model=param.fine_tuning_model

        method= self.finetuning_model if self.model_param['fine_tuning'] else 'frozen'+self.frozen_model
        method+='_'+self.model_param['bert']
        method+='_'+self.corpus_param['corpus']

        if self.corpus_param['corpus_src'] != '':
           method+='_src:'+self.corpus_param['corpus_src']

        self.traning_param={
        'num_ep':param.num_ep,
        'switch_ep':5,
        'trensfer_type':'seq',
        'trensfer_switch':3,
        'token_mean':False,
        'batch_size':32,
        'F_type':param.F_type,
        'exp_tag':method,
        'lr':param.lr
        }
        
        method+='_ep'+str(self.traning_param['num_ep'])
        method+='_lr'+str(self.traning_param['lr'])
        self.traning_param['exp_tag']=method
        print(method)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("°°°°°°°°°°°°°°°°°°°°  GPU  °°°°°°°°°°°°°°°°°°")
            self.device = torch.device('cuda')
        else:
            print("°°°°°°°°°°°°°°°°°°°°  CPU  °°°°°°°°°°°°°°°°°°")
            self.device = torch.device('cpu')




global_param=Hyperparameter()
