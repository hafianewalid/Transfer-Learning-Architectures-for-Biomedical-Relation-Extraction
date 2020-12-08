import torch.nn as nn
from opt_einsum.backends import torch
import torch.nn.functional as F
import torch
import Bert

Switch = False


def get_layer(type_l='linear', inputs=28, outputs=10, drop=0.5, in_c=1, out_c=10, ker_size=[5, 2], stride=[1, 2]):
    '''
    this function get layer or block of network
    :param type_l: string indicate the type of layer
    :param inputs: the input size
    :param outputs: the output size
    :param drop: the drop out proba
    :param in_c: number of input channels
    :param out_c: number of output channels
    :param ker_size: size of conv kernel and polling kernel
    :param stride: the strides of convolution and polling
    :return: the network block
    '''

    if (type_l == 'linear'):
        return [nn.Linear(inputs, outputs)]

    if (type_l == 'linear_relu'):
        return [nn.Linear(inputs, outputs),
                nn.ReLU(inplace=True)]

    if (type_l == 'linear_dropout'):
        return [nn.Dropout(drop),
                nn.Linear(inputs, outputs)]

    if (type_l == 'linear_dropout_relu'):
        return [nn.Dropout(drop),
                nn.Linear(inputs, outputs),
                nn.ReLU(inplace=True)
                ]

    if (type_l == 'conv'):
        return [
            nn.Conv1d(in_c, out_c,
                      kernel_size=ker_size[0],
                      stride=stride[0],
                      padding=int((ker_size[0] - 1) / 2), bias=True
                      ),
            nn.ReLU(inplace=True)
        ]
    if (type_l == 'conv1D'):
        return [
            nn.Conv1d(in_c, out_c,
                      kernel_size=ker_size[0],
                      stride=stride[0],
                      padding=int((ker_size[0] - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=ker_size[1], stride=stride[1])
        ]
    if (type_l == 'conv2D'):
        return [
            nn.Conv2d(in_c, out_c,
                      kernel_size=ker_size[0],
                      stride=stride[0],
                      padding=int((ker_size[0] - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=ker_size[1], stride=stride[1])
        ]
    if (type_l == 'conv1D_bn_relu'):
        return [nn.Conv1d(in_c, out_c,
                          kernel_size=ker_size[0],
                          stride=1,
                          padding=int((ker_size[0] - 1) / 2), bias=True),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True)]

    if (type_l == 'conv1D_bn_bloc'):
        bloc = []
        bloc.extend(get_layer(type_l='conv_bn_relu', inputs=inputs, outputs=outputs, drop=drop,
                              in_c=in_c, out_c=out_c, ker_size=ker_size, stride=stride))
        bloc.extend(get_layer(type_l='conv_bn_relu', inputs=inputs, outputs=outputs, drop=drop,
                              in_c=out_c, out_c=out_c, ker_size=ker_size, stride=stride))
        bloc.append(nn.MaxPool1d(kernel_size=ker_size[1], stride=stride[1]))

        return bloc


class Fully_Connected(nn.Module):
    def __init__(self, FC_L_size=[200], input=768, l2reg=0.05, out=3, out_src=0, Dr=[0.5, 0.5, 0.5, 0.5]):

        super(Fully_Connected, self).__init__()

        self.l2_reg = l2reg
        self.Weight = []
        self.trensfer = out_src != 0

        self.FC_layers = []
        self.FC = get_layer(type_l='linear_dropout_relu', inputs=input, outputs=FC_L_size[0], drop=Dr[0])
        self.FC_layers.extend(self.FC)
        self.Weight.append(self.FC[1])
        for i in range(1, len(FC_L_size)):
            self.FC = get_layer(type_l='linear_dropout_relu', inputs=FC_L_size[i - 1], outputs=FC_L_size[i], drop=Dr[i])
            self.FC_layers.extend(self.FC)
            self.Weight.append(self.FC[1])
        self.FC_layers.append(self.FC)
        self.FC_classifier = nn.Sequential(*self.FC_layers)

        self.out_layers = nn.Linear(FC_L_size[-1], out)
        self.out_src_layers = nn.Linear(FC_L_size[-1], out_src) if self.trensfer else None

    def forward(self, x, xr):
        x1 = self.FC_classifier(x)
        if self.trensfer:
            x1 = self.out_layers(x1) if xr[0, 0] == 0 else self.out_src_layers(x1)
        else:
            x1 = self.out_layers(x1)

        return x1

    def penalty(self):
        l2_w = 0
        for w in self.Weight:
            l2_w += w.weight.norm(2)
        return self.l2_reg * l2_w


class CNN(nn.Module):

    def __init__(self, inc=1, xr_size=200, xr=False, out=3, out_src=0, ker_size=[3, 2],
                 strides=[1, 2], nb_filtres=[3]
                 , Dr=[0.5, 0.5], FC_L_size=[64], l2reg=0.05
                 , padding=True, maxpooling_dir=False, w_size=500):


        super(CNN, self).__init__()

        self.l2_reg = l2reg
        self.Weight = []
        self.xr, self.padding, self.maxpooling_dir, self.w_size = xr, padding, maxpooling_dir, w_size

        nb_input = inc

        self.Conv_layers = []
        self.conv_classifier = get_layer(type_l='conv', in_c=nb_input, out_c=nb_input * nb_filtres[0],
                                         ker_size=ker_size, stride=strides)
        self.Conv_layers.extend(self.conv_classifier)
        self.Weight.append(self.conv_classifier[0])
        for i in range(1, len(nb_filtres)):
            nb_input *= nb_filtres[i - 1]
            self.conv_classifier = get_layer(type_l='conv', in_c=nb_input, out_c=nb_input * nb_filtres[i],
                                             ker_size=ker_size, stride=strides)
            self.Conv_layers.extend(self.conv_classifier)
            self.Weight.append(self.conv_classifier[0])
        self.Conv_classifer = nn.Sequential(*self.Conv_layers)

        pooling_size = inc * nb_filtres[-1] if maxpooling_dir else w_size
        fc_input = pooling_size + xr_size if xr else pooling_size


        self.FC_layers = []
        self.FC = get_layer(type_l='linear_dropout_relu', inputs=fc_input, outputs=FC_L_size[0], drop=Dr[0])
        self.FC_layers.extend(self.FC)
        self.Weight.append(self.FC[1])
        for i in range(1, len(FC_L_size)):
            self.FC = get_layer(type_l='linear_dropout_relu', inputs=FC_L_size[i - 1], outputs=FC_L_size[i], drop=Dr[i])
            self.FC_layers.extend(self.FC)
            self.Weight.append(self.FC[1])
        self.FC = nn.Linear(FC_L_size[-1], out)
        self.FC_layers.append(self.FC)
        self.FC_classifier = nn.Sequential(*self.FC_layers)

    def forward(self, x, xr):
        '''
        the forward function
        :param x: the first input
        :param xr: the second input
        :return: the output
        '''

        # convolution
        x = self.Conv_classifer(x)
        if self.maxpooling_dir:
            # global average polling of cnn features map in direction of channels
            x = F.adaptive_avg_pool2d(x, (x.size()[1], 1))
        else:
            # global max polling of cnn features map in direction of words
            x = F.adaptive_max_pool2d(x, (x.size()[2], 1))

        x = x.view(x.size()[0], -1)

        if self.padding:
            x = nn.ConstantPad1d((0, self.w_size - x.size()[1]), 0)(x)

        if self.xr:
            x = torch.cat((x, xr), 1)

        x = self.FC_classifier(x)

        return x

    def penalty(self):
        l2_w = 0
        for w in self.Weight:
            l2_w += w.weight.norm(2)
        return self.l2_reg * l2_w


class CNN_residual(nn.Module):

    def __init__(self, inc=1, xr_size=200, out=3, out_src=0, ker_size=[5, 2],
                 srides=[1, 2], nb_filtres=[16]
                 , Dr=[0.25, 0.25, 0.0, 0.0], FC_L_size=[250], l2reg=0.05):

        super(CNN_residual, self).__init__()

        self.l2_reg = l2reg
        self.Weight = []

        nb_input = inc

        self.Conv_layers = []
        self.conv_classifier = get_layer(type_l='conv2', in_c=nb_input, out_c=nb_input * nb_filtres[0],
                                         ker_size=ker_size, stride=srides)
        self.Conv_layers.extend(self.conv_classifier)
        self.Weight.append(self.conv_classifier[0])
        for i in range(1, len(nb_filtres)):
            nb_input *= nb_filtres[i - 1]
            self.conv_classifier = get_layer(type_l='conv2', in_c=nb_input, out_c=nb_input * nb_filtres[i],
                                             ker_size=ker_size, stride=srides)
            self.Conv_layers.extend(self.conv_classifier)
            self.Weight.append(self.conv_classifier[0])

        self.Conv_classifer = nn.Sequential(*self.Conv_layers)

        ############### CNN 2 ////////////////
        nb_input2 = nb_input * nb_filtres[-1]

        self.Conv_layers2 = []
        self.conv_classifier2 = get_layer(type_l='conv2', in_c=nb_input2, out_c=nb_input2 * nb_filtres[0],
                                          ker_size=ker_size, stride=srides)
        self.Conv_layers2.extend(self.conv_classifier2)
        self.Weight.append(self.conv_classifier2[0])
        for i in range(1, len(nb_filtres)):
            nb_input2 *= nb_filtres[i - 1]
            self.conv_classifier2 = get_layer(type_l='conv2', in_c=nb_input2, out_c=nb_input2 * nb_filtres[i],
                                              ker_size=ker_size, stride=srides)
            self.Conv_layers.extend(self.conv_classifier2)
            self.Weight.append(self.conv_classifier2[0])

        self.Conv_classifer2 = nn.Sequential(*self.Conv_layers2)

        ##########################################


        fc_input = nb_input * nb_filtres[-1] + nb_input2 * nb_filtres[-1] + xr_size

        self.FC_layers = []
        self.FC = get_layer(type_l='linear_dropout_relu', inputs=fc_input, outputs=FC_L_size[0], drop=Dr[0])
        self.FC_layers.extend(self.FC)
        self.Weight.append(self.FC[1])
        for i in range(1, len(FC_L_size)):
            self.FC = get_layer(type_l='linear_dropout_relu', inputs=FC_L_size[i - 1], outputs=FC_L_size[i], drop=Dr[i])
            self.FC_layers.extend(self.FC)
            self.Weight.append(self.FC[1])
        self.FC = nn.Linear(FC_L_size[-1], out)
        self.FC_layers.append(self.FC)
        self.FC_classifier = nn.Sequential(*self.FC_layers)

    def forward(self, x, xr):

        conv1 = self.Conv_classifer(x)
        # x = F.adaptive_avg_pool2d(x, (x.size()[1], 1))
        x1 = F.adaptive_max_pool2d(conv1, (conv1.size()[1], 1))
        x1 = x1.view(x1.size()[0], -1)

        conv2 = self.Conv_classifer2(conv1)
        x2 = F.adaptive_max_pool2d(conv2, (conv2.size()[1], 1))
        x2 = x2.view(x2.size()[0], -1)

        x = torch.cat((x1, x2, xr), 1)

        x = self.FC_classifier(x)

        return x

    def penalty(self):
        l2_w = 0
        for w in self.Weight:
            l2_w += w.weight.norm(2)
        return self.l2_reg * l2_w


class RNN(nn.Module):
    def __init__(self, out=3, out_src=0):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, dropout=0.5, input_size=768, hidden_size=200,
                           batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = nn.Linear(400, out)

    def forward(self, x, xr):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def penalty(self):
        return self.fc.penalty()


class RNN_CNN(nn.Module):
    def __init__(self, out=3, out_src=0, seq=True):

        super(RNN_CNN, self).__init__()

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, dropout=0.5, input_size=768, hidden_size=200,
                           batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.seq = seq
        inc = 400 if self.seq else 768
        self.cnn = CNN(inc=inc, FC_L_size=[500], xr_size=400, xr=True, padding=False, maxpooling_dir=True, out=out)

    def forward(self, x, xr):
        x = x.permute(0, 2, 1)
        if self.seq:
            x, _ = self.rnn(x)
            x1 = x[:, 0, :]
            x1 = self.dropout(x1)
            x = x.permute(0, 2, 1)
            x = self.cnn(x, x1)
        else:
            x1, _ = self.rnn(x)
            x1 = x1[:, 0, :]
            x1 = self.dropout(x1)
            x = x.permute(0, 2, 1)
            x = self.cnn(x, x1)

        return x

    def penalty(self):
        return self.fc.penalty()


class Bert_finetuning(nn.Module):
    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)

        self.trensfer = out_src != 0
        self.out_layers = nn.Linear(768, out)
        self.out_src_layers = nn.Linear(768, out_src) if self.trensfer else None

    def penalty(self):
        l2 = self.fc[2].weight.norm(2)
        return l2 * 0.1

    def forward(self, x, xr):
        activity_layers, _ = self.bert_model(x)
        if not isinstance(activity_layers, torch.Tensor):
            activity_layers = activity_layers[-1]
        out = activity_layers[:, 0, :]

        if self.trensfer:
            y = self.out_layers(out) if xr[0, 0] == 0 else self.out_src_layers(out)
        else:
            y = self.out_layers(out)

        return y


class Bert_finetuning_CNN(nn.Module):
    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning_CNN, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)
        self.cnn = CNN(inc=768, Dr=[0.5, 0.5], nb_filtres=[8], FC_L_size=[768], ker_size=[3, 2], padding=False,
                       maxpooling_dir=True, out=out, out_src=out_src)

    def forward(self, x, xr):
        activity_layers, _ = self.bert_model(x)
        if not isinstance(activity_layers, torch.Tensor):
            activity_layers = activity_layers[-1]
        activity_layers = activity_layers.permute(0, 2, 1)
        y = self.cnn(activity_layers[:, :, :], xr)
        return y


class Bert_finetuning_CNN_residual(nn.Module):
    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning_CNN_residual, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)
        self.cnn = CNN(inc=768, nb_filtres=[3], FC_L_size=[768], ker_size=[3, 2], padding=False, maxpooling_dir=True,
                       out=out, out_src=0, w_size=768)

    def forward(self, x, xr):
        activity_layers, _ = self.bert_model(x)
        activity_layers = torch.stack(activity_layers)
        activity_layers = activity_layers.permute(1, 3, 2, 0)
        activity_layers = activity_layers[:, :, 0, :]
        y = self.cnn(activity_layers, xr)
        return y


class Bert_finetuning_CNN2(nn.Module):
    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning_CNN, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)
        self.fc = nn.Linear(200, out)
        self.cnn1 = CNN(inc=768, Dr=[0.5, 0.5], nb_filtres=[8], FC_L_size=[768], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)
        self.cnn2 = CNN(inc=768, Dr=[0.5, 0.5], nb_filtres=[8], FC_L_size=[768], ker_size=[5, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

    def forward(self, x, xr):
        activity_layers, _ = self.bert_model(x)
        if not isinstance(activity_layers, torch.Tensor):
            activity_layers = activity_layers[-1]

        activity_layers = activity_layers.permute(0, 2, 1)
        y1 = self.cnn1(activity_layers[:, :, :], xr)

        activity_layers = torch.stack(activity_layers)
        activity_layers = activity_layers.permute(1, 3, 2, 0)
        activity_layers = activity_layers[:, :, 0, :]
        activity_layers = activity_layers.new_tensor()
        y2 = self.cnn2(activity_layers[:, :, :], xr)

        y = self.fc(torch.cat([y1, y2], dim=0))
        return y


class Bert_finetuning_Mixed(nn.Module):
    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning_Mixed, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)

        self.cnn = CNN(inc=768, Dr=[0.5, 0.5], nb_filtres=[8], FC_L_size=[768], ker_size=[3, 2], padding=False,
                       maxpooling_dir=True, out=out, out_src=out_src)

        self.out_layers = nn.Linear(768, out)

    def forward(self, x, xr):

        if (Switch):
            self.bert_model.eval()
            # self.Features, _ = self.bert_model(x)
            activity_layers = x.permute(0, 2, 1)

        else:
            activity_layers, _ = self.bert_model(x)
            if not isinstance(activity_layers, torch.Tensor):
                activity_layers = activity_layers[-1]

        if (Switch):
            y = self.cnn(activity_layers[:, :, :], xr)
        else:
            out = activity_layers[:, 0, :]
            y = self.out_layers(out)

        return y


class Bert_finetuning_CNN_Entity(nn.Module):

    def __init__(self, out=3, out_src=0, bert_type='bert'):
        super(Bert_finetuning_CNN_Entity, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)

        self.cnn1 = CNN(inc=768, Dr=[0.25, 0.25], nb_filtres=[3], FC_L_size=[100], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

        self.cnn2 = CNN(inc=768, Dr=[0.25, 0.25], nb_filtres=[3], FC_L_size=[100], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

        self.cnn3 = CNN(inc=768, Dr=[0.25, 0.25], nb_filtres=[3], FC_L_size=[100], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

        self.cnn4 = CNN(inc=768, Dr=[0.25, 0.25], nb_filtres=[3], FC_L_size=[100], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

        self.cnn5 = CNN(inc=768, Dr=[0.25, 0.25], nb_filtres=[3], FC_L_size=[100], ker_size=[3, 2], padding=False,
                        maxpooling_dir=True, out=100, out_src=out_src)

        self.fc = nn.Sequential(
            nn.LayerNorm(500),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(500, 100),
            nn.Hardtanh(),
            nn.LayerNorm(100),
            nn.Linear(100, out),
            nn.Hardtanh()
        )

    def forward(self, x, xr):
        activity_layers, _ = self.bert_model(x)

        a1, e1, a2, e2, a3 = [], [], [], [], []

        for i in range(xr.size()[0]):
            e1start, e1end, e2start, e2end = xr[i, 0], xr[i, 1], xr[i, 2], xr[i, 3]

            a1.append(activity_layers[i, :e1start + 1, :])

            e1.append(activity_layers[i, e1start:e1end + 1, :])

            a2.append(activity_layers[i, e1end - 1:e2start + 1, :])

            e2.append(activity_layers[i, e2start:e2end + 1, :])

            a3.append(activity_layers[i, e2end:, :])

            if a2[-1].size()[0] == 0:
                a2[-1] = torch.FloatTensor([[0.] * (768 * 1)] * 3).cuda()

        a1 = torch.nn.utils.rnn.pad_sequence(a1)
        e1 = torch.nn.utils.rnn.pad_sequence(e1)
        a2 = torch.nn.utils.rnn.pad_sequence(a2)
        e2 = torch.nn.utils.rnn.pad_sequence(e2)
        a3 = torch.nn.utils.rnn.pad_sequence(a3)

        a1 = a1.permute(1, 2, 0)
        e1 = e1.permute(1, 2, 0)
        a2 = a2.permute(1, 2, 0)
        e2 = e2.permute(1, 2, 0)
        a3 = a3.permute(1, 2, 0)

        y1 = self.cnn1(a1, xr)
        y2 = self.cnn2(e1, xr)
        y3 = self.cnn3(a2, xr)
        y4 = self.cnn4(e2, xr)
        y5 = self.cnn5(a3, xr)

        y = self.fc(torch.cat([y1, y2, y3, y4, y5], dim=1))

        return y
