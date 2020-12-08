import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

colors = ['salmon', 'yellowgreen', 'coral', 'lightskyblue', 'gold', 'lightcoral', 'bisque', 'tan', 'lavender']

def labels_split(label):
    #new_labels=[ l if len(l)<13 else l[:int(len(l)/2)+1]+"-\n-"+l[int(len(l)/2)+1:] for l in label ]
    #return new_labels
    return label

def save(path,title,title_update=True,w=12,h=6):
    if not os.path.exists(path):
        os.makedirs(path)
    figure = plt.gcf()
    figure.set_size_inches(w,h)
    if title_update :
        plt.title(title)
    plt.savefig(path + title + '.png')
    plt.show()

def circle_plot(dict,path,title=""):
    labels=[i+"  ("+str(dict[i])+")" for i in dict.keys()]
    sizes = [dict[i] for i in dict.keys()]

    plt.pie(sizes, labels=labels_split(labels), colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    save(path, title)

def histo(hist,path,title=""):

    sns.set()
    M=[[key,hist[key]] for key in hist.keys()]
    M=sorted(M,key=lambda x: x[1])
    M.reverse()
    X,Y=[i[0] for i in M],[i[1] for i in M]
    plt.xticks(range(len(X)),X, rotation=70)
    plt.xlabel('Word', fontsize=13,color='r')
    plt.ylabel('Frequency', fontsize=13,color='r')
    plt.bar(labels_split(X),Y,1, ec='w')
    plt.subplots_adjust(bottom=0.33)
    save(path, title)


def box(dict,path,title=""):
    labels=dict.keys()
    data=[dict[i]for i in labels]
    plt.boxplot(data,
            notch=True,
            vert=False,
            patch_artist=True,
            labels=labels_split(labels))
    save(path,title)


def plot_heatmap(corpora_list, bert, mat):
    print(corpora_list, bert, mat)
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    title="Similarity " + bert

    ax.set_xticks(np.arange(len(corpora_list)))
    ax.set_yticks(np.arange(len(corpora_list)))

    ax.set_xticklabels(corpora_list)
    ax.set_yticklabels(corpora_list)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(corpora_list)):
        for j in range(len(corpora_list)):
            text = ax.text(j, i, mat[i, j],
                           ha="center", va="center", color="w")

    fig.tight_layout()
    save('plot/Similarity/',title)


def Data_visual(df,corpora,bert,label):

    plt=sns.relplot(x="x1", y="x2", hue="y",
                    col="Methode",data=df,alpha=0.3,
                    palette = sns.color_palette("hls", len(label)),
                   )
    save('plot/'+corpora+'__'+bert+'_Visualisation/',corpora+'__'+bert+' Visualisation',title_update=False,
                    w=20,h=6)
