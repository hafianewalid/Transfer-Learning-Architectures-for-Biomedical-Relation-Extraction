import glob
import pandas
import matplotlib.pyplot as plt
import pylab
import sys

def stat(corpus):
    
    results=glob.glob("result/"+corpus+"/res/*.res") if corpus!='' else glob.glob("*.res")
     
    for res in results:
        file=open(res,"r")
        head=""
        txt=file.read()
        file.close()
        if not txt.startswith("precision"):
            txt="precision recall fscore .\n"+txt
            file=open(res,"w")
            file.write(txt)
            file.close()

    res_frame=pandas.DataFrame(columns=['precision','recall','fscore'])

    for res in results:
         frame=(pandas.read_csv(res,sep=" ")).drop(['.'],axis=1)
         res_frame=res_frame.append(frame,ignore_index=True)

    print("//////////// all result ///////////")
    print(res_frame)

    print("/////////////////////////// stat //////////////////////////")
    print("///////////////////////////////////////////////////////////")
    min=pandas.DataFrame(res_frame.min(),columns=["min"])
    max=pandas.DataFrame(res_frame.max(),columns=["max"])
    mean=pandas.DataFrame(res_frame.mean(),columns=["mean"])
    std=pandas.DataFrame(res_frame.std(),columns=["std"])

    stat=pandas.concat([min,max,mean,std],axis=1)
    print(stat)

    data = res_frame['precision'],res_frame['recall'],res_frame['fscore']
    BoxName = ['precision','recall','fscore']

    plt.boxplot(data)
    plt.ylim(0,1)
    pylab.xticks([1,2,3], BoxName)

    plt.savefig('performence_boxs.png')
    plt.show()

if len(sys.argv)>1 :
   stat(sys.argv[1]) 
else :
   stat('')

