from matplotlib import pyplot as plt
from matplotlib import text
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

#labels7 = ['neutral', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad']
#labels6 = ['angry', 'surprise', 'disgust', 'fear', 'happy', 'sad']
#labels7v2 = ['angry', 'surprise', 'disgust', 'fear', 'happy', 'sad', 'contempt']

zhCN=True


labels7 = ['平静', '愤怒', '惊讶', '厌恶', '害怕', '快乐', '悲伤']
labels6 = ['愤怒', '惊讶', '厌恶', '害怕', '快乐', '悲伤']
labels7v2 = ['愤怒', '惊讶', '厌恶', '害怕', '快乐', '悲伤', '轻视']

def checkDit(value, df):
    temp=str(value)
    l=len(temp)
    if l<(df+3):
        return temp
    else:
        format='%0.'+str(df)+'f'
        nt=format%value
        if len(temp.split('.')[0])>1:
            return nt[0:(3+df)]
        else:
            return nt[0:(2+df)]

##change the cmap for Gray or Color display.
def plot_confusion_matrix(cm, tag, labels, title=None, cmap = plt.cm.binary, details=False, df=2, colorbar=True):
    fsw=len(labels)*1.7
    fsh=len(labels)*1.45
    fsize=int(len(labels)/3+20)
    if zhCN:
        #font={'family':'Simhei','weight':'bold','size':str(fsize)}
        #plt.rc(['font',font])
        #font=FontProperties(family='Simhei',size=fsize)
        mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号
    print('Font size: %d'%fsize)
    plt.figure(figsize=(fsw, fsh))
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    format="%0."+str(df)+"f"
    thresshold=1.0/10**(df)
    print(thresshold)
    #format="%0.1f"
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if x_val==y_val:
            plt.text(x_val, y_val, checkDit(c,df), color='white', fontsize=fsize, va='center', ha='center')
            #if c <thresshold or c==100:
            #    plt.text(x_val, y_val, '%d'%(int(c)), color='white', fontsize=fsize, va='center', ha='center')
            ##elif c==100:
            ##    plt.text(x_val, y_val, format %(c,), color='red', fontsize=fsize, va='center', ha='center')
            #else:
            #    plt.text(x_val, y_val, format %(c), color='white', fontsize=fsize, va='center', ha='center')
        else:
            plt.text(x_val, y_val, checkDit(c,df), color='black', fontsize=fsize, va='center', ha='center')
            #if c < thresshold or c==100:
            #    plt.text(x_val, y_val, '%d'%(int(c)), color='black', fontsize=fsize, va='center', ha='center')
            ##elif c>0:
            ##    plt.text(x_val, y_val, format %(c), color='blue', fontsize=fsize, va='center', ha='center')
            #else:
            #    plt.text(x_val, y_val, format %(c), color='black', fontsize=fsize, va='center', ha='center')

    tick_marks = np.array(range(len(labels)))+1.0
    plt.gca().set_xticks(tick_marks, minor = True)
    plt.gca().set_yticks(tick_marks, minor = True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    #plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title)
    if colorbar:
        cbar=plt.colorbar()
        if zhCN:
            cbar.set_label('准确率 (%)', size=fsize)
        else:
            cbar.set_label('Accuracy (%)', size=fsize)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(fsize)
    xlocations = np.array(range(len(labels)))
    if zhCN:
        plt.xticks(xlocations, labels, size=fsize)
    else:
        plt.xticks(xlocations, labels, size=fsize, rotation=60)
    #plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels, size=fsize)
    if details:
        plt.ylabel('GroundTruth')
        plt.xlabel('Predict')
    plt.savefig(tag+'.jpg')
    plt.close()


if __name__=='__main__':
    #cm=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3384615384615385, 0.05333333333333334, 0.0, 0.011904761904761904, 0.0, 0.013333333333333334, 0.02702702702702703], [0.19487179487179487, 0.0, 0.6912751677852349, 0.0, 0.0, 0.05333333333333334, 0.0], [0.3487179487179487, 0.013333333333333334, 0.0, 0.07142857142857142, 0.0, 0.02666666666666667, 0.06756756756756757], [0.29743589743589743, 0.0, 0.22818791946308725, 0.011904761904761904, 0.0, 0.013333333333333334, 0.0], [0.046153846153846156, 0.0, 0.006711409395973154, 0.0, 0.0, 0.9333333333333333, 0.0], [0.3435897435897436, 0.013333333333333334, 0.0, 0.011904761904761904, 0.0, 0.0, 0.06756756756756757]]
    #plot_confusion_matrix(cm, 'MStest', labels7)
    #cm=[[0.8071065989847716, 0.02666666666666667, 0.16, 0.047619047619047616, 0.03125, 0.0196078431372549, 0.02702702702702703], [0.27411167512690354, 0.08, 0.02666666666666667, 0.047619047619047616, 0.0, 0.013071895424836602, 0.06756756756756757], [0.07614213197969544, 0.0, 0.8133333333333334, 0.05952380952380952, 0.0, 0.0392156862745098, 0.02702702702702703], [0.27411167512690354, 0.02666666666666667, 0.02, 0.08333333333333333, 0.0, 0.032679738562091505, 0.17567567567567569], [0.116751269035533, 0.02666666666666667, 0.4066666666666667, 0.05952380952380952, 0.03125, 0.013071895424836602, 0.0], [0.06091370558375635, 0.013333333333333334, 0.03333333333333333, 0.047619047619047616, 0.020833333333333332, 0.8366013071895425, 0.013513513513513514], [0.233502538071066, 0.09333333333333334, 0.013333333333333334, 0.03571428571428571, 0.041666666666666664, 0.0196078431372549, 0.12162162162162163]]
    #plot_confusion_matrix(cm, 'FACE++')
    
    Test1=[[88.5714285714286,0,4.28571428571429,5,0.714285714285714,1.42857142857143],
                                [0,95.7142857142857,0,2.85714285714286,0,1.42857142857143],
                                [2.14285714285714,0,92.1428571428572,1.42857142857143,0,4.28571428571429],
                                [1.42857142857143,5.71428571428571,2.14285714285714,85.0000000000000,2.14285714285714,3.57142857142857],
                                [0.714285714285714,0,0,0.714285714285714,98.5714285714286,0],
                                [2.14285714285714,0,5,4.28571428571429,0,88.5714285714286]]
    plot_confusion_matrix(Test1, 'Test1', labels6)
    


    Test2=[[97.7777777777778,0,0.740740740740741,0,0,0,1.48148148148148],
                                [0,98.7755102040816,0,0,0,0,1.22448979591837],
                                [0.568181818181818,0,99.4318181818182,0,0,0,0],
                                [0,0,0,100,0,0,0],
                                [0,0,0,0,100,0,0],
                                [1.19047619047619,0,0,0,0,98.8095238095238,0],
                                [0,0,0,0,0,0,100]]
    plot_confusion_matrix(Test2, 'Test2', labels7v2)
