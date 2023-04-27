import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_mAP
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_precision_recall/'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def simulate_predictions(file_markup_gt, file_markup_out, N_FP=10,N_FN=10):
    df = pd.read_csv(file_markup_gt, sep=' ')
    w = int(numpy.nanmean(numpy.abs((df.iloc[:,1]-df.iloc[:,3]).values)))
    h = int(numpy.nanmean(numpy.abs((df.iloc[:,2]-df.iloc[:,4]).values)))
    df_temp = pd.DataFrame(df.iloc[0, :].copy()).T
    W,H = 800,600

    N = df.shape[0]
    df_res = df.iloc[numpy.random.choice(N,N-N_FN,replace=False)].copy()
    for i in range(N_FP):
        cx, cy = int(W * numpy.random.rand()), int(H * numpy.random.rand())
        r = int(numpy.random.rand() * N)
        df_temp.iloc[0,0] = df.iloc[r,0]
        df_temp.iloc[0,5] = df.iloc[r,5]
        df_temp.iloc[0,1:5] = numpy.array((cx,cy,cx+w,cy+h))
        df_res = pd.concat([df_res,df_temp],axis=0)

    df_res['conf'] = numpy.random.rand(df_res.shape[0])
    df_res = df_res.sort_values(by=df_res.columns[0])
    df_res.to_csv(file_markup_out, index=False, sep=' ')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    simulate_predictions(folder_in+'boxes.txt', folder_out+'pred.txt', N_FP=100,N_FN=30)

    precisions,recalls,confidences,class_IDs = tools_mAP.get_precsion_recall_data_from_markups(folder_in+'boxes.txt', folder_out+'pred.txt',iuo_th=0.5,ovp_th=None,ovd_th=None,delim=' ')
    for i,pr in enumerate(zip(precisions,recalls)):
        tools_mAP.plot_precision_recall(pr[0],pr[1], filename_out=folder_out+'PR_%d.png'%i)

    tools_mAP.draw_boxes(class_ID=0, folder_annotation=folder_in, file_markup_true=folder_in+'boxes.txt', file_markup_pred=folder_out+'pred.txt', path_out=folder_out, delim=' ', confidence=0.01,metric='pr')
