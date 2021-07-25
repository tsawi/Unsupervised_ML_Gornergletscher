

import h5py
import pandas as pd
import numpy as np
from obspy import read
from scipy.fft import fft, fftfreq

import datetime as dtt

import datetime
from scipy.stats import kurtosis
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from scipy import spatial

from scipy.signal import butter, lfilter
import librosa
# # sys.path.insert(0, '../01_DataPrep')
from scipy.io import loadmat
from sklearn.decomposition import PCA
# sys.path.append('.')
from sklearn.metrics import silhouette_samples
import scipy as sp
import scipy.signal

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report



##########################################################################################



def butter_bandpass(fmin, fmax, fs, order=5):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fmin, fmax, fs, order=5):
    b, a = butter_bandpass(fmin, fmax, fs, order=order)
    y = lfilter(b, a, data)
    return y

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

##################################################################################################
##################################################################################################
#   _               _            _       _                          _                 _   _
#  | |             (_)          | |     | |                        | |               | | (_)
#  | |__   __ _ ___ _  ___    __| | __ _| |_ __ _    _____  ___ __ | | ___  _ __ __ _| |_ _  ___  _ __
#  | '_ \ / _` / __| |/ __|  / _` |/ _` | __/ _` |  / _ \ \/ / '_ \| |/ _ \| '__/ _` | __| |/ _ \| '_ \
#  | |_) | (_| \__ \ | (__  | (_| | (_| | || (_| | |  __/>  <| |_) | | (_) | | | (_| | |_| | (_) | | | |
#  |_.__/ \__,_|___/_|\___|  \__,_|\__,_|\__\__,_|  \___/_/\_\ .__/|_|\___/|_|  \__,_|\__|_|\___/|_| |_|
#                                                            | |
#                                                            |_|
##################################################################################################


def dateToEventID(cat):


    evID = []

    for i, dt in enumerate(cat.datetime):

        a = str(dt)
        b = a.replace('-','').replace(':','').replace(' ','')[3:]


        evID.append(b)

    cat['event_ID'] = evID

    return cat


def getWF(evID,dataH5_path,station,channel,fmin,fmax,fs):

    with h5py.File(dataH5_path,'a') as fileLoad:

        wf_data = fileLoad[f'waveforms/{station}/{channel}'].get(str(evID))[:]

    wf_filter = butter_bandpass_filter(wf_data, fmin,fmax,fs,order=4)
    wf_zeromean = wf_filter - np.mean(wf_filter)

    return wf_zeromean



def getSpectra(evID,station,path_proj,normed=True):

    if normed == False:
        ##right now saving normed
        try:
            mat = loadmat(f'{path_proj}01_input/{station}/specMats_nonNormed/{evID}.mat')
        except:
            mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')

    else:

        mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')


    specMat = mat.get('STFT')

    matSum = specMat.sum(1)

    mat.get('fSTFT')
    return matSum,specMat

def getSpectra_fromWF(evID,dataH5_path,station,channel,normed=True):
## get WF from H5 and calc full sgram for plotting

    with h5py.File(dataH5_path,'r') as dataFile:

        wf_data = dataFile[f'waveforms/{station}/{channel}'].get(str(evID))[:]


        fs = dataFile['spec_parameters/'].get('fs')[()]

        # fmin =
        nperseg = dataFile['spec_parameters/'].get('nperseg')[()]
        noverlap = dataFile['spec_parameters/'].get('noverlap')[()]
        nfft = dataFile['spec_parameters/'].get('nfft')[()]


        fmax = dataFile['spec_parameters/'].get('fmax')[()]
        fmax = np.ceil(fmax)
        fmin = dataFile['spec_parameters/'].get('fmin')[()]
        fmin = np.floor(fmin)
        fSTFT = dataFile['spec_parameters/'].get('fSTFT')[()]
        tSTFT = dataFile['spec_parameters/'].get('tSTFT')[()]

        sgram_mode = dataFile['spec_parameters/'].get('mode')[()].decode('utf-8')
        scaling = dataFile['spec_parameters/'].get('scaling')[()].decode('utf-8')


    fs = int(np.ceil(fs))

    fSTFT, tSTFT, STFT_0 = sp.signal.spectrogram(x=wf_data,
                                                fs=fs,
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                #nfft=Length of the FFT used, if a zero padded FFT is desired
                                                nfft=nfft,
                                                scaling=scaling,
                                                axis=-1,
                                                mode=sgram_mode)

    if normed:
        STFT_norm = STFT_0 / np.median(STFT_0)  ##norm by median
    else:
        STFT_norm = STFT_0
    STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0)  ##convert to dB
    specMat = np.maximum(0, STFT_dB) #make sure nonnegative
    specMatsum = specMat.sum(1)


    return specMatsum,specMat,fSTFT

def getSpectraMedian(path_proj,cat00,k,station,normed=True):
    catk = cat00[cat00.Cluster == k]

    for j,evID in enumerate(catk.event_ID.iloc):

        if normed==False:
            matSum,specMat = getSpectra(evID,station,path_proj,normed=False)

        else:
            matSum,specMat = getSpectra(evID,station,path_proj,normed=True)

        if j == 0:
            specMatsum_med = np.zeros(len(matSum))


        specMatsum_med = np.vstack([specMatsum_med,matSum])



    specMatsum_med = np.median(specMatsum_med,axis=0)
    return specMatsum_med




def calcFFT(wf_data,lenData,fs,roll=100):
    '''


    Parameters
    ----------
    wf_data : np array
    lenData : number of samples
    fs : samploing rate
    roll : samples to calculate rolling average The default is 100.

    Returns
    -------
    rollingf : rolling average of frequency bins
    rollingFFT : rolling average of FFT

    '''

    # x = np.linspace(0.0, fs*lenData, lenData, endpoint=False)
    y = wf_data
    yf = fft(y)
    xf = fftfreq(lenData, 1/fs)[:lenData//2]
    real_fft = 2.0/lenData * np.abs(yf[0:lenData//2])

    df_spectra = pd.DataFrame({'fft':real_fft,
                               'f':xf})

    rollingFFT = df_spectra.fft.rolling(roll, min_periods=1).mean()
    rollingf = df_spectra.f.rolling(roll, min_periods=1).mean()



    return rollingf, rollingFFT


def getSgram(path_proj,evID,station,tSTFT=[0]):


    mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')

    specMat = mat.get('STFT')
    date = pd.to_datetime('200' + str(evID))

    x = [date + dtt.timedelta(seconds=i) for i in tSTFT]

    return specMat,x



def makeHourlyDF(ev_perhour_clus):

    """
    Returns dataframe of events binned by hour of day

    ev_perhour_resamp : pandas dataframe indexed by datetime

    """
    ev_perhour_resamp = ev_perhour_clus.resample('H').event_ID.count()



    hour_labels = list(ev_perhour_resamp.index.hour.unique())

    hour_labels.sort()
    #

    ev_perhour_resamp_list = list(np.zeros(len(hour_labels)))
    ev_perhour_mean_list = list(np.zeros(len(hour_labels)))




    hour_index = 0

    for ho in range(len(hour_labels)):
        hour_name = hour_labels[hour_index]
        ev_count = 0

#         print(hour_name)
        for ev in range(len(ev_perhour_resamp)):

            if ev_perhour_resamp.index[ev].hour == hour_name:
                ev_perhour_resamp_list[ho] += ev_perhour_resamp[ev]

                ev_count += 1

#         print(str(ev_count) + ' events in hour #' + str(hour_name))

        ev_perhour_mean_list[ho] = ev_perhour_resamp_list[ho] / ev_count

        hour_index += 1



##TS 2021/06/17 -- TS adjust hours here to CET
    hour_labels = [h + 2 for h in hour_labels]
    hour_labels[hour_labels==24] = 0
    hour_labels[hour_labels==25] = 1

    ev_perhour_resamp_df = pd.DataFrame({ 'EvPerHour' : ev_perhour_resamp_list,
                                          'MeanEvPerHour' : ev_perhour_mean_list},
                          index=hour_labels)


    ev_perhour_resamp_df['Hour'] = hour_labels



    return ev_perhour_resamp_df





def getDailyTempDiff(garciaDF_H,garciaDF_D,**plt_kwargs):

    tstart      =     plt_kwargs['tstartreal']
    tend        =     plt_kwargs['tendreal']

    garciaDF_H1 = garciaDF_H[garciaDF_H.datetime>=tstart]
    garciaDF_H1 = garciaDF_H1[garciaDF_H1.datetime<tend]

    garciaDF_D1 = garciaDF_D[garciaDF_D.datetime>=tstart]
    garciaDF_D1 = garciaDF_D1[garciaDF_D1.datetime<tend]


    temp_H = garciaDF_H1.temp_H.bfill()
    temp_H_a = np.array(temp_H)

    temp_H_a_r = temp_H_a.reshape(len(garciaDF_D1),24)
    mean_diff = []
    for i in range(len(temp_H_a_r[:,0])):
    #     plt.plot(temp_H_a_r[i,:] - garciaDF_D1.temp_D.iloc[i])
        mean_diff.append(temp_H_a_r[i,:] - garciaDF_D1.temp_D.iloc[i])


    mean_mean_diff = np.mean(mean_diff,axis=0)
    return mean_mean_diff
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getFP(evID,path_proj,outfile_name):
    
    with h5py.File(path_proj + outfile_name,'r') as MLout:

        fp = MLout['SpecUFEX_output/fprints'].get(str(evID))[:]
        
        return fp
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getFeatures(catalog,filetype,fmin,fmax,fs,path_WF,nfft,dataH5_path,station,channel):

    columns=['event_ID','datetime','datetime_index','Cluster','RSAM','SC','P2P','VAR']
    df = pd.DataFrame(columns=columns)
    # RSAM_norm = 0
    # P2P_norm = 0
    # SC_norm = 0
    # VAR_norm = 0
    # byCluster=1

    for i,evID in enumerate(catalog.event_ID):



        wf_filter = getWF(evID,dataH5_path,station,channel,fmin,fmax,fs)

        date = pd.to_datetime(catalog.datetime.iloc[i])
        cluster = catalog.Cluster.iloc[i]


        RSAM = np.log10(np.sum(np.abs(wf_filter)))
#         RSAM_norm = RSAM_norm + (RSAM / np.max(RSAM))
        sc = np.mean(librosa.feature.spectral_centroid(y=np.array(wf_filter), sr=fs))


        # f = np.fft.fft(wf_filter)
        # f_real = np.real(f)
        # mag_spec = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=len(f)+nfft*2)[0]
        # freqs = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=len(f)+nfft*2)[1]
        # dominant_freq = freqs[np.where(mag_spec == mag_spec.max())]




#         SC_norm = SC_norm + (sc / np.max(sc))
        p2p = np.log10(np.max(wf_filter) - np.min(wf_filter))
#         P2P_norm = P2P_norm + (p2p / np.max(p2p))
        var = np.var(wf_filter)
#         VAR_norm = VAR_norm + (var / np.max(var))
        kurt = kurtosis(wf_filter)

        df = df.append(
                  {'event_ID':evID,
                   'datetime':date,
                   'datetime_index':date,
                   'Cluster':cluster,
                   'log10RSAM':RSAM,
                   'SC':sc,
                   'log10P2P':p2p,
                   'VAR':var,
                   'kurt':kurt},
                   ignore_index=True)


    # df['RSAM_norm'] = [r/df.RSAM.max() for r in df.RSAM]
    # df['SC_norm'] = [r/df.SC.max() for r in df.SC]
    # df['P2P_norm'] = [r/df.P2P.max() for r in df.P2P]
    # df['VAR_norm'] = [r/df.RSAM.max() for r in df.VAR]

    df = df.set_index('datetime_index')

    return df




def getLocationFeatures(map_catalog,stn,station):

    columns=['event_ID','datetime','datetime_index','Cluster','Elevation_m','Depth_m','DistXY_m','DistXYZ_m']
    df_loc = pd.DataFrame(columns=columns)

    stnX = np.array(stn[stn.name=='G7'+station].X)[0]
    stnY = np.array(stn[stn.name=='G7'+station].Y)[0]

    for i,evID in enumerate(map_catalog.event_ID):

        XX = map_catalog.X_m.iloc[i]
        YY = map_catalog.Y_m.iloc[i]
        elev = map_catalog.Elevation_m.iloc[i]
        ZZ = map_catalog.Depth_m.iloc[i]

        # XXabs = np.abs(XX)
        # YYabs = np.abs(YY)
        # ZZabs = ZZ

        disstXY = spatial.distance.euclidean([XX,YY],[stnX,stnY])
        disstXYZ = spatial.distance.euclidean([XX,YY,ZZ],[stnX,stnY,0])


#         X.loc[index, f'distXY_m'] = disstXY
#         X.loc[index, f'depth_m'] = ZZ


        date = pd.to_datetime(map_catalog.datetime.iloc[i])
        cluster = map_catalog.Cluster.iloc[i]

        df_loc = df_loc.append(
                  {'event_ID':evID,
                   'datetime':date,
                   'datetime_index':date,
                   'Cluster':cluster,
                   'Elevation_m':elev,
                   'Depth_m':ZZ,
                    'DistXY_m':disstXY,
                   'DistXYZ_m':disstXYZ},
                    ignore_index=True)

    df_loc = df_loc.set_index('datetime_index')

    return df_loc




def getNMFOrder(W,numPatterns):
    maxColVal = np.zeros(numPatterns)
    maxColFreq = np.zeros(numPatterns)

    order = list(range(0,numPatterns))

    for j in range(len(W[1])):
        maxColFreq[j] = W[:,j].argmax()
        maxColVal[j] = W[:,j].max()

    #% make dict of rearranged NMF dict


    W_df = pd.DataFrame({'order':order,
                          'maxColFreq':maxColFreq,
                          'maxColVal':maxColVal
                          })

    W_df_sort = W_df.sort_values(by='maxColFreq')


    order_swap = list(W_df_sort.order)

    return order_swap


def resortByNMF(matrix,order_swap):

    matrix_new = matrix.copy()

    for o in range(matrix.shape[1]):

        o_swap = order_swap[o]

        matrix_new[:,o] = matrix[:,o_swap]

    return matrix_new





##################################################################################################
# ##################################################################################################
#        _           _            _
#       | |         | |          (_)
#    ___| |_   _ ___| |_ ___ _ __ _ _ __   __ _
#   / __| | | | / __| __/ _ \ '__| | '_ \ / _` |
#  | (__| | |_| \__ \ ||  __/ |  | | | | | (_| |
#   \___|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
#                                          __/ |
#                                         |___/
##################################################################################################



def linearizeFP(path_proj,outfile_name,cat00):

    X = []
    with h5py.File(path_proj + outfile_name,'r') as MLout:
        for evID in cat00.event_ID:
            fp = MLout['SpecUFEX_output/fprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PCAonFP(path_proj,outfile_name,cat00,numPCA=3,stand=True):
    ## performcs pca on fingerprints, returns catalog with PCs for each event
#returns a PCA sklearn object, a dataframe of cat00 but with columns for PCs, and a numpy array of PCs (N x numPC)


    X = linearizeFP(path_proj,outfile_name,cat00)


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    sklearn_pca = PCA(n_components=numPCA)

    Y_pca = sklearn_pca.fit_transform(X_st)

    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    pca_df = pd.DataFrame(data=Y_pca,columns=pc_cols,index=cat00.index)

    PCA_df = pd.concat([cat00,pca_df], axis=1)

    return sklearn_pca, PCA_df, Y_pca






# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PVEofPCA(path_proj,outfile_name,cat00,numPCMax=100,cum_pve_thresh=.8,stand='MinMax'):

    X = linearizeFP(path_proj,outfile_name,cat00)


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    numPCA_range = range(1,numPCMax)


    for numPCA in numPCA_range:

        sklearn_pca = PCA(n_components=numPCA)

        Y_pca = sklearn_pca.fit_transform(X_st)

        pve = sklearn_pca.explained_variance_ratio_

        cum_pve = pve.sum()
        print(numPCA,cum_pve)
        if cum_pve >= cum_pve_thresh:

            print('break')
            break



    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    PCA_df = pd.DataFrame(data = Y_pca, columns = pc_cols)


    return PCA_df, numPCA, cum_pve


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getTopFCat(cat,topF=1,startInd=0,distMeasure = "SilhScore"):
    """
    Make dataframe of most representative events


    Parameters
    ----------
    cat : pandas.DataFrame
        Catalog of events, must have 'Cluster', 'SS', and/or 'euc_dist' columns
    topf : int, default 1
        Number of top 'F' events in each cluster
    startInd : int, default 0
        Index to start counting top events
    distMeasure : {'SilhScore','EucDist'} default 'SilhScore'
        Type of clustering measure

    Returns
    -------
    cat_topF : pandas.DataFrame
        top {topF} of each cluster in catalog

    """

    cat_topF = pd.DataFrame();


    Kopt = np.max(cat.Cluster.unique())

    for k in range(1,Kopt+1):

        cat0 = cat.where(cat.Cluster==k).dropna();

        if distMeasure == "SilhScore":
            cat0 = cat0.sort_values(by='SS',ascending=False)

        if distMeasure == "EucDist":
            cat0 = cat0.sort_values(by='euc_dist',ascending=True)

        try:
            cat0 = cat0[startInd:startInd+topF]
        except: #if less than topF number events in cluster
            cat0 = cat0

            print(f"sampled all {len(cat0)} events in cluster!")


        cat_topF = cat_topF.append(cat0);


    ## sometimes these types get changed in the process?
    ## can comment out next two lines
    cat_topF['Cluster'] = [int(c) for c in cat_topF.Cluster];

    cat_topF['datetime_index'] = [pd.to_datetime(d) for d in cat_topF.datetime];

    cat_topF = cat_topF.sort_values(by =['Cluster','datetime'])

    cat_topF = cat_topF.set_index('datetime_index')


    return cat_topF



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def calcSilhScore(path_proj,outfile_name,cat00,range_n_clusters,numPCA,distMeasure = "SilhScore",Xtype='fingerprints',stand=True):
    """


    Parameters
    ----------

    range_n_clusters : range type - 2 : Kmax clusters
    numPCA : number of principal components to perform clustering on (if not on FPs)
    Xtype : cluster directly on fingerprints or components of PCA. The default is 'fingerprints'.


    Returns
    -------
    Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters.

    """

## Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters
## Returns altered cat00 dataframe with cluster labels and SS scores,
## Returns NEW catall dataframe with highest SS scores



## alt. X = 'PCA'

    if Xtype == 'fingerprints':
        X = linearizeFP(path_proj,outfile_name,cat00)
        pca_df = cat00
    elif Xtype == 'PCA':
        __, pca_df, X = PCAonFP(path_proj,outfile_name,cat00,numPCA=numPCA,stand=stand);

    maxSilScore = 0

    sse = []
    avgSils = []
    centers = []

    for n_clusters in range_n_clusters:

        print(f"kmeans on {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X)

        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)

        #save centroids
        centers.append(kmeans.cluster_centers_ )

        #kmeans loss function
        sse.append(kmeans.inertia_)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

#         %  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)

        # avgSil = np.median(sample_silhouette_values)

        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            euc_dist_best = euc_dist
            ss_best       = sample_silhouette_values


    print(f"Best cluster: {Kopt}")
    pca_df['Cluster'] = cluster_labels_best
    pca_df['SS'] = ss_best
    pca_df['euc_dist'] = euc_dist_best


    ## make df for  top SS score rep events
    catall = getTopFCat(pca_df,topF=1,startInd=0,distMeasure = distMeasure)







    return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



def compileSpectraFromWF(cat00,dataH5_path,station,channel,fmin,fmax):
    '''
    ## from summing spectrograms
    ## calc spectra from waveform for each event, from summing spectrograms
    ## put into dataframe with freqz as columns, with event IDs and datetimes


    Parameters
    ----------
    cat00 : TYPE
        DESCRIPTION.
    dataH5_path : TYPE
        DESCRIPTION.
    station : TYPE
        DESCRIPTION.
    channel : TYPE
        DESCRIPTION.
    fmin : TYPE
        DESCRIPTION.
    fmax : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''





    Xspec = []
    for i, evID in enumerate(cat00.event_ID):


        specMatsum,__,fSTFT = getSpectra_fromWF(evID,dataH5_path,station,channel,normed=True)

        fSTFT_trim, specMatsum_trim = trimSpectra(fSTFT,specMatsum,fmin,fmax)


        Xspec.append(specMatsum_trim)


    Xspec = np.array(Xspec)

    sgram_df = pd.DataFrame(Xspec,columns=fSTFT_trim)
    sgram_df['event_ID'] = [evID for evID in cat00.event_ID]

    return fSTFT_trim, Xspec, sgram_df
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



def trimSpectra(fSTFT,specMatsum,fmin,fmax):
    specMatsum_trim = specMatsum[fSTFT>fmin]
    fSTFT_trim      = fSTFT[fSTFT>fmin]

    specMatsum_trim = specMatsum_trim[fSTFT_trim<fmax]
    fSTFT_trim      = fSTFT_trim[fSTFT_trim<fmax]

    return fSTFT_trim, specMatsum_trim



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def KMeansSpectra(sgram_df,range_nclusters=range(2,10)):
    '''


    Parameters
    ----------
    sgram_df : pandas dataframe where columns 1:-2 are data, and the last col is event_ID
                    X = np.array(sgram_df.drop(labels=['event_ID'],axis=1))

    range_nclusters : range type, set to [K] to force K clusters,
                    DEFAULT try 2-10 clusters

    Returns
    -------
    None.

    '''
## take Kmeans of spectra, add cluster labels to sgram sf


    maxSilScore = 0
    sse = []
    avgSils = []
    centers = []

    X = np.array(sgram_df.drop(labels=['event_ID'],axis=1))

    for n_clusters in range_nclusters:

        print(f"kmeans on {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X)

        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl) + 1 for ccl in cluster_labels_0]

        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)

        #save centroids
        centers.append(kmeans.cluster_centers_ )

        #kmeans loss function
        sse.append(kmeans.inertia_)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        #%  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)
        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            euc_dist_best = euc_dist
            ss_best       = sample_silhouette_values


    print(f"Best cluster: {Kopt}")
    sgram_df['Cluster'] = cluster_labels_best
    sgram_df['SS'] = ss_best
    sgram_df['euc_dist'] = euc_dist_best



    return sgram_df



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo






def swapLabels(cat,A,B):
## swap label A to B
    dummy_variable = 999
    cat_swapped = cat.copy()
    cat_swapped.Cluster = cat_swapped.Cluster.replace(A,dummy_variable)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(B,A)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(dummy_variable,B)


    return cat_swapped




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo








# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.OTHERoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
############ OTHER
############ OTHER
############ OTHER
############ OTHER



def CalcScaleMAE(path_proj,cat00,Kopt,scale_range,RMM,sel_state,station,normed='median'):


    R2_all = []
    mae_Cbest= 1e12 #best of all clusters
    sca_Cbest = 1
    for k in range(1,Kopt+1):
#     for k in [2]:

        print(k)
        mae_min = 1e12 #best within a cluster
        sca_keep = 1


        if normed=='median':
            specMatsum_med_orig=getSpectraMedian(path_proj,cat00,k,station,normed=True)


        rec_state = RMM[:,sel_state[k-1]]

        if normed == 'max':
            specMatsum_med_orig=getSpectraMedian(path_proj,cat00,k,station,normed=True)
            specMatsum_med_orig = specMatsum_med_orig / np.max(specMatsum_med_orig)
            rec_state = rec_state / np.max(rec_state)

        for i, sca in enumerate(scale_range):

            specMatsum_med = specMatsum_med_orig * sca


#             mae_temp = r2_score(specMatsum_med, rec_state)
            mae_temp = sklearn.metrics.mean_absolute_error(specMatsum_med, rec_state)


            if mae_temp < mae_min:

                mae_min = mae_temp
                sca_keep = sca

        if mae_min < mae_Cbest:

            kmax = k
            mae_Cbest = mae_min
            sca_Cbest = sca_keep

            print(k, ': ', mae_Cbest, sca_Cbest)

        R2_k=r2_score(specMatsum_med, rec_state)

        R2_all.append(R2_k)

    return mae_Cbest,sca_Cbest, R2_all, kmax






##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def CalcDiffPeak(path_proj,cat00,k,RMM,sel_state,station):


    specMatsum=getSpectraMedian(path_proj,cat00,k,station,normed=True)
    rec_state = RMM[:,sel_state[k-1]]



    maxIDR = np.argwhere(rec_state==np.max(rec_state))
    maxIDS = np.argwhere(specMatsum==np.max(specMatsum))

    peak_rec_state = rec_state[maxIDR]
    peak_spec      = specMatsum[maxIDS]

    scale = peak_rec_state - peak_spec

    return int(peak_rec_state), int(peak_spec), int(scale)
