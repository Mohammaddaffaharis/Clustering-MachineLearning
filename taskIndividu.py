import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math

def readData():
    '''
    Digunakan untuk membaca file .csv dan merubahnya menjadi dataframe
    '''
    fileTrain = pd.read_csv('kendaraan_train.csv')
    fileTest = pd.read_csv('kendaraan_test.csv')
    return fileTrain,fileTest
##PRE-PROCESSING
def missingValuesHandler(file):
    '''
    Digunakan untuk mengisi cell yang bernilai null. Untuk Kolom Jenis_Kelamin,
    SIM, Kode_Daerah, Sudah_Asuransi, Umur_Kendaraan, Kendaraan_Rusak, dan
    Kanal_Penjualan menggunakan Modus. Sedangkan untuk Umur, Premi, dan
    Lama_Berlangganan menggunakan Mean.
    '''
    modeJenisKelamin = file['Jenis_Kelamin'].mode()
    file['Jenis_Kelamin'].fillna(modeJenisKelamin[0], inplace=True)
    meanUmur = file['Umur'].mean()
    file['Umur'].fillna(meanUmur, inplace = True)
    modeSIM = file['SIM'].mode()
    file['SIM'].fillna(modeSIM[0], inplace=True)
    modeKodeDaerah = file['Kode_Daerah'].mode()
    file['Kode_Daerah'].fillna(modeKodeDaerah[0], inplace=True)
    modeSudahAsuransi = file['Sudah_Asuransi'].mode()
    file['Sudah_Asuransi'].fillna(modeSudahAsuransi[0], inplace=True)
    modeUmurKendaraan = file['Umur_Kendaraan'].mode()
    file['Umur_Kendaraan'].fillna(modeUmurKendaraan[0], inplace=True)
    modeKendaraanRusak = file['Kendaraan_Rusak'].mode()
    file['Kendaraan_Rusak'].fillna(modeKendaraanRusak[0], inplace=True)
    meanPremi = file['Premi'].mean()
    file['Premi'].fillna(meanPremi, inplace=True)
    modeKanalPenjualan = file['Kanal_Penjualan'].mode()
    file['Kanal_Penjualan'].fillna(modeKanalPenjualan[0], inplace=True)
    meanLamaBerlangganan = file['Lama_Berlangganan'].mean()
    file['Lama_Berlangganan'].fillna(meanLamaBerlangganan, inplace=True)
def transformUmurKendaraan(file):
    '''
    jika umur kendaraan kurang dari satu tahun maka nilai cell akan dirubah menjadi 3
    jika umur kendaraan lebih dari satu tahun dan kurang dari dua tahun maka nilai cell akan dirubah menjadi 2
    jika umur kendaraan lebih dari dua tahun maka nilai cell akan dirubah menjadi 1
    '''
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '< 1 Tahun', 3)
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '1-2 Tahun', 2)
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '> 2 Tahun', 1)
    return file
def transformJenisKelamin(file):
    '''                                                               isPria   isWanita
    Jika Jenis Kelamin Pria maka nilai cell akan dirubah menjadi   [    1    ,    0    ]
    Jika Jenis Kelamin Wanita maka nilai cell akan dirubah menjadi [    0    ,    1    ]
    '''
    pria = []
    wanita = []
    for index, row in file.iterrows():
        if row['Jenis_Kelamin'] == 'Pria':
            pria.append(1)
            wanita.append(0)
        else:
            pria.append(0)
            wanita.append(1)
    file['isPria'] = pria
    file['isWanita'] = wanita
    return file
def transformKendaraanRusak(file):
    '''                                                              isRusak  isTidakRusak
    Jika Jenis Kelamin Pria maka nilai cell akan dirubah menjadi   [    1    ,    0    ]
    Jika Jenis Kelamin Wanita maka nilai cell akan dirubah menjadi [    0    ,    1    ]
    '''
    rusak = []
    tidak = []
    for index, row in file.iterrows():
        if row['Kendaraan_Rusak'] == 'Tidak':
            tidak.append(1)
            rusak.append(0)
        else:
            tidak.append(0)
            rusak.append(1)
    file['isRusak'] = rusak
    file['isTidakRusak'] = tidak
    return file
def scaling(file):
    """
    Melakukan Normalisasi menggunakan metode Min-Max Normalization. Karena Column Umur,
    Kode_Daerah, Umur_Kendaraan, Premi, Kanal_Penjualan, dan Lama_Berlangganan belum berada di range
    antara 0 s/d 1 maka perlu discaling terlebih dahulu.
    """
    file['Umur'] = (file['Umur']-file['Umur'].min())/(file['Umur'].max()-file['Umur'].min())
    file['Kode_Daerah'] = (file['Kode_Daerah'] - file['Kode_Daerah'].min()) / (file['Kode_Daerah'].max() - file['Kode_Daerah'].min())
    file['Umur_Kendaraan'] = (file['Umur_Kendaraan'] - file['Umur_Kendaraan'].min()) / (file['Umur_Kendaraan'].max() - file['Umur_Kendaraan'].min())
    file['Premi'] = (file['Premi'] - file['Premi'].min()) / (file['Premi'].max() - file['Premi'].min())
    file['Kanal_Penjualan'] = (file['Kanal_Penjualan'] - file['Kanal_Penjualan'].min()) / (file['Kanal_Penjualan'].max() - file['Kanal_Penjualan'].min())
    file['Lama_Berlangganan'] = (file['Lama_Berlangganan'] - file['Lama_Berlangganan'].min()) / (file['Lama_Berlangganan'].max() - file['Lama_Berlangganan'].min())
def preProcessing(file):
    #panggil fungsi pre-processing yang sudah dibuat sebelumnya
    missingValuesHandler(file)
    transformKendaraanRusak(file)
    transformJenisKelamin(file)
    transformUmurKendaraan(file)
    scaling(file)
    #convert feature Umur_Kedaraan menjadi float
    file["Umur_Kendaraan"] = pd.to_numeric(file["Umur_Kendaraan"])
    #membuat dataframe baru berdasarkan proses preprocessing yang telah dibuat
    newFile = pd.DataFrame(file, columns=['isPria', 'Umur', 'Kode_Daerah', 'Sudah_Asuransi',
                                        'Umur_Kendaraan', 'isRusak', 'Kanal_Penjualan',
                                        'Lama_Berlangganan'])
    #Dimensionality Reduction
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(newFile)
    principal_df = pd.DataFrame(data=principal_components)
    return principal_df

def euclidian(x1,y1,x2,y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
def initialCentroid(k):
    cent = {
        i + 1: [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]
        for i in range(k)
    }
    return cent
def assignNodeToCentroid(file, cent, colmap):
    #melakukan perhitungan jarak untuk setiap titik terhadap setiap centroid
    for i in cent.keys():
        arr = []
        for index, row in file.iterrows():
            arr.append(euclidian(row[0], row[1], cent[i][0], cent[i][1]))
        file['Jarak_Dari_Centroid{}'.format(i)] = arr
    #melakukan listing terhadap kolom jarak centroid ke titik
    colJarakTitikCentroid = []
    for i in cent.keys():
        colJarakTitikCentroid.append('Jarak_Dari_Centroid{}'.format(i))
    #mencari centroid terdekat ke titik
    file['Centroid_Terdekat'] = file.loc[:, colJarakTitikCentroid].idxmin(axis=1)
    file['Centroid_Terdekat'] = file['Centroid_Terdekat'].map(lambda x: int(x.lstrip('Jarak_Dari_Centroid')))
    file['Color'] = file['Centroid_Terdekat'].map(lambda x: colmap[x])
    return file
def updateCentroid(file,cent):
    for i in cent.keys():
        cent[i][0] = np.mean(file[file['Centroid_Terdekat'] == i][0])
        cent[i][1] = np.mean(file[file['Centroid_Terdekat'] == i][1])
    return cent

########MAIN########
tr,te = readData()
dataframe = preProcessing(te)
dataframe.to_csv('DataAfterPreProcessing.csv')
plt.scatter(dataframe[0], dataframe[1],color='red', alpha=0.5, s=1.5)
plt.title('Data After Pre-Processing')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

k = 2
colmap = {1: '#9e2828', 2: '#289e37', 3: '#28439e', 4: '#29fff8', 5: '#ffa600', 6: '#ffff00', 7: '#e100ff', 8:'#787878'}
cent = initialCentroid(k)
dataframe = assignNodeToCentroid(dataframe, cent, colmap)
while True:
    closest_centroids = dataframe['Centroid_Terdekat'].copy(deep=True)
    cent = updateCentroid(dataframe,cent)
    dataframe = assignNodeToCentroid(dataframe, cent, colmap)
    if closest_centroids.equals(dataframe['Centroid_Terdekat']):
        break
print(cent)
plt.scatter(dataframe[0], dataframe[1], color=dataframe['Color'], alpha=0.5, s=1.5)
plt.title('Data After Clustering')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
for i in range(k):
    plt.scatter(cent[i+1][0], cent[i+1][1], color=colmap[i+1],marker='X',s=40,edgecolors='k')
plt.show()
dataframe.to_csv('DataAfterClustering.csv')