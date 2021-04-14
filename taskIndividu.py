import pandas as pd
from math import sqrt

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
    jika umur kendaraan kurang dari satu tahun maka nilai cell akan dirubah menjadi 1
    jika umur kendaraan lebih dari satu tahun dan kurang dari dua tahun maka nilai cell akan dirubah menjadi 0.5
    jika umur kendaraan lebih dari dua tahun maka nilai cell akan dirubah menjadi 0
    '''
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '< 1 Tahun', 1)
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '1-2 Tahun', 0.5)
    file['Umur_Kendaraan'] = file.Umur_Kendaraan.mask(file.Umur_Kendaraan == '> 2 Tahun', 0)
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
    '''
    Jika kendaraan tidak pernah rusak maka nilai cell akan dirubah menjadi 1
    Jika kendaraan pernah rusak maka nilai cell akan dirubah menjadi 0
    '''
    file['Kendaraan_Rusak'] = file.Kendaraan_Rusak.mask(file.Kendaraan_Rusak == 'Tidak', 1)
    file['Kendaraan_Rusak'] = file.Kendaraan_Rusak.mask(file.Kendaraan_Rusak == 'Pernah', 0)
    return file
def scaling(file):
    """
    Melakukan Normalisasi menggunakan metode Min-Max Normalization. Karena Column Umur,
    Kode_Daerah, Premi, Kanal_Penjualan, dan Lama_Berlangganan belum berada di range
    antara 0 s/d 1 maka perlu discaling terlebih dahulu.
    """
    file['Umur'] = (file['Umur']-file['Umur'].min())/(file['Umur'].max()-file['Umur'].min())
    file['Kode_Daerah'] = (file['Kode_Daerah'] - file['Kode_Daerah'].min()) / (file['Kode_Daerah'].max() - file['Kode_Daerah'].min())
    file['Premi'] = (file['Premi'] - file['Premi'].min()) / (file['Premi'].max() - file['Premi'].min())
    file['Kanal_Penjualan'] = (file['Kanal_Penjualan'] - file['Kanal_Penjualan'].min()) / (file['Kanal_Penjualan'].max() - file['Kanal_Penjualan'].min())
    file['Lama_Berlangganan'] = (file['Lama_Berlangganan'] - file['Lama_Berlangganan'].min()) / (file['Lama_Berlangganan'].max() - file['Lama_Berlangganan'].min())

def euclidian(x1,y1,x2,y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

########MAIN########
tr,te = readData()
missingValuesHandler(tr)
transformKendaraanRusak(tr)
transformJenisKelamin(tr)
transformUmurKendaraan(tr)
scaling(tr)
tr.to_csv('train.csv')
