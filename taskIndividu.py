import pandas as pd
from math import sqrt

def readData():
    fileTrain = pd.read_csv('kendaraan_train.csv')
    fileTest = pd.read_csv('kendaraan_test.csv')
    return fileTrain,fileTest

def missingValuesHandler(file):
    modeJenisKelamin = file['Jenis_Kelamin'].mode()
    file.Jenis_Kelamin.fillna(modeJenisKelamin[0], inplace=True)
    meanUmur = file['Umur'].mean()
    file.Umur.fillna(meanUmur, inplace = True)
    modeSIM = file['SIM'].mode()
    file.SIM.fillna(modeSIM[0], inplace=True)
    modeKodeDaerah = file['Kode_Daerah'].mode()
    file.Kode_Daerah.fillna(modeKodeDaerah[0], inplace=True)
    modeSudahAsuransi = file['Sudah_Asuransi'].mode()
    file.Sudah_Asuransi.fillna(modeSudahAsuransi[0], inplace=True)
    modeUmurKendaraan = file['Umur_Kendaraan'].mode()
    file.Umur_Kendaraan.fillna(modeUmurKendaraan[0], inplace=True)
    modeKendaraanRusak = file['Kendaraan_Rusak'].mode()
    file.Kendaraan_Rusak.fillna(modeKendaraanRusak[0], inplace=True)
    meanPremi = file['Premi'].mean()
    file.Premi.fillna(meanPremi, inplace=True)
    modeKanalPenjualan = file['Kanal_Penjualan'].mode()
    file.Kanal_Penjualan.fillna(modeKanalPenjualan[0], inplace=True)
    meanLamaBerlangganan = file['Lama_Berlangganan'].mean()
    file.Lama_Berlangganan.fillna(meanLamaBerlangganan, inplace=True)

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
    '''
    Jika kendaraan tidak pernah rusak maka nilai cell akan dirubah menjadi 2
    Jika kendaraan pernah rusak maka nilai cell akan dirubah menjadi 1
    '''
    file['Kendaraan_Rusak'] = file.Kendaraan_Rusak.mask(file.Kendaraan_Rusak == 'Tidak', 2)
    file['Kendaraan_Rusak'] = file.Kendaraan_Rusak.mask(file.Kendaraan_Rusak == 'Pernah', 1)
    return file

tr,te = readData()
missingValuesHandler(tr)
tr.to_csv('train.csv')
