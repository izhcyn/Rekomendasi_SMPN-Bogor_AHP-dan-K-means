import numpy as np

class AHP:
    def __init__(self, matriks_perbandingan_berpasangan):
        self.matriks_perbandingan_berpasangan = matriks_perbandingan_berpasangan
        self.vektor_prioritas = self.hitung_vektor_prioritas(matriks_perbandingan_berpasangan)
        self.rasio_konsistensi = self.periksa_konsistensi(matriks_perbandingan_berpasangan, self.vektor_prioritas)

    def normalisasi(self, matriks):
        jumlah_kolom = np.sum(matriks, axis=0)
        matriks_ternormalisasi = matriks / jumlah_kolom[np.newaxis, :]
        return matriks_ternormalisasi

    def hitung_vektor_prioritas(self, matriks_perbandingan_berpasangan):
        matriks_ternormalisasi = self.normalisasi(matriks_perbandingan_berpasangan)
        vektor_prioritas = np.mean(matriks_ternormalisasi, axis=1)
        return vektor_prioritas

    def periksa_konsistensi(self, matriks_perbandingan_berpasangan, vektor_prioritas):
        vektor_jumlah_berbobot = np.dot(matriks_perbandingan_berpasangan, vektor_prioritas)
        lambda_max = np.mean(vektor_jumlah_berbobot / vektor_prioritas)
        ci = (lambda_max - len(vektor_prioritas)) / (len(vektor_prioritas) - 1)
        ri = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        cr = ci / ri[len(vektor_prioritas)]
        return cr

    def nilai_alternatif(self, alternatif):
        if not isinstance(alternatif, np.ndarray):
            alternatif = np.array(alternatif)
        
        alternatif_ternormalisasi = self.normalisasi(alternatif)
        skor = np.dot(alternatif_ternormalisasi, self.vektor_prioritas)
        return skor
