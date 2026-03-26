import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. TAM ZIRHLI VERİ OKUMA FONKSİYONU (Excel Uyumlu Hale Getirildi)
def veriyi_guvenli_oku(dosya_yolu):
    """
    Excel (.xlsx) dosyalarını doğrudan okur. 
    Eğer dosya CSV olsaydı encoding denemesi gerekirdi ama Excel için 
    pd.read_excel kullanmak yeterlidir.
    """
    try:
        # Excel dosyaları için doğrudan pd.read_excel kullanılır.
        df = pd.read_excel(dosya_yolu)
        print(f"✅ BİLGİ: '{dosya_yolu}' Excel formatında başarıyla okundu!")
        return df
    except Exception as e:
        # Eğer dosya gerçekten bozuksa veya Excel değilse hata verir
        raise ValueError(f"Dosya okunamadı! Hata: {e}")

# --- ANA PROGRAM ---
dosya_adi = "CarSales.xlsx"

try:
    # Zırhlı okuma fonksiyonumuzu çağırıyoruz
    df = veriyi_guvenli_oku(dosya_yolu=dosya_adi)
    
    # Kolon isimlerindeki gizli boşlukları temizleyelim
    df.columns = df.columns.str.strip()

    # B Maddesi: Çoklu Doğrusal Regresyon (Multiple Linear Regression)
    X_kolonlari = ['Engine_size', 'Curb_weight'] 
    y_kolonu = 'Price_in_thousands' 
    
    # Sadece analizde kullanacağımız kolonları alıyoruz
    model_verisi = df[X_kolonlari + [y_kolonu]]
    
    # Eksik verileri (NaN) veri setinden atıyoruz
    model_verisi = model_verisi.dropna()
    son_satir = len(model_verisi)
    print(f"🧹 Eksik veriler temizlendi. (Model için {son_satir} satır kullanılacak)")
    
    # Matrisleri (X ve y) oluşturma
    X = model_verisi[X_kolonlari].values
    y = model_verisi[y_kolonu].values

    # Veriyi bölme (%80 Eğitim, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli Eğitme (Fit)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train) 
    
    # Bulunan Katsayıları Gösterme
    print("\n--- Model Katsayıları ---")
    print(f"Kesişim Noktası (Intercept): {reg_model.intercept_:.4f}")
    for kolon, katsayi in zip(X_kolonlari, reg_model.coef_):
        print(f"{kolon} Katsayısı: {katsayi:.4f}")

    # Test Verisi ile Başarıyı Ölçme
    y_tahmin = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_tahmin)
    r2 = r2_score(y_test, y_tahmin)
    
    print("\n--- Model Başarı Metrikleri ---")
    print(f"MSE (Hata Kareler Ortalaması): {mse:.4f}")
    print(f"R-Kare (R2 Score): {r2:.4f}")

    # ---------------------------------------------------------
    # 2. 3 BOYUTLU GÖRSELLEŞTİRME (GÖRÜNÜRLÜK İÇİN GÜNCELLENDİ)
    # ---------------------------------------------------------
    print("\nGrafik oluşturuluyor ve kaydediliyor, lütfen bekleyin...")
    
    fig = plt.figure(figsize=(10, 8))
    # 3D grafik oluşturmak için projection='3d' kullanıyoruz
    ax = fig.add_subplot(111, projection='3d')

    # Test verilerini 3 boyutlu uzayda kırmızı noktalar olarak çizdiriyoruz
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', s=50, label='Gerçek Fiyatlar (Test)')

    # Modelin öğrendiği "Tahmin Düzlemini" (Regression Plane) oluşturmak için bir zemin/ızgara hazırlıyoruz
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    x1_surf, x2_surf = np.meshgrid(np.linspace(x1_min, x1_max, 20), 
                                   np.linspace(x2_min, x2_max, 20))

    # Oluşturduğumuz bu ızgaradaki her bir noktanın Z eksenindeki (Fiyat) karşılığını modelimize tahmin ettiriyoruz
    X_surf = np.c_[x1_surf.ravel(), x2_surf.ravel()]
    y_surf = reg_model.predict(X_surf).reshape(x1_surf.shape)

    # Tahmin düzlemini mavi ve yarı saydam (alpha=0.4) olarak çizdiriyoruz
    ax.plot_surface(x1_surf, x2_surf, y_surf, color='blue', alpha=0.4)

    # Eksen isimlendirmeleri
    ax.set_title('Motor Hacmi ve Ağırlığa Göre Fiyat Tahmini')
    ax.set_xlabel('Motor Hacmi (Engine_size)')
    ax.set_ylabel('Araç Ağırlığı (Curb_weight)')
    ax.set_zlabel('Fiyat (Bin Dolar)')
    ax.legend()

    # --- YENİ EKLENEN KISIM: Grafiği Dosya Olarak Kaydetme ---
    grafik_adi = "3D_Regresyon_Plani.png"
    plt.savefig(grafik_adi, dpi=300) # dpi=300 yüksek çözünürlük için
    print(f"✅ GÖRSEL ÇIKTI: Grafik '{grafik_adi}' adıyla projenin olduğu klasöre kaydedildi.")

    # --- GÜNCELLEME: Bazı Mac'lerde pencerenin açılmasına yardımcı olmak için ---
    print("Grafik penceresi açılmaya çalışılıyor (Eğer açılmazsa kaydedilen dosyayı kontrol et)...")
    plt.show() # Pencereyi açmaya çalış

except KeyError as e:
    print(f"❌ HATA: {e} isimli sütun bulunamadı. Lütfen veri setindeki kolon başlıklarını kontrol et.")
except Exception as e:
    print(f"❌ Beklenmedik bir hata oluştu: {e}")