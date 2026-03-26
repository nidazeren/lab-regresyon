import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. VERİ OKUMA VE GELİŞMİŞ GÜVENLİK (ROBUSTNESS)
# Dosya adını buraya yazıyoruz. Eğer senin dosyanın adı farklıysa (örneğin 'Car_sales.csv') burayı değiştirmelisin.
dosya_adi = 'CarSales.xlsx'

try:
    # HATA BURADAYDI: pd.read_csv yerine pd.read_excel kullanıyoruz
    df = pd.read_excel(dosya_adi)
    print("Harika! Veri seti başarıyla yüklendi!")
except FileNotFoundError:
    print("-" * 40)
    print(f"HATA: '{dosya_adi}' bulunamadı.")
    print("Python şu an çalıştığı klasörde bu dosyayı göremiyor.")
    print("\nŞu an bulunduğunuz klasördeki dosyalar şunlar:")
    # os.listdir('.') komutu bulunduğumuz dizindeki her şeyi listeler
    for mevcut_dosya in os.listdir('.'):
        print(f"  -> {mevcut_dosya}")
    print("-" * 40)
    print("Lütfen yukarıdaki listeden verinizin GÜNCEL ADINI kopyalayıp koddaki 'dosya_adi' değişkenine yapıştırın.")
    exit()
except Exception as e:
    print(f"HATA: Beklenmeyen bir sorun oluştu: {e}")
    exit()

# 2. VERİ TEMİZLEME VE HAZIRLIK
df_clean = df[['Horsepower', 'Price_in_thousands']].dropna()

X = df_clean[['Horsepower']].values
y = df_clean['Price_in_thousands'].values

# 3. EĞİTİM VE TEST VERİSİNİ AYIRMA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. POLİNOMİYAL DÖNÜŞÜM (Degree=3)
poly_converter = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_converter.fit_transform(X_train)
X_test_poly = poly_converter.transform(X_test)

# 5. MODELİ EĞİTME
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. TAHMİN VE BAŞARI ÖLÇÜMÜ
y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("POLİNOMİYAL REGRESYON (Derece=3) SONUÇLARI")
print(f"Ortalama Kare Hatası (MSE): {mse:.2f}")
print(f"R-Kare (R-Squared) Değeri : {r2:.2f}")
print("-" * 30)

# 7. GÖRSELLEŞTİRME
plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='blue', label='Gerçek Test Verileri')

X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_smooth_poly = poly_converter.transform(X_smooth)
y_smooth_pred = model.predict(X_smooth_poly)

plt.plot(X_smooth, y_smooth_pred, color='red', linewidth=3, label='Polinomiyal Model (Derece=3)')

plt.title('Polinomiyal Regresyon: Beygir Gücü vs Fiyat', fontsize=14)
plt.xlabel('Beygir Gücü (Horsepower)', fontsize=12)
plt.ylabel('Fiyat (Bin Dolar)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()