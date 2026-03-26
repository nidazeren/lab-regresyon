import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. VERİ OKUMA ADIMI (Düzeltildi)
# ---------------------------------------------------------
dosya_adi = 'CarSales.xlsx'

try:
    # Excel dosyaları için pd.read_excel kullanılır. 
    # Not: Bilgisayarında 'openpyxl' kütüphanesi yüklü olmalıdır.
    df = pd.read_excel(dosya_adi)
    print("Harika! Veri seti başarıyla yüklendi.\n")
except FileNotFoundError:
    print(f"HATA: '{dosya_adi}' adlı dosya bulunamadı!")
    exit()
except Exception as e:
    print(f"Beklenmeyen bir hata oluştu: {e}")
    exit()

# ---------------------------------------------------------
# 2. SÜTUN SEÇİMİ VE VERİ TEMİZLİĞİ (Dokunulmadı)
# ---------------------------------------------------------
df.columns = df.columns.str.strip()

X_column = 'Horsepower'
y_column = 'Price_in_thousands'

if X_column not in df.columns or y_column not in df.columns:
    print(f"HATA: Sütunlar bulunamadı! Mevcut sütunlar: {df.columns.tolist()}")
    exit()

df_temiz = df.dropna(subset=[X_column, y_column])

X = df_temiz[[X_column]]
y = df_temiz[y_column]

print(f"Temizlenmiş veri sayısı: {len(df_temiz)}\n")

# ---------------------------------------------------------
# 3. EĞİTİM VE TEST VERİSİNİ AYIRMA (Dokunulmadı)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. MODELİ OLUŞTURMA VE EĞİTME (Dokunulmadı)
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("Model başarıyla eğitildi!\n")

# ---------------------------------------------------------
# 5. TAHMİN VE DEĞERLENDİRME (Dokunulmadı)
# ---------------------------------------------------------
y_tahmin = model.predict(X_test)

mse = mean_squared_error(y_test, y_tahmin)
r2 = r2_score(y_test, y_tahmin)

print("--- MODEL BAŞARI SONUÇLARI ---")
print(f"Hata Kareler Ortalaması (MSE): {mse:.2f}")
print(f"R-Kare Skoru (Başarı Oranı): {r2:.2f}\n")

# ---------------------------------------------------------
# 6. GÖRSELLEŞTİRME (Dokunulmadı)
# ---------------------------------------------------------
plt.scatter(X_test, y_test, color='red', label='Gerçek Fiyatlar')
plt.plot(X_test, y_tahmin, color='blue', linewidth=2, label='Tahmin Doğrusu')

plt.title('Beygir Gücü vs Fiyat')
plt.xlabel('Beygir Gücü (HP)')
plt.ylabel('Fiyat (Bin Dolar)')
plt.legend()
plt.grid(True)
plt.show()