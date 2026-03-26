import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import io

def run_ridge_regression():
    # 1. ADIM: VERİYİ OKUMA (Excel Formatı İçin Düzeltildi)
    file_name = "CarSales.xlsx"
    
    try:
        # Excel dosyasını CSV gibi zorlamak yerine, kendi özel fonksiyonuyla açıyoruz.
        df = pd.read_excel(file_name)
        df.columns = df.columns.str.strip()
        print("✅ Veri seti başarıyla yüklendi!")
    except Exception as e:
        print(f"❌ Hata: Dosya hiçbir yöntemle kurtarılamadı. Hata detayı: {e}")
        return

    # 2. ADIM: VERİ TEMİZLİĞİ (DATA CLEANING)
    features = ['Horsepower', 'Engine_size', 'Curb_weight']
    target = 'Price_in_thousands'
    
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"❌ Hata: Şu sütunlar veri setinde bulunamadı: {missing_cols}")
        print(f"Mevcut sütunlar şunlar: {df.columns.tolist()}")
        return

    # Eksik verileri (NaN) siliyoruz.
    df_cleaned = df[features + [target]].dropna()
    print(f"✅ Eksik veriler temizlendi. Model {len(df_cleaned)} adet araç verisiyle eğitilecek.\n")

    X = df_cleaned[features]
    y = df_cleaned[target]

    # 3. ADIM: VERİYİ EĞİTİM VE TEST OLARAK BÖLME
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. ADIM: FARKLI ALPHA DEĞERLERİNİ TEST ETME (RIDGE REGRESYON)
    alphas_to_test = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    mse_scores = []
    r2_scores = []

    print("--- RIDGE REGRESYON ALPHA TEST SONUÇLARI ---")
    for alpha_val in alphas_to_test:
        ridge_model = Ridge(alpha=alpha_val)
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        
        print(f"Alpha: {alpha_val:6.2f} | MSE (Hata): {mse:7.2f} | R-Squared (Başarı): {r2:5.4f}")

    # 5. ADIM: GRAFİKSEL GÖSTERİM (GÖRSELLEŞTİRME)
    plt.figure(figsize=(10, 5))
    plt.plot(alphas_to_test, mse_scores, marker='o', linestyle='-', color='teal', linewidth=2)
    plt.title('Ridge Regresyon: Alpha (Ceza) Değerine Karşı Modelin MSE (Hata) Değişimi', fontsize=14)
    plt.xlabel('Alpha Değeri (Ceza Şiddeti)', fontsize=12)
    plt.ylabel('Ortalama Kare Hatası (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ridge_regression()
