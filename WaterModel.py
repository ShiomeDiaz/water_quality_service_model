# models/WaterModel.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import os

# 1. Verificar existencia del dataset
ruta_dataset = './dataset_con_ICA_completo.csv'
if not os.path.exists(ruta_dataset):
    raise FileNotFoundError(f"❌ El archivo {ruta_dataset} no existe. Verifica la ruta.")

# 2. Cargar y preparar datos
try:
    df = pd.read_csv(ruta_dataset, parse_dates=['Date'])
    print("✅ Dataset cargado correctamente")
    print(f"📊 Muestras: {len(df)}, Características: {len(df.columns)}")
except Exception as e:
    print(f"❌ Error al cargar el dataset: {str(e)}")
    exit()

# 3. Ingeniería de características temporales
df['dia_año'] = df['Date'].dt.dayofyear
df['mes'] = df['Date'].dt.month
df['dia_semana'] = df['Date'].dt.dayofweek

# 4. Configurar características y variable objetivo
features = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'pH', 'E_coli', 'Coliformes_totales', 'Turbidez', 
    'Nitratos', 'Fosfatos', 'DBO5', 'Solidos_suspendidos',
    'dia_año', 'mes', 'dia_semana', 'Location_ID'
]

target = 'ICA_calculado'

# Verificar columnas existentes
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise KeyError(f"❌ Columnas faltantes: {missing}")

# 5. Preprocesamiento
preprocessor = ColumnTransformer([
    ('encoder', OneHotEncoder(handle_unknown='ignore'), ['Location_ID'])
], remainder='passthrough')

# 6. Pipeline del modelo
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ))
])

# 7. Validación temporal
tscv = TimeSeriesSplit(n_splits=5)
print("\n🔍 Validación cruzada temporal:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
    X_train, X_test = df.iloc[train_idx][features], df.iloc[test_idx][features]
    y_train, y_test = df.iloc[train_idx][target], df.iloc[test_idx][target]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Fold {fold}: RMSE = {rmse:.2f}, R² = {r2:.4f}")

# 8. Entrenar modelo final
print("\n🚀 Entrenando modelo final...")
model.fit(df[features], df[target])
print("✅ Modelo entrenado")

# 9. Función de predicción
def predecir_proximo_dia():
    """Predice el ICA para todas las ubicaciones al día siguiente"""
    ultima_fecha = df['Date'].max()
    nuevas_fechas = pd.date_range(
        start=ultima_fecha + pd.DateOffset(1), 
        periods=len(df['Location_ID'].unique()), 
        freq='D'
    )
    
    nuevo_df = pd.DataFrame({
        'Date': nuevas_fechas,
        'Location_ID': df['Location_ID'].unique()
    })
    
    # Mantener valores de otras características (último registro conocido)
    for col in features:
        if col not in ['Date', 'Location_ID']:
            nuevo_df[col] = df.groupby('Location_ID')[col].last().values
    
    # Calcular características temporales
    nuevo_df['dia_año'] = nuevo_df['Date'].dt.dayofyear
    nuevo_df['mes'] = nuevo_df['Date'].dt.month
    nuevo_df['dia_semana'] = nuevo_df['Date'].dt.dayofweek
    
    return nuevo_df, model.predict(nuevo_df[features])

# 10. Ejemplo de uso
print("\n🔮 Generando predicciones...")
nuevos_datos, predicciones = predecir_proximo_dia()
resultados = pd.DataFrame({
    'Location_ID': nuevos_datos['Location_ID'],
    'Fecha_prediccion': nuevos_datos['Date'],
    'ICA_predicho': predicciones
})

print("\n📅 Predicciones para el día siguiente:")
print(resultados.head())

# 11. Importancia de características (opcional)
try:
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importancia = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': model.named_steps['regressor'].feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("\n📊 Características más importantes:")
    print(importancia.head(10))
except Exception as e:
    print(f"\n⚠️ No se pudo obtener importancia de características: {str(e)}")

# Guardar modelo (opcional)
# joblib.dump(model, 'modelo_ICA.pkl')
# print("✅ Modelo guardado como modelo_ICA.pkl")
# Guardar resultados de predicción
resultados.to_csv('predicciones_ICA.csv', index=False, encoding='utf-8-sig')
print("✅ Resultados de predicción guardados como predicciones_ICA.csv")

# Obtener métricas de entrenamiento
y_train_pred = model.predict(df[features])
train_rmse = np.sqrt(mean_squared_error(df[target], y_train_pred))
train_r2 = r2_score(df[target], y_train_pred)

print(f"Entrenamiento - RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"Validación - RMSE: {0.74:.4f}, R²: {0.9915:.4f}")  # Tus resultados

# Comparar con métricas de validación
print("\n🔍 Comparación de métricas:")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, df[features], df[target], cv=tscv,
    scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Entrenamiento')
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validación')
plt.xlabel('Muestras de Entrenamiento')
plt.ylabel('R²')
plt.title('Curvas de Aprendizaje')
plt.legend()
plt.show()
# Mostrar métricas finales
print("\n📈 Métricas finales del modelo:")
print(f"RMSE de entrenamiento: {train_rmse:.4f}")
print(f"R² de entrenamiento: {train_r2:.4f}")
print(f"RMSE de validación: {0.74:.4f}")  # Tus resultados
print(f"R² de validación: {0.9915:.4f}")  # Tus resultados
# Guardar el modelo final
import joblib
joblib.dump(model, 'modelo_final_ICA.pkl')
print("✅ Modelo final guardado como modelo_final_ICA.pkl")
# Guardar resultados de predicción
resultados.to_csv('predicciones_ICA.csv', index=False, encoding='utf-8-sig')
print("✅ Resultados de predicción guardados como predicciones_ICA.csv")

# Verificar si hay características con varianza casi cero
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # 1% de varianza mínima
X_filtered = selector.fit_transform(df[features])

print(f"Características eliminadas: {len(features) - X_filtered.shape[1]}")

# Dividir en datos totalmente independientes (ej: últimos 30 días)
test_size = 30
X_train, X_test = df.iloc[:-test_size][features], df.iloc[-test_size:][features]
y_train, y_test = df.iloc[:-test_size][target], df.iloc[-test_size:][target]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R² en datos futuros: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE en datos futuros: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
