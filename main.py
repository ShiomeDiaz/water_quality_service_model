# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List
from datetime import timedelta

app = FastAPI(title="API de Predicción de Calidad del Agua")

# Cargar modelo y datos
try:
    model = joblib.load('modelo_final_ICA.pkl')
    df = pd.read_csv('dataset_con_ICA_completo.csv', parse_dates=['Date'])
    print("✅ Modelo y datos cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar recursos: {e}")
    model = None
    df = pd.DataFrame()

# Modelo Pydantic para predicciones
class PredictionInput(BaseModel):
    Location_ID: int
    fecha: date
    features_0_to_10: List[float]  # [0.93, 0.38, ..., 0.38] (11 valores)
    pH: float
    E_coli: float
    Coliformes_totales: float
    Turbidez: float
    Nitratos: float
    Fosfatos: float
    DBO5: float
    Solidos_suspendidos: float

# Endpoint de predicción
@app.post("/predict-next-day")
async def predict_next_day(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        # Convertir fecha a características temporales
        fecha_dt = datetime.combine(input_data.fecha, datetime.min.time())
        
        # Crear diccionario con todas las características
        features = {
            "Location_ID": input_data.Location_ID,
            "dia_año": fecha_dt.timetuple().tm_yday,
            "mes": fecha_dt.month,
            "dia_semana": fecha_dt.weekday(),
            **{str(i): val for i, val in enumerate(input_data.features_0_to_10)},
            "pH": input_data.pH,
            "E_coli": input_data.E_coli,
            "Coliformes_totales": input_data.Coliformes_totales,
            "Turbidez": input_data.Turbidez,
            "Nitratos": input_data.Nitratos,
            "Fosfatos": input_data.Fosfatos,
            "DBO5": input_data.DBO5,
            "Solidos_suspendidos": input_data.Solidos_suspendidos
        }
        
        # Crear DataFrame para predicción
        prediction_df = pd.DataFrame([features])
        
        # Reordenar columnas como el modelo espera
        column_order = ['Location_ID', 'dia_año', 'mes', 'dia_semana', 
                        '0','1','2','3','4','5','6','7','8','9','10',
                        'pH','E_coli','Coliformes_totales','Turbidez',
                        'Nitratos','Fosfatos','DBO5','Solidos_suspendidos']
        prediction_df = prediction_df[column_order]
        
        # Realizar predicción
        prediction = model.predict(prediction_df)
        # Dentro del endpoint, después de obtener la fecha:
        fecha_prediccion = input_data.fecha + timedelta(days=1)
        
        return {
            "location_id": input_data.Location_ID,
            "fecha_prediccion": fecha_prediccion.isoformat(),
            "ica_predicho": round(float(prediction[0]), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3035)
