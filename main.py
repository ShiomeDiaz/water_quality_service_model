# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from datetime import date  # ¡Faltaba esta importación!

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
    pH: float
    E_coli: float
    Coliformes_totales: float
    Turbidez: float
    Nitratos: float
    Fosfatos: float
    DBO5: float
    Solidos_suspendidos: float

# Endpoint 1: Predicción de ICA
@app.post("/predict-next-day", summary="Predice el ICA para el día siguiente")
async def predict_next_day(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        # Crear DataFrame con datos de entrada
        input_dict = input_data.dict()
        fecha = input_dict.pop('fecha')
        
        # Calcular características temporales
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        dia_año = fecha_dt.timetuple().tm_yday
        mes = fecha_dt.month
        dia_semana = fecha_dt.weekday()
        
        features = {
            **input_dict,
            'dia_año': dia_año,
            'mes': mes,
            'dia_semana': dia_semana
        }
        
        # Crear dataframe para predicción
        prediction_df = pd.DataFrame([features])
        
        # Realizar predicción
        prediction = model.predict(prediction_df)
        
        return {
            "location_id": input_data.Location_ID,
            "fecha_prediccion": fecha,
            "ica_predicho": round(float(prediction[0]), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3035)
# To run the API, use the command:
# uvicorn apiModel:app --reload