"""
api/main.py: Phase 8.3, the forecast as a web service (FastAPI).

Start it from the project folder:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs  -> a free, clickable test page FastAPI builds for you.

Why an API? Any system (the Streamlit app, a till system, a phone app) can ask for a
forecast over the web without knowing anything about Python or the model.
"""
import time
from datetime import date as Date
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse  # sends the visitor to another page
from pydantic import BaseModel, Field

from src import config
from src.predict import forecast_day, load_artifacts, summary

app = FastAPI(title="Uxbridge Lounge Demand Forecaster",
              description="Daily food and drink demand forecast, recommended prep and stockout risk.",
              version="1.0.0")


class ForecastRequest(BaseModel):
    """What the caller must send. Pydantic checks it automatically and rejects bad input (error 422)."""
    date: Date = Field(..., examples=["2026-07-04"])
    weather: Literal["Sunny", "Cloudy", "Rainy", "Snow"] = Field(..., examples=["Sunny"])
    temperature_c: float = Field(..., ge=-10, le=35, examples=[22])
    festival: Literal[tuple(config.FESTIVALS)] = "None"
    local_event: Literal[tuple(config.LOCAL_EVENTS)] = "None"
    planned_prep: Optional[Dict[str, int]] = Field(None, examples=[{"Masala Dosa": 25}])


@app.get("/", include_in_schema=False)
def home():
    """The bare web address (e.g. the Vercel link) opens the clickable API test page."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Is the service up and are the models loaded? Used by monitoring tools."""
    meta = load_artifacts()["meta"]
    return {"status": "ok", "model_trained_at": meta["trained_at"], "test_MAE": meta["test_MAE"]}


@app.get("/menu")
def menu():
    """The 24 items the model can forecast."""
    return load_artifacts()["meta"]["menu"]


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """Forecast every menu item for one day."""
    start = time.perf_counter()
    try:
        table = forecast_day(req.date, req.weather, req.temperature_c, req.festival,
                             req.local_event, req.planned_prep)
    except ValueError as err:  # input the model can't handle
        raise HTTPException(status_code=422, detail=str(err))
    return {"date": str(req.date), "summary": summary(table), "items": table.to_dict(orient="records"),
            "latency_ms": round(1000 * (time.perf_counter() - start), 1)}
