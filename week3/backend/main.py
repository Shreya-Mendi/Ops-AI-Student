from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import data

app = FastAPI(title="NYC Cab Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    """
    Kubernetes readiness probe — checks that loaded data passed validation.
    Returns 200 if data is clean or degraded-but-usable; 503 if load failed entirely.
    """
    validation = data._validate_demand_df_cached()
    if validation["load_failed"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "data_load_failed"},
        )
    return {
        "status": "ready",
        "data_valid": validation["is_valid"],
        "num_issues": validation["num_issues"],
        "issues": [i["type"] for i in validation["issues"]],
        "degraded": not validation["is_valid"],
    }


@app.get("/api/heatmap")
def heatmap(
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    holiday: str = Query("regular"),
):
    return data.get_heatmap(hour, dow, holiday, date)


@app.get("/api/forecast")
def forecast(
    zone_id: int = Query(...),
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    steps: int = Query(16, ge=1, le=96),
):
    return data.forecast_demand(zone_id, hour, dow, steps, date)


@app.get("/api/recommendations")
def recommendations(
    zone_id: int = Query(...),
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    n: int = Query(3, ge=1, le=10),
    holiday: str = Query("regular"),
):
    return data.get_recommendations(zone_id, hour, dow, n, holiday, date)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
