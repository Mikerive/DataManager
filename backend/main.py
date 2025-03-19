from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.db.Database import Database
from backend.endpoints.raw_data_endpoint import router as raw_data_router
from backend.endpoints.processed_data_endpoint import router as processed_data_router
from backend.endpoints.tickers_endpoint import router as tickers_router

db = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Include routers
app.include_router(raw_data_router)
app.include_router(processed_data_router)
app.include_router(tickers_router)

@app.get('/health')
async def health_check():
    return {'status': 'healthy'}