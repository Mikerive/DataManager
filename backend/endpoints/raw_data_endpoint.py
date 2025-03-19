from fastapi import APIRouter
from backend.db.RawData import RawData

router = APIRouter()

class RawDataEndpoint:
    @router.get('/raw_data')
    async def get_raw_data(self):
        return await RawData.fetch_all()

    @router.get('/raw_data/{item_id}')
    async def get_raw_data_by_id(self, item_id: int):
        return await RawData.fetch_by_id(item_id)

    @router.post('/raw_data')
    async def create_raw_data(self, item_data: dict):
        await RawData.create(item_data)
        return {'message': 'Item created successfully'}

    @router.put('/raw_data/{item_id}')
    async def update_raw_data(self, item_id: int, item_data: dict):
        await RawData.update(item_id, item_data)
        return {'message': 'Item updated successfully'}

    @router.delete('/raw_data/{item_id}')
    async def delete_raw_data(self, item_id: int):
        await RawData.delete(item_id)
        return {'message': 'Item deleted successfully'}

# Include the router in the main app
# app.include_router(router)
