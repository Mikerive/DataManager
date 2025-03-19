from fastapi import APIRouter
from db import ProcessedData

router = APIRouter()

class ProcessedDataEndpoint:
    
    @router.get('/processed_data/{item_id}')
    async def get_processed_data_by_id(self, item_id: int):
        return await ProcessedData.fetch_by_id(item_id)

    @router.post('/processed_data')
    async def create_processed_data(self, item_data: dict):
        await ProcessedData.create(item_data)
        return {'message': 'Item created successfully'}

    @router.put('/processed_data/{item_id}')
    async def update_processed_data(self, item_id: int, item_data: dict):
        await ProcessedData.update(item_id, item_data)
        return {'message': 'Item updated successfully'}

    @router.delete('/processed_data/{item_id}')
    async def delete_processed_data(self, item_id: int):
        await ProcessedData.delete(item_id)
        return {'message': 'Item deleted successfully'}

# Include the router in the main app
# app.include_router(router)
