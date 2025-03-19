backend/
├── data_loader/      # Handles data loading operations
├── data_processor/   # Processes loaded data
├── db/               # Database models and connection management
│   ├── database.py   # Manages DB connections with retry logic
│   ├── raw_data.py   # Raw data DB operations
│   ├── processed_data.py  # Processed data DB operations
│   ├── indicator_data.py  # Indicator data DB operations
│   └── tickers.py    # Ticker management
├── endpoints/        # API endpoints/routes
├── utils/            # Utility functions
└── main.py           # Application entry point (FastAPI)
