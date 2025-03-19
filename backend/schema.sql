-- Enable TimescaleDB extension (must be run in the database first)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ==========================
-- 🏦 Main Table: Tickers
-- ==========================
-- Note: Ticker symbols should be stored in uppercase to match Tiingo API format
-- Example: 'AAPL', 'SPY', 'MSFT'
CREATE TABLE tickers (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,  -- e.g., 'AAPL', 'SPY' (uppercase)
    company_name TEXT NOT NULL,
    ipo_date DATE,
    sector TEXT,
    industry TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================
-- 📈 Raw Data Table (Partitioned by Ticker)
-- Stores OHLCV data with timestamps per ticker
-- Matches Tiingo IEX API format: https://api.tiingo.com/iex/<ticker>/prices
-- ==========================
CREATE TABLE raw_data_template (
    id SERIAL PRIMARY KEY,
    ticker_id INT NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,  -- Matches Tiingo's timestamp field
    quote_timestamp TIMESTAMPTZ,     -- Matches Tiingo's quoteTimestamp field
    last_sale_timestamp TIMESTAMPTZ, -- Matches Tiingo's lastSaleTimeStamp field
    last DECIMAL(10, 4),            -- Matches Tiingo's last field
    last_size INT,                  -- Matches Tiingo's lastSize field
    tngo_last DECIMAL(10, 4),       -- Matches Tiingo's tngoLast field
    prev_close DECIMAL(10, 4),      -- Matches Tiingo's prevClose field
    open DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's open field
    high DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's high field
    low DECIMAL(10, 4) NOT NULL,    -- Matches Tiingo's low field
    mid DECIMAL(10, 4),             -- Matches Tiingo's mid field
    volume BIGINT NOT NULL,         -- Matches Tiingo's volume field
    bid_size INT,                   -- Matches Tiingo's bidSize field
    bid_price DECIMAL(10, 4),       -- Matches Tiingo's bidPrice field
    ask_size INT,                   -- Matches Tiingo's askSize field
    ask_price DECIMAL(10, 4),       -- Matches Tiingo's askPrice field
    UNIQUE (ticker_id, timestamp) ON CONFLICT (ticker_id, timestamp) DO UPDATE SET 
        quote_timestamp = EXCLUDED.quote_timestamp,
        last_sale_timestamp = EXCLUDED.last_sale_timestamp,
        last = EXCLUDED.last,
        last_size = EXCLUDED.last_size,
        tngo_last = EXCLUDED.tngo_last,
        prev_close = EXCLUDED.prev_close,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        mid = EXCLUDED.mid,
        volume = EXCLUDED.volume,
        bid_size = EXCLUDED.bid_size,
        bid_price = EXCLUDED.bid_price,
        ask_size = EXCLUDED.ask_size,
        ask_price = EXCLUDED.ask_price
) PARTITION BY LIST (ticker_id);

-- Convert to hypertable for efficient time-series querying
SELECT create_hypertable('raw_data_template', 'timestamp');
-- Index for fast lookups
CREATE INDEX idx_raw_data_ticker_time ON raw_data_template (ticker_id, timestamp DESC);

-- ==========================
-- 📊 Processed Data Table (Partitioned by Ticker)
-- Stores different bar calculations based on raw data
-- ==========================
CREATE TABLE processed_data_template (
    id SERIAL PRIMARY KEY,
    ticker_id INT NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    bar_type TEXT NOT NULL CHECK (bar_type IN ('hourly', 'daily', 'volume', 'entropy', 'tick')),
    last DECIMAL(10, 4),            -- Matches Tiingo's last field
    last_size INT,                  -- Matches Tiingo's lastSize field
    tngo_last DECIMAL(10, 4),       -- Matches Tiingo's tngoLast field
    prev_close DECIMAL(10, 4),      -- Matches Tiingo's prevClose field
    open DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's open field
    high DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's high field
    low DECIMAL(10, 4) NOT NULL,    -- Matches Tiingo's low field
    mid DECIMAL(10, 4),             -- Matches Tiingo's mid field
    volume BIGINT NOT NULL,         -- Matches Tiingo's volume field
    bid_size INT,                   -- Matches Tiingo's bidSize field
    bid_price DECIMAL(10, 4),       -- Matches Tiingo's bidPrice field
    ask_size INT,                   -- Matches Tiingo's askSize field
    ask_price DECIMAL(10, 4),       -- Matches Tiingo's askPrice field
    UNIQUE (ticker_id, timestamp, bar_type) ON CONFLICT (ticker_id, timestamp, bar_type) DO UPDATE SET 
        last = EXCLUDED.last,
        last_size = EXCLUDED.last_size,
        tngo_last = EXCLUDED.tngo_last,
        prev_close = EXCLUDED.prev_close,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        mid = EXCLUDED.mid,
        volume = EXCLUDED.volume,
        bid_size = EXCLUDED.bid_size,
        bid_price = EXCLUDED.bid_price,
        ask_size = EXCLUDED.ask_size,
        ask_price = EXCLUDED.ask_price
) PARTITION BY LIST (ticker_id, bar_type);

-- Convert to hypertable
SELECT create_hypertable('processed_data_template', 'timestamp');
-- Index for fast lookups
CREATE INDEX idx_processed_data_ticker_time ON processed_data_template (ticker_id, timestamp DESC);

-- ==========================
-- 📉 Indicator Data Table (Partitioned by Ticker)
-- Stores technical indicator calculations
-- ==========================
CREATE TABLE indicator_data_template (
    id SERIAL PRIMARY KEY,
    ticker_id INT NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    bar_type TEXT NOT NULL CHECK (bar_type IN ('hourly', 'daily', 'volume', 'entropy', 'tick')),
    indicator_name TEXT NOT NULL CHECK (indicator_name IN ('Williams %R', 'SMA_50', 'SMA_200', 'MACD')),
    indicator_value NUMERIC NOT NULL,
    UNIQUE (ticker_id, timestamp, bar_type, indicator_name) ON CONFLICT (ticker_id, timestamp, bar_type, indicator_name) DO UPDATE SET 
        indicator_value = EXCLUDED.indicator_value
) PARTITION BY LIST (ticker_id, bar_type, indicator_name);

-- Convert to hypertable
SELECT create_hypertable('indicator_data_template', 'timestamp');
