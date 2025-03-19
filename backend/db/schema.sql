-- Template table for raw price data
-- Each ticker will have its own partition
CREATE TABLE IF NOT EXISTS raw_data_template (
    ticker_id INTEGER NOT NULL,  -- References tickers.id
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC(10,4) NOT NULL,
    high NUMERIC(10,4) NOT NULL,
    low NUMERIC(10,4) NOT NULL,
    close NUMERIC(10,4) NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker_id, timestamp)
) PARTITION BY LIST (ticker_id);

-- Create index on timestamp for each partition
CREATE INDEX IF NOT EXISTS raw_data_template_timestamp_idx ON raw_data_template(timestamp);

-- Create index on ticker_id for partitioning
CREATE INDEX IF NOT EXISTS raw_data_template_ticker_id_idx ON raw_data_template(ticker_id); 