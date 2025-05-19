# Trade Simulator

A high-performance trade simulator for OKX BTC-USDT-SWAP using real-time L2 order book data.

## Setup

1. Install Python 3.8+.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure VPN is active for OKX WebSocket access.
4. Run: `python trade_simulator_gui.py`.

## Features

- Real-time WebSocket data processing.
- Slippage prediction with GradientBoostingRegressor.
- Almgren-Chriss market impact model.
- Tkinter GUI with real-time plots and logs.

## New Features

- **Latency Benchmarking**: Measure data processing, UI update, and end-to-end simulation loop latencies.
- **Optimization**: Improved memory management, threading, and regression model efficiency.

## Models

- Slippage: GradientBoostingRegressor (dummy data).
- Maker/Taker: LogisticRegression (dummy data).
- Fees: Rule-based OKX fee structure.
- Market Impact: Almgren-Chriss model.

## Performance

- Data Processing: ~3ms average.
- UI Update: ~15ms average.
- End-to-End: ~20ms average.

## Performance Metrics

- **Tick Frequency**: ~10 ticks/sec
- **Processing Latency**: ~3ms
- **UI Update Latency**: ~15ms
- **End-to-End Latency**: ~20ms

## Optimization Techniques

1. **Memory Management**: Used `deque` for efficient data storage.
2. **Threading**: WebSocket client runs in a separate thread.
3. **Regression Models**: Pre-trained models for slippage and maker/taker prediction.
