import asyncio
import websockets
import json
import logging
import time
import numpy as np
from collections import deque
import threading
import queue
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

WEBSOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"

# --- Global Configuration ---
CONFIG = {
    "exchange": "OKX",
    "spot_asset": "BTC-USDT-SWAP",
    "order_type": "market",
    "quantity_usd": 100.0,
    "volatility_annualized_config": 0.5,
    "fee_tier": 1,
    "ac_total_time_to_execute_H_hours": 1.0,
    "ac_num_trades_N": 10,
    "ac_risk_aversion_lambda": 1e-6,
    "ac_eta_temp_impact_coeff": 0.1,
    "ac_gamma_perm_impact_coeff": 0.1,
}

# --- Data Structures ---
MAX_DATA_POINTS = 1000
price_data = deque(maxlen=MAX_DATA_POINTS)
volume_data = deque(maxlen=MAX_DATA_POINTS)
mid_price_data = deque(maxlen=MAX_DATA_POINTS)
order_book_imbalance_data = deque(maxlen=MAX_DATA_POINTS)

# --- Model Paths ---
SLIPPAGE_MODEL_PATH = "slippage_model.joblib"
MAKER_TAKER_MODEL_PATH = "maker_taker_model.joblib"


# --- Model Initialization ---
def train_slippage_model() -> GradientBoostingRegressor:
    """Train a dummy slippage model for demonstration."""
    X = pd.DataFrame(
        {
            "spread_bps": np.random.rand(1000) * 10,
            "ask_price_1": np.random.rand(1000) * 50000,
            "bid_price_1": np.random.rand(1000) * 50000,
            "ask_qty_1": np.random.rand(1000) * 10,
            "bid_qty_1": np.random.rand(1000) * 10,
            "quantity_asset": np.random.rand(1000) * 0.1,
            "obi": np.random.uniform(-1, 1, 1000),
        }
    )
    y = X["spread_bps"] * 0.01 + np.random.normal(0, 0.01, 1000)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, SLIPPAGE_MODEL_PATH)
    return model


def train_maker_taker_model() -> LogisticRegression:
    """Train a dummy maker/taker model for demonstration."""
    X = pd.DataFrame(
        {
            "obi": np.random.uniform(-1, 1, 1000),
            "vol": np.random.rand(1000) * 0.01,
            "spread_bps": np.random.rand(1000) * 10,
        }
    )
    y = np.random.randint(0, 2, 1000)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MAKER_TAKER_MODEL_PATH)
    return model


slippage_model_regressor = None
try:
    slippage_model_regressor = joblib.load(SLIPPAGE_MODEL_PATH)
    logging.info(f"Loaded slippage model from {SLIPPAGE_MODEL_PATH}")
except FileNotFoundError:
    logging.warning(f"Slippage model not found. Training new model.")
    slippage_model_regressor = train_slippage_model()

maker_taker_logistic_model = None
try:
    maker_taker_logistic_model = joblib.load(MAKER_TAKER_MODEL_PATH)
    logging.info(f"Loaded maker/taker model from {MAKER_TAKER_MODEL_PATH}")
except FileNotFoundError:
    logging.warning(f"Maker/Taker model not found. Training new model.")
    maker_taker_logistic_model = train_maker_taker_model()


# --- Helper Functions ---
def fetch_daily_volume(symbol: str = "BTC-USDT-SWAP") -> float:
    """Fetch daily trading volume from OKX API."""
    try:
        url = f"https://www.okx.com/api/v5/market/history-trade?instId={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            trades = response.json()["data"]
            total_volume_usd = sum(
                float(trade["sz"]) * float(trade["px"]) for trade in trades
            )
            return total_volume_usd * (24 * 3600 / len(trades))  # Scale to daily
        return 1_000_000_000  # Fallback
    except Exception as e:
        logging.error(f"Error fetching daily volume: {e}")
        return 1_000_000_000


def extract_slippage_features(
    asks: List[Tuple[str, str]],
    bids: List[Tuple[str, str]],
    quantity_usd: float,
    mid_price: float,
    order_book_imbalance: float,
) -> pd.DataFrame:
    """Extract features for slippage model."""
    if not asks or not bids or mid_price == 0:
        return pd.DataFrame(
            [[0] * 7],
            columns=[
                "spread_bps",
                "ask_price_1",
                "bid_price_1",
                "ask_qty_1",
                "bid_qty_1",
                "quantity_asset",
                "obi",
            ],
        )

    try:
        ask_price_1 = float(asks[0][0])
        bid_price_1 = float(bids[0][0])
        ask_qty_1 = float(asks[0][1])
        bid_qty_1 = float(bids[0][1])
        spread = ask_price_1 - bid_price_1
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        quantity_asset = quantity_usd / mid_price if mid_price > 0 else 0

        return pd.DataFrame(
            [
                {
                    "spread_bps": spread_bps,
                    "ask_price_1": ask_price_1,
                    "bid_price_1": bid_price_1,
                    "ask_qty_1": ask_qty_1,
                    "bid_qty_1": bid_qty_1,
                    "quantity_asset": quantity_asset,
                    "obi": order_book_imbalance,
                }
            ]
        )
    except (ValueError, IndexError) as e:
        logging.warning(f"Error extracting slippage features: {e}")
        return pd.DataFrame(
            [[0] * 7],
            columns=[
                "spread_bps",
                "ask_price_1",
                "bid_price_1",
                "ask_qty_1",
                "bid_qty_1",
                "quantity_asset",
                "obi",
            ],
        )


def predict_expected_slippage_reg(
    asks: List[Tuple[str, str]],
    bids: List[Tuple[str, str]],
    quantity_usd: float,
    mid_price: float,
    order_book_imbalance: float,
) -> float:
    """Predict slippage using regression model or fallback to walk-the-book."""
    if slippage_model_regressor:
        features_df = extract_slippage_features(
            asks, bids, quantity_usd, mid_price, order_book_imbalance
        )
        try:
            if not features_df.empty:
                return float(max(0, slippage_model_regressor.predict(features_df)[0]))
        except Exception as e:
            logging.error(f"Error in slippage prediction: {e}")

    return calculate_expected_slippage_walk_book(asks, bids, quantity_usd)


def calculate_expected_slippage_walk_book(
    asks: List[Tuple[str, str]], bids: List[Tuple[str, str]], quantity_usd: float
) -> float:
    """Calculate slippage by walking the order book."""
    if not asks or not bids:
        return 0.0

    try:
        mid_price = (float(asks[0][0]) + float(bids[0][0])) / 2
        if mid_price == 0:
            return 0.0
        target_quantity_asset = quantity_usd / mid_price

        filled_quantity = 0
        cost = 0
        for price_str, quantity_str in asks:
            price = float(price_str)
            quantity = float(quantity_str)
            if filled_quantity + quantity >= target_quantity_asset:
                needed_quantity = target_quantity_asset - filled_quantity
                cost += needed_quantity * price
                filled_quantity += needed_quantity
                break
            else:
                cost += quantity * price
                filled_quantity += quantity

        if filled_quantity < target_quantity_asset:
            logging.warning(
                f"Not enough liquidity: Filled {filled_quantity}/{target_quantity_asset}"
            )
            return (
                float("inf")
                if filled_quantity == 0
                else (cost / filled_quantity - float(asks[0][0]))
                / float(asks[0][0])
                * 100
            )

        avg_fill_price = cost / filled_quantity
        entry_price = float(asks[0][0])
        return (
            ((avg_fill_price - entry_price) / entry_price) * 100
            if entry_price > 0
            else 0.0
        )
    except (ValueError, IndexError) as e:
        logging.warning(f"Error in walk-the-book slippage: {e}")
        return 0.0


def calculate_expected_fees(
    quantity_usd: float, fee_tier: int, order_type: str = "market"
) -> float:
    """Calculate fees based on OKX fee structure."""
    taker_fee_rates = {1: 0.001, 2: 0.0008, 3: 0.0006, 4: 0.0004}
    fee_rate = (
        taker_fee_rates.get(fee_tier, 0.001)
        if order_type.lower() == "market"
        else 0.001
    )
    return quantity_usd * fee_rate


def annualize_tick_volatility(
    tick_volatility: float,
    ticks_per_second: float = 1,
    trading_days_per_year: int = 252,
    hours_per_day: int = 24,
) -> float:
    """Annualize per-tick volatility."""
    if tick_volatility == 0:
        return 0.0
    seconds_per_year = trading_days_per_year * hours_per_day * 60 * 60
    ticks_per_year = ticks_per_second * seconds_per_year
    return tick_volatility * np.sqrt(ticks_per_year)


def calculate_market_impact_almgren_chriss(
    quantity_usd: float,
    mid_price: float,
    tick_volatility: float,
    daily_volume_usd: float,
    config: Dict,
) -> float:
    """Calculate market impact using Almgren-Chriss model."""
    if mid_price == 0 or daily_volume_usd == 0:
        return 0.0

    try:
        X = quantity_usd / mid_price
        sigma_annualized = annualize_tick_volatility(
            tick_volatility, ticks_per_second=1
        )
        if sigma_annualized == 0:
            sigma_annualized = config.get("volatility_annualized_config", 0.5)

        H_hours = config.get("ac_total_time_to_execute_H_hours", 1.0)
        N = config.get("ac_num_trades_N", 10)
        risk_aversion = config.get("ac_risk_aversion_lambda", 1e-6)
        eta = config.get("ac_eta_temp_impact_coeff", 0.1)
        T_days = H_hours / 24.0

        if T_days == 0 or N == 0:
            return 0.0

        cost_variance = (risk_aversion / 2) * (X**2) * (sigma_annualized**2) * T_days
        cost_temporary_impact = (
            eta * (X**2) / (N * T_days) if N * T_days > 0 else float("inf")
        )
        total_impact_cost_usd = (cost_variance + cost_temporary_impact) * mid_price

        return total_impact_cost_usd if total_impact_cost_usd != float("inf") else 0.0
    except Exception as e:
        logging.error(f"Error in Almgren-Chriss: {e}")
        return 0.0


def predict_maker_taker_proportion(
    asks: List[Tuple[str, str]],
    bids: List[Tuple[str, str]],
    mid_price: float,
    order_book_imbalance: float,
    tick_volatility: float,
) -> Dict[str, float]:
    """Predict maker/taker proportion."""
    if maker_taker_logistic_model and mid_price > 0:
        try:
            spread_bps = ((float(asks[0][0]) - float(bids[0][0])) / mid_price) * 10000
            features = pd.DataFrame(
                [
                    {
                        "obi": order_book_imbalance,
                        "vol": tick_volatility,
                        "spread_bps": spread_bps,
                    }
                ]
            )
            prob = maker_taker_logistic_model.predict_proba(features)[0]
            return {"maker": prob[0], "taker": prob[1]}
        except Exception as e:
            logging.warning(f"Error in maker/taker prediction: {e}")

    return {"maker": 0.0, "taker": 1.0}


# --- Latency Benchmarking ---
def benchmark_latency(latencies: deque) -> Dict[str, float]:
    """Calculate latency statistics."""
    if not latencies:
        return {"avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}
    return {
        "avg": np.mean(latencies),
        "max": np.max(latencies),
        "min": np.min(latencies),
        "count": len(latencies),
    }


class RealTradeSimulator:
    def __init__(
        self, output_queue: queue.Queue, config_override: Optional[Dict] = None
    ):
        self.output_queue = output_queue
        self.config = CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        self.daily_volume_usd_estimate = fetch_daily_volume(self.config["spot_asset"])
        self._is_running = False
        self._stop_event = asyncio.Event()
        self._thread = None
        self._loop = None
        self.mid_price_data = deque(maxlen=MAX_DATA_POINTS)
        self.order_book_imbalance_data = deque(maxlen=MAX_DATA_POINTS)
        self._tick_timestamps = deque(maxlen=100)  # For tick frequency
        self._processing_latencies = deque(maxlen=1000)  # For latency benchmarking

    async def _websocket_client_async(self):
        """WebSocket client with reconnection logic."""
        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    WEBSOCKET_URL, ping_interval=20
                ) as websocket:
                    logging.info("Connected to WebSocket")
                    self._is_running = True
                    retry_delay = 1
                    while not self._stop_event.is_set():
                        try:
                            start_time = time.perf_counter()
                            message_str = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0
                            )
                            self._tick_timestamps.append(time.time())
                            data = json.loads(message_str)

                            if (
                                "asks" not in data
                                or "bids" not in data
                                or "timestamp" not in data
                            ):
                                continue

                            asks = [
                                (float(p), float(q)) for p, q in data.get("asks", [])
                            ]
                            bids = [
                                (float(p), float(q)) for p, q in data.get("bids", [])
                            ]
                            timestamp = data.get("timestamp")

                            if asks and bids:
                                current_mid_price = (asks[0][0] + bids[0][0]) / 2
                                self.mid_price_data.append(current_mid_price)
                                total_bid_volume = sum(q for _, q in bids[:10])
                                total_ask_volume = sum(q for _, q in asks[:10])
                                current_order_book_imbalance = (
                                    (total_bid_volume - total_ask_volume)
                                    / (total_bid_volume + total_ask_volume)
                                    if total_bid_volume + total_ask_volume > 0
                                    else 0
                                )
                                self.order_book_imbalance_data.append(
                                    current_order_book_imbalance
                                )
                            else:
                                current_mid_price = 0
                                current_order_book_imbalance = 0

                            min_volatility_window = 100
                            tick_volatility = 0.0
                            if len(self.mid_price_data) >= min_volatility_window:
                                prices = np.array(
                                    list(self.mid_price_data)[-min_volatility_window:]
                                )
                                log_returns = np.log(prices[1:] / prices[:-1])
                                log_returns = log_returns[np.isfinite(log_returns)]
                                if len(log_returns) > 1:
                                    tick_volatility = np.std(log_returns)

                            expected_slippage_pct = predict_expected_slippage_reg(
                                asks,
                                bids,
                                self.config["quantity_usd"],
                                current_mid_price,
                                current_order_book_imbalance,
                            )
                            expected_fees_usd = calculate_expected_fees(
                                self.config["quantity_usd"],
                                self.config["fee_tier"],
                                self.config["order_type"],
                            )
                            expected_market_impact_usd = (
                                calculate_market_impact_almgren_chriss(
                                    self.config["quantity_usd"],
                                    current_mid_price,
                                    tick_volatility,
                                    self.daily_volume_usd_estimate,
                                    self.config,
                                )
                            )
                            slippage_cost_usd = (
                                expected_slippage_pct / 100
                            ) * self.config["quantity_usd"]
                            net_cost_usd = (
                                slippage_cost_usd
                                + expected_fees_usd
                                + expected_market_impact_usd
                            )
                            maker_taker_prop = predict_maker_taker_proportion(
                                asks,
                                bids,
                                current_mid_price,
                                current_order_book_imbalance,
                                tick_volatility,
                            )

                            end_time = time.perf_counter()
                            internal_latency_ms = (end_time - start_time) * 1000
                            self._processing_latencies.append(internal_latency_ms)

                            output_data = {
                                "Timestamp": timestamp,
                                "Expected Slippage (%)": f"{expected_slippage_pct:.4f}",
                                "Expected Fees (USD)": f"{expected_fees_usd:.4f}",
                                "Expected Market Impact (USD)": f"{expected_market_impact_usd:.4f}",
                                "Net Cost (USD)": f"{net_cost_usd:.4f}",
                                "Maker Proportion": f"{maker_taker_prop['maker']:.2f}",
                                "Taker Proportion": f"{maker_taker_prop['taker']:.2f}",
                                "Internal Latency (ms)": f"{internal_latency_ms:.2f}",
                                "Current Volatility (raw tick)": f"{tick_volatility:.8f}",
                                "Order Book Imbalance": f"{current_order_book_imbalance:.4f}",
                            }
                            try:
                                self.output_queue.put_nowait(output_data)
                            except queue.Full:
                                logging.warning("Output queue full")

                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed as e:
                            logging.error(f"WebSocket closed: {e}")
                            break
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
        self._is_running = False

    def _run_client_in_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._websocket_client_async())
        finally:
            self._loop.close()

    def start(self):
        if self._is_running:
            logging.warning("Simulator already running")
            return
        logging.info("Starting simulator")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_client_in_thread, daemon=True)
        self._thread.start()

    def stop(self):
        logging.info("Stopping simulator")
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        else:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._is_running = False

    def update_config(self, new_config_values: Dict):
        """Update configuration with validation."""
        for key, value in new_config_values.items():
            if key in [
                "quantity_usd",
                "ac_total_time_to_execute_H_hours",
                "ac_num_trades_N",
                "volatility_annualized_config",
            ] and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"Invalid {key}: {value}. Must be positive.")
            if key == "fee_tier" and value not in [1, 2, 3, 4]:
                raise ValueError(f"Invalid fee_tier: {value}. Must be 1, 2, 3, or 4.")
        self.config.update(new_config_values)
        if "daily_volume_usd_estimate" in new_config_values:
            self.daily_volume_usd_estimate = new_config_values[
                "daily_volume_usd_estimate"
            ]
        logging.info(f"Config updated: {self.config}")

    def get_latency_stats(self) -> Dict:
        """Return latency statistics."""
        if not self._processing_latencies:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}
        latencies = list(self._processing_latencies)
        return {
            "avg": np.mean(latencies),
            "max": np.max(latencies),
            "min": np.min(latencies),
            "count": len(latencies),
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """Return latency statistics for processing."""
        return benchmark_latency(self._processing_latencies)

    def get_tick_frequency(self) -> float:
        """Calculate tick frequency."""
        if len(self._tick_timestamps) < 2:
            return 0.0
        intervals = np.diff(self._tick_timestamps)
        return 1 / np.mean(intervals) if intervals.size > 0 else 0.0


async def _websocket_client_async(self):
    """WebSocket client with reconnection logic."""
    retry_delay = 1
    while not self._stop_event.is_set():
        try:
            async with websockets.connect(WEBSOCKET_URL, ping_interval=20) as websocket:
                logging.info("Connected to WebSocket")
                self._is_running = True
                retry_delay = 1
                while not self._stop_event.is_set():
                    try:
                        start_time = time.perf_counter()
                        message_str = await asyncio.wait_for(
                            websocket.recv(), timeout=1.0
                        )
                        self._tick_timestamps.append(time.time())
                        data = json.loads(message_str)

                        if (
                            "asks" not in data
                            or "bids" not in data
                            or "timestamp" not in data
                        ):
                            continue

                        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
                        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                        timestamp = data.get("timestamp")

                        if asks and bids:
                            current_mid_price = (asks[0][0] + bids[0][0]) / 2
                            self.mid_price_data.append(current_mid_price)
                            total_bid_volume = sum(q for _, q in bids[:10])
                            total_ask_volume = sum(q for _, q in asks[:10])
                            current_order_book_imbalance = (
                                (total_bid_volume - total_ask_volume)
                                / (total_bid_volume + total_ask_volume)
                                if total_bid_volume + total_ask_volume > 0
                                else 0
                            )
                            self.order_book_imbalance_data.append(
                                current_order_book_imbalance
                            )
                        else:
                            current_mid_price = 0
                            current_order_book_imbalance = 0

                        min_volatility_window = 100
                        tick_volatility = 0.0
                        if len(self.mid_price_data) >= min_volatility_window:
                            prices = np.array(
                                list(self.mid_price_data)[-min_volatility_window:]
                            )
                            log_returns = np.log(prices[1:] / prices[:-1])
                            log_returns = log_returns[np.isfinite(log_returns)]
                            if len(log_returns) > 1:
                                tick_volatility = np.std(log_returns)

                        expected_slippage_pct = predict_expected_slippage_reg(
                            asks,
                            bids,
                            self.config["quantity_usd"],
                            current_mid_price,
                            current_order_book_imbalance,
                        )
                        expected_fees_usd = calculate_expected_fees(
                            self.config["quantity_usd"],
                            self.config["fee_tier"],
                            self.config["order_type"],
                        )
                        expected_market_impact_usd = (
                            calculate_market_impact_almgren_chriss(
                                self.config["quantity_usd"],
                                current_mid_price,
                                tick_volatility,
                                self.daily_volume_usd_estimate,
                                self.config,
                            )
                        )
                        slippage_cost_usd = (expected_slippage_pct / 100) * self.config[
                            "quantity_usd"
                        ]
                        net_cost_usd = (
                            slippage_cost_usd
                            + expected_fees_usd
                            + expected_market_impact_usd
                        )
                        maker_taker_prop = predict_maker_taker_proportion(
                            asks,
                            bids,
                            current_mid_price,
                            current_order_book_imbalance,
                            tick_volatility,
                        )

                        end_time = time.perf_counter()
                        internal_latency_ms = (end_time - start_time) * 1000
                        self._processing_latencies.append(internal_latency_ms)

                        output_data = {
                            "Timestamp": timestamp,
                            "Expected Slippage (%)": f"{expected_slippage_pct:.4f}",
                            "Expected Fees (USD)": f"{expected_fees_usd:.4f}",
                            "Expected Market Impact (USD)": f"{expected_market_impact_usd:.4f}",
                            "Net Cost (USD)": f"{net_cost_usd:.4f}",
                            "Maker Proportion": f"{maker_taker_prop['maker']:.2f}",
                            "Taker Proportion": f"{maker_taker_prop['taker']:.2f}",
                            "Internal Latency (ms)": f"{internal_latency_ms:.2f}",
                            "Current Volatility (raw tick)": f"{tick_volatility:.8f}",
                            "Order Book Imbalance": f"{current_order_book_imbalance:.4f}",
                        }
                        try:
                            self.output_queue.put_nowait(output_data)
                        except queue.Full:
                            logging.warning("Output queue full")

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logging.error(f"WebSocket closed: {e}")
                        break
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
    self._is_running = False
    logging.info(f"Tick frequency: {self.get_tick_frequency():.2f} ticks/sec")


if __name__ == "__main__":
    logging.info("Starting the trade simulator...")
    output_queue = queue.Queue(maxsize=1000)
    simulator = RealTradeSimulator(output_queue=output_queue)
    simulator.start()

    try:
        while True:
            if not output_queue.empty():
                output_data = output_queue.get()
            time.sleep(0.1)  # Prevent busy-waiting
    except KeyboardInterrupt:
        logging.info("Stopping the trade simulator...")
        simulator.stop()
