import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(ticker="AAPL", period="5y"):
    """
    Fetches historical market data, generates a synthetic trade history for a
    risk-averse trader, and creates a visualization.

    The trader's strategy is based on a 50/200-day moving average crossover.
    - Go LONG when 50-day SMA > 200-day SMA.
    - Go SHORT when 50-day SMA < 200-day SMA.

    Args:
        ticker (str): The stock ticker to use (e.g., "AAPL").
        period (str): The period for which to fetch the data (e.g., "5y" for 5 years).

    Returns:
        None. Saves data to CSV files and the plot to a PNG file.
    """
    print(f"Fetching market data for {ticker}...")
    stock_data = yf.download(ticker, period=period, auto_adjust=True)

    if stock_data.empty:
        print(f"No data found for ticker {ticker}. Exiting.")
        return

    # --- Market Data Preparation ---
    market_data_filename = f"{ticker.lower()}_market_data.csv"
    stock_data.to_csv(market_data_filename)
    print(f"Market data saved to '{market_data_filename}'")

    # --- Synthetic Trade History Generation ---
    print("Generating synthetic trade history for a risk-averse trader...")

    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data.dropna(inplace=True)

    stock_data['Position'] = np.where(stock_data['SMA50'] > stock_data['SMA200'], 1, -1)
    stock_data['Trade'] = stock_data['Position'].diff()

    trade_history = []
    buy_signals = []
    sell_signals = []

    for index, row in stock_data.iterrows():
        trade_action = "HOLD"
        if row['Trade'] == 2:  # From -1 (short) to 1 (long) -> BUY
            trade_action = "BUY"
            buy_signals.append(index)
        elif row['Trade'] == -2:  # From 1 (long) to -1 (short) -> SELL
            trade_action = "SELL"
            sell_signals.append(index)

        if trade_action != "HOLD":
            trade_history.append({
                "Date": index.strftime('%Y-%m-%d'),
                "Price": row['Close'],
                "Action": trade_action
            })

    trade_history_df = pd.DataFrame(trade_history)
    trade_history_filename = f"{ticker.lower()}_trade_history.csv"
    trade_history_df.to_csv(trade_history_filename, index=False)
    print(f"Synthetic trade history saved to '{trade_history_filename}'")

    # --- Visualization ---
    print("Generating trade history visualization...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot prices and SMAs
    ax.plot(stock_data['Close'], label='AAPL Close Price', color='skyblue', linewidth=2)
    ax.plot(stock_data['SMA50'], label='50-Day SMA', color='orange', linestyle='--')
    ax.plot(stock_data['SMA200'], label='200-Day SMA', color='purple', linestyle='--')

    # Plot buy and sell signals
    ax.plot(buy_signals, stock_data.loc[buy_signals]['Close'], '^', markersize=10,
            color='g', label='BUY Signal', alpha=0.9)
    ax.plot(sell_signals, stock_data.loc[sell_signals]['Close'], 'v', markersize=10,
            color='r', label='SELL Signal', alpha=0.9)

    ax.set_title(f'{ticker} Trade History (Risk-Averse Strategy)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    plot_filename = f"{ticker.lower()}_trade_history.png"
    plt.savefig(plot_filename)
    print(f"Trade history plot saved to '{plot_filename}'")

    print("\nData generation complete.")


if __name__ == "__main__":
    generate_synthetic_data()
