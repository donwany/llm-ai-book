# 1. Configure the schema: <stock.yaml>
# name: stock_price
# enabled: true
# required: false
# description: >-
#   The StockPricePlugin connects to the Yahoo Finance API using yfinance to fetch
#   real-time stock market data. It retrieves and returns the latest stock price
#   information based on the provided stock symbol, including the current market price.
#
# parameters:
#   - name: stock_symbol
#     type: str
#     required: true
#     description: The stock symbol for which the current market price is to be retrieved.
#
# returns:
#   - name: stock_data
#     type: dict
#     description: >-
#       A dictionary containing the stock symbol and its current market price. The format is:
#       {
#         "symbol": <stock_symbol>,
#         "current_price": <current_market_price>
#       }

# 2. Implement the python code: stock.py
import yfinance as yf
from typing import Dict, Union
from taskweaver.plugin import Plugin, register_plugin

# Define the detailed prompt for the plugin
stock_price_prompt = """
This plugin is designed to fetch real-time stock market data. It performs the following tasks:
- Retrieves the latest stock price information based on the provided stock symbol.
- Extracts key financial details such as the current market price.
- Finally, it returns this information in a structured dictionary format, as shown below:
  {
    "symbol": <stock_symbol>,
    "current_price": <current_market_price>
  }
"""


@register_plugin
class StockPricePlugin(Plugin):
    def __call__(self, stock_symbols: Union[str, list]) -> Union[Dict[str, Union[str, float]], Dict[str, str]]:
        # Ensure stock_symbols is a list
        if isinstance(stock_symbols, str):
            stock_symbols = [stock_symbols]
        results = {}
        for symbol in stock_symbols:
            try:
                # Fetch stock data using yfinance
                stock = yf.Ticker(symbol)
                # Check if stock exists and has info
                if not stock.info:
                    results[symbol] = {"error": f"No data found for symbol: {symbol}"}
                    continue
                # Extract the current stock price
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                # Check if current price is available
                if current_price is None:
                    results[symbol] = {"error": f"Current price not available for symbol: {symbol}"}
                    continue
                # Store the result in the dictionary
                results[symbol] = {
                    "symbol": symbol,
                    "current_price": current_price
                }
            except Exception as e:
                results[symbol] = {"error": str(e)}
        return results

# 3. Call the plugin
# python -m taskweaver -p ./project/
