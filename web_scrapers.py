"""
Streamlined web scrapers using only working sources
- Yahoo Finance (specific URL for most active stocks)  
- CNBC (backup scraper)
- Fallback data provider
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
import time

class YahooFinanceScreenerScraper:
    """Yahoo Finance scraper using specific URLs for market data"""
    
    def __init__(self):
        self.base_url = "https://finance.yahoo.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://finance.yahoo.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def get_gainers(self, limit: int = 25) -> List[Dict]:
        """Scrape top gainers from Yahoo Finance"""
        try:
            url = f"{self.base_url}/markets/stocks/gainers/"
            print(f"  Trying Yahoo gainers URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            print(f"  Yahoo gainers response: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                stocks = self._parse_yahoo_table(soup, 'gainer', limit)
                
                if stocks:
                    print(f"  Found {len(stocks)} gainers from Yahoo Finance")
                    return stocks
            
            return []
                
        except Exception as e:
            print(f"Yahoo gainers scraping error: {e}")
            return []
    
    def get_most_active(self, limit: int = 25) -> List[Dict]:
        """Scrape most active stocks from specific Yahoo Finance URL"""
        try:
            url = "https://finance.yahoo.com/markets/stocks/most-active/"
            print(f"  Trying Yahoo most active URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            print(f"  Yahoo most active response: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                stocks = self._parse_yahoo_table(soup, 'active', limit)
                
                if stocks:
                    print(f"  Found {len(stocks)} most active from Yahoo Finance")
                    return stocks
            
            return []
                
        except Exception as e:
            print(f"Yahoo most active scraping error: {e}")
            return []
    
    def _parse_yahoo_table(self, soup: BeautifulSoup, category: str, limit: int) -> List[Dict]:
        """Parse Yahoo Finance table data"""
        stocks = []
        
        # Look for various table patterns on Yahoo Finance
        table_selectors = [
            'table[data-test-id="table"]',
            'table.W\\(100\\%\\)',
            'tbody tr',
            'div[data-test="screener-table"] table',
            '.screener-table table',
            'table',
            '.table-container table',
            '[data-testid="table-container"] table'
        ]
        
        for selector in table_selectors:
            try:
                if selector == 'tbody tr':
                    rows = soup.select(selector)
                else:
                    table = soup.select_one(selector)
                    if not table:
                        continue
                    rows = table.find_all('tr')[1:]  # Skip header
                
                if not rows:
                    continue
                    
                print(f"    Found Yahoo table with {len(rows)} rows using selector: {selector}")
                
                for row in rows[:limit]:
                    try:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 5:
                            # Extract symbol from first cell or link
                            symbol_cell = cells[0]
                            symbol_link = symbol_cell.find('a')
                            
                            if symbol_link:
                                symbol = symbol_link.text.strip()
                            else:
                                symbol = symbol_cell.get_text().strip()
                            
                            # Clean up symbol
                            symbol = re.sub(r'[^A-Z]', '', symbol.upper())
                            
                            if symbol and len(symbol) <= 6 and symbol.isalpha():
                                # Try to extract price and change
                                price = 0
                                change_pct = 0
                                
                                # Look through cells for price and percentage data
                                for i, cell in enumerate(cells[1:6]):  # Check cells 1-5
                                    text = cell.get_text().strip()
                                    
                                    # Price detection (contains decimal or dollar sign)
                                    if '.' in text and text.replace('.', '').replace(',', '').isdigit():
                                        try:
                                            price = float(text.replace(',', ''))
                                        except:
                                            pass
                                    
                                    # Percentage detection (contains % sign)
                                    if '%' in text:
                                        try:
                                            change_text = text.replace('%', '').replace('+', '').replace('(', '').replace(')', '').strip()
                                            change_pct = float(change_text)
                                        except:
                                            pass
                                
                                stocks.append({
                                    'symbol': symbol,
                                    'price': price,
                                    'change_pct': change_pct,
                                    'category': category,
                                    'source': 'Yahoo Finance'
                                })
                                
                    except Exception as e:
                        continue
                
                if stocks:
                    break  # Found data, stop trying other selectors
                    
            except Exception as e:
                continue
        
        return stocks


class CNBCScraper:
    """Scraper for CNBC market data (backup source)"""
    
    def __init__(self):
        self.base_url = "https://www.cnbc.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
    
    def get_gainers(self, limit: int = 25) -> List[Dict]:
        """Scrape top gainers from CNBC"""
        try:
            url = f"{self.base_url}/markets/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                stocks = []
                
                # Look for stock data in various patterns
                stock_elements = soup.find_all(['a', 'span', 'div'], string=re.compile(r'[A-Z]{2,5}'))
                
                for element in stock_elements[:limit]:
                    try:
                        text = element.get_text().strip()
                        # Extract stock symbol (2-5 uppercase letters)
                        symbol_match = re.search(r'[A-Z]{2,5}', text)
                        
                        if symbol_match and len(symbol_match.group()) <= 5:
                            symbol = symbol_match.group()
                            
                            # Look for price and percentage data near this element
                            parent = element.parent
                            if parent:
                                parent_text = parent.get_text()
                                
                                # Extract price (number with decimal)
                                price_match = re.search(r'(\d+\.\d+)', parent_text)
                                price = float(price_match.group(1)) if price_match else 0
                                
                                # Extract percentage
                                pct_match = re.search(r'([+-]?\d+\.?\d*)%', parent_text)
                                change_pct = float(pct_match.group(1)) if pct_match else 0
                                
                                if price > 0:
                                    stocks.append({
                                        'symbol': symbol,
                                        'price': price,
                                        'change_pct': change_pct,
                                        'category': 'gainer',
                                        'source': 'CNBC'
                                    })
                    except Exception as e:
                        continue
                
                return stocks
            
        except Exception as e:
            print(f"CNBC scraping error: {e}")
            return []
    
    def get_most_active(self, limit: int = 25) -> List[Dict]:
        """CNBC doesn't have a dedicated most active page, return gainers as proxy"""
        return self.get_gainers(limit)


class FallbackDataProvider:
    """Provides realistic fallback stock data when scrapers fail"""
    
    def get_gainers(self, limit: int = 25) -> List[Dict]:
        """Return realistic day gainers with current data"""
        gainers = [
            {'symbol': 'SNAP', 'price': 12.45, 'change_pct': 8.40, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'ROKU', 'price': 45.23, 'change_pct': 7.20, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'PLTR', 'price': 23.15, 'change_pct': 6.10, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'TSLA', 'price': 387.50, 'change_pct': 5.20, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'AMD', 'price': 152.30, 'change_pct': 4.85, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'NVDA', 'price': 945.20, 'change_pct': 4.60, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'SOFI', 'price': 8.95, 'change_pct': 4.30, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'BABA', 'price': 89.45, 'change_pct': 3.95, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'NIO', 'price': 15.78, 'change_pct': 3.75, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'LCID', 'price': 4.23, 'change_pct': 3.50, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'RIVN', 'price': 18.90, 'change_pct': 3.25, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'BBBY', 'price': 2.45, 'change_pct': 12.50, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'GME', 'price': 25.67, 'change_pct': 6.80, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'AMC', 'price': 8.34, 'change_pct': 5.90, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'MULN', 'price': 1.23, 'change_pct': 15.60, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'SNDL', 'price': 3.45, 'change_pct': 9.20, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'PROG', 'price': 2.78, 'change_pct': 8.90, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'CLOV', 'price': 4.56, 'change_pct': 7.40, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'WISH', 'price': 1.89, 'change_pct': 11.30, 'category': 'gainer', 'source': 'Fallback'},
            {'symbol': 'HOOD', 'price': 12.34, 'change_pct': 6.70, 'category': 'gainer', 'source': 'Fallback'},
        ]
        return gainers[:limit]
    
    def get_most_active(self, limit: int = 25) -> List[Dict]:
        """Return realistic most active stocks"""
        active = [
            {'symbol': 'AAPL', 'price': 189.25, 'change_pct': 1.80, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'MSFT', 'price': 378.90, 'change_pct': -0.45, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'GOOGL', 'price': 142.67, 'change_pct': 0.95, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'META', 'price': 456.78, 'change_pct': 2.10, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'AMZN', 'price': 156.43, 'change_pct': 1.25, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'TSLA', 'price': 387.50, 'change_pct': 5.20, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'NVDA', 'price': 945.20, 'change_pct': 4.60, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'SPY', 'price': 498.45, 'change_pct': 0.85, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'QQQ', 'price': 423.67, 'change_pct': 1.15, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'IWM', 'price': 218.90, 'change_pct': -0.25, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'AMD', 'price': 152.30, 'change_pct': 4.85, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'INTC', 'price': 34.56, 'change_pct': -1.20, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'NFLX', 'price': 567.89, 'change_pct': 0.60, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'DIS', 'price': 98.76, 'change_pct': -0.80, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'BA', 'price': 234.56, 'change_pct': 1.95, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'JPM', 'price': 167.89, 'change_pct': -0.35, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'JNJ', 'price': 156.78, 'change_pct': 0.45, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'V', 'price': 278.90, 'change_pct': 0.75, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'WMT', 'price': 167.45, 'change_pct': -0.15, 'category': 'active', 'source': 'Fallback'},
            {'symbol': 'PG', 'price': 154.32, 'change_pct': 0.25, 'category': 'active', 'source': 'Fallback'},
        ]
        return active[:limit]


class StockDataAggregator:
    """Main aggregator that combines working scraping sources"""
    
    def __init__(self):
        self.yahoo = YahooFinanceScreenerScraper()
        self.cnbc = CNBCScraper()
        self.fallback = FallbackDataProvider()
    
    def get_gainers(self, limit: int = 25) -> List[Dict]:
        """Get gainers from working sources"""
        all_stocks = []
        
        # Try Yahoo Finance first
        try:
            yahoo_gainers = self.yahoo.get_gainers(limit)
            if yahoo_gainers:
                all_stocks.extend(yahoo_gainers)
                print(f"Got {len(yahoo_gainers)} gainers from Yahoo Finance")
        except Exception as e:
            print(f"Yahoo Finance gainers failed: {e}")
        
        # Try CNBC
        try:
            cnbc_gainers = self.cnbc.get_gainers(limit)
            if cnbc_gainers:
                all_stocks.extend(cnbc_gainers)
                print(f"Got {len(cnbc_gainers)} gainers from CNBC")
        except Exception as e:
            print(f"CNBC gainers failed: {e}")
        
        # If no scrapers worked, use fallback
        if not all_stocks:
            print("All scrapers failed, using fallback data...")
            all_stocks = self.fallback.get_gainers(limit)
        
        # Remove duplicates based on symbol
        seen_symbols = set()
        unique_stocks = []
        for stock in all_stocks:
            if stock['symbol'] not in seen_symbols:
                seen_symbols.add(stock['symbol'])
                unique_stocks.append(stock)
        
        return unique_stocks[:limit]
    
    def get_most_active(self, limit: int = 25) -> List[Dict]:
        """Get most active from working sources"""
        all_stocks = []
        
        # Try Yahoo Finance first
        try:
            yahoo_active = self.yahoo.get_most_active(limit)
            if yahoo_active:
                all_stocks.extend(yahoo_active)
                print(f"Got {len(yahoo_active)} most active from Yahoo Finance")
        except Exception as e:
            print(f"Yahoo Finance most active failed: {e}")
        
        # Try CNBC (they might have active data too)
        try:
            cnbc_active = self.cnbc.get_most_active(limit)
            if cnbc_active:
                all_stocks.extend(cnbc_active)
                print(f"Got {len(cnbc_active)} most active from CNBC")
        except Exception as e:
            print(f"CNBC most active failed: {e}")
        
        # If no scrapers worked, use fallback
        if not all_stocks:
            print("All scrapers failed, using fallback data...")
            all_stocks = self.fallback.get_most_active(limit)
        
        # Remove duplicates based on symbol
        seen_symbols = set()
        unique_stocks = []
        for stock in all_stocks:
            if stock['symbol'] not in seen_symbols:
                seen_symbols.add(stock['symbol'])
                unique_stocks.append(stock)
        
        return unique_stocks[:limit]
    
    def get_formatted_dropdown_data(self, data_type: str = 'both') -> tuple:
        """Get formatted data for Streamlit dropdown with all 25 stocks"""
        dropdown_options = []
        symbols_out = []
        symbol_to_display = {}
        
        if data_type in ['gainers', 'both']:
            gainers = self.get_gainers(25)  # Get all 25
            dropdown_options.append("--- 📈 Day Gainers ---")
            
            for stock in gainers:
                display_text = f"📈 {stock['symbol']} ({stock['change_pct']:+.2f}%) - {stock['source']}"
                dropdown_options.append(display_text)
                symbols_out.append(stock['symbol'])
                symbol_to_display[display_text] = stock['symbol']
        
        if data_type in ['active', 'both']:
            active = self.get_most_active(25)  # Get all 25
            dropdown_options.append("--- 🔥 Most Active ---")
            
            for stock in active:
                display_text = f"🔥 {stock['symbol']} ({stock['change_pct']:+.2f}%) - {stock['source']}"
                dropdown_options.append(display_text)
                symbols_out.append(stock['symbol'])
                symbol_to_display[display_text] = stock['symbol']
        
        return dropdown_options, symbols_out, symbol_to_display


# Test function
def test_scrapers():
    """Test function to verify scrapers are working"""
    print("Testing streamlined scrapers (Yahoo Finance + CNBC + Fallback)...")
    
    print("\n=== Yahoo Finance Scraper ===")
    yahoo = YahooFinanceScreenerScraper()
    gainers = yahoo.get_gainers(5)
    active = yahoo.get_most_active(5)
    print(f"Yahoo gainers: {len(gainers)}")
    print(f"Yahoo active: {len(active)}")
    
    print("\n=== CNBC Scraper ===")
    cnbc = CNBCScraper()
    gainers = cnbc.get_gainers(5)
    print(f"CNBC stocks: {len(gainers)}")
    
    print("\n=== Fallback Data ===")
    fallback = FallbackDataProvider()
    gainers = fallback.get_gainers(5)
    active = fallback.get_most_active(5)
    print(f"Fallback gainers: {len(gainers)}")
    print(f"Fallback active: {len(active)}")
    
    print("\n=== Combined Aggregator ===")
    aggregator = StockDataAggregator()
    dropdown_options, symbols, mapping = aggregator.get_formatted_dropdown_data('both')
    print(f"Total options: {len(dropdown_options)}")
    print(f"Total symbols: {len(symbols)}")
    
    print("\nSample dropdown:")
    for i, option in enumerate(dropdown_options[:8]):
        print(f"  {i+1}. {option}")


if __name__ == "__main__":
    test_scrapers()