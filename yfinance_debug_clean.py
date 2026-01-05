"""
Clean debug script to test only working web scrapers
Removed non-working MarketWatch and Investing.com scrapers
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web_scrapers import CNBCScraper, YahooFinanceScreenerScraper, FallbackDataProvider, StockDataAggregator
    print("✅ Successfully imported web scraping modules")
except ImportError as e:
    print(f"❌ Failed to import web scraping modules: {e}")
    print("Install required packages: pip install requests beautifulsoup4")
    sys.exit(1)

import time

def test_individual_scrapers():
    """Test each working scraper individually"""
    
    print("\n" + "="*50)
    print("TESTING WORKING SCRAPERS ONLY")
    print("="*50)
    
    # Only test the scrapers that might work
    scrapers = [
        ("Yahoo Finance Markets", YahooFinanceScreenerScraper()),
        ("CNBC", CNBCScraper()),
    ]
    
    for name, scraper in scrapers:
        print(f"\n--- Testing {name} ---")
        
        # Test gainers
        print(f"Testing {name} gainers...")
        gainers = scraper.get_gainers(limit=5)
        if gainers:
            print(f"✅ Found {len(gainers)} gainers:")
            for stock in gainers[:3]:
                print(f"   {stock['symbol']}: {stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
        else:
            print(f"❌ No gainers found")
        
        time.sleep(1)  # Be nice to servers
        
        # Test most active
        print(f"Testing {name} most active...")
        actives = scraper.get_most_active(limit=5)
        if actives:
            print(f"✅ Found {len(actives)} most active:")
            for stock in actives[:3]:
                print(f"   {stock['symbol']}: {stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
        else:
            print(f"❌ No most active found")
        
        time.sleep(1)

def test_fallback():
    """Test the fallback data provider"""
    print("\n" + "="*50)
    print("TESTING FALLBACK DATA")
    print("="*50)
    
    fallback = FallbackDataProvider()
    
    gainers = fallback.get_gainers()
    print(f"\nFallback gainers: {len(gainers)} stocks")
    for stock in gainers[:5]:
        print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
    
    actives = fallback.get_most_active()
    print(f"\nFallback most active: {len(actives)} stocks")
    for stock in actives[:5]:
        print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_pct']:+.2f}%)")

def test_aggregator():
    """Test the main aggregator that combines all working sources"""
    print("\n" + "="*50)
    print("TESTING STOCK DATA AGGREGATOR")
    print("="*50)
    
    aggregator = StockDataAggregator()
    
    print("\nGetting gainers from working sources...")
    gainers = aggregator.get_gainers()
    print(f"Total gainers found: {len(gainers)}")
    
    # Group by source
    sources = {}
    for stock in gainers:
        source = stock['source']
        if source not in sources:
            sources[source] = []
        sources[source].append(stock)
    
    for source, stocks in sources.items():
        print(f"\n{source}: {len(stocks)} stocks")
        for stock in stocks[:3]:
            print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
    
    print("\nGetting most active from working sources...")
    actives = aggregator.get_most_active()
    print(f"Total most active found: {len(actives)}")
    
    # Group by source  
    sources = {}
    for stock in actives:
        source = stock['source']
        if source not in sources:
            sources[source] = []
        sources[source].append(stock)
    
    for source, stocks in sources.items():
        print(f"\n{source}: {len(stocks)} stocks")
        for stock in stocks[:3]:
            print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_pct']:+.2f}%)")

if __name__ == "__main__":
    print("Clean Web Scraper Debug Tool")
    print("Testing only working scrapers (Yahoo Finance Markets + CNBC)...")
    
    # Test individual scrapers
    test_individual_scrapers()
    
    # Test fallback
    test_fallback()
    
    # Test aggregator
    test_aggregator()
    
    print("\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)