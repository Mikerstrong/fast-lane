"""
Debug script to test web scraping functionality for yfinance alternative
Run this to verify MarketWatch and Investing.com scrapers are working
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web_scrapers import StockDataAggregator, YahooFinanceScreenerScraper, CNBCScraper, FallbackDataProvider, MarketWatchScraper, InvestingComScraper
    print("✅ Successfully imported web scraping modules")
except ImportError as e:
    print(f"❌ Failed to import web scraping modules: {e}")
    print("Install required packages: pip install requests beautifulsoup4")
    sys.exit(1)

def test_individual_scrapers():
    """Test each scraper individually"""
    print("\n" + "="*60)
    print("🔍 TESTING INDIVIDUAL SCRAPERS")
    print("="*60)
    
    # Test MarketWatch
    print("\n📊 Testing MarketWatch Scraper...")
    mw = MarketWatchScraper()
    
    try:
        print("  → Fetching gainers...")
        gainers = mw.get_gainers(3)
        print(f"  → Found {len(gainers)} gainers")
        for i, stock in enumerate(gainers, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ MarketWatch gainers error: {e}")
    
    try:
        print("  → Fetching most active...")
        active = mw.get_most_active(3)
        print(f"  → Found {len(active)} most active stocks")
        for i, stock in enumerate(active, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ MarketWatch most active error: {e}")
    
    # Test Investing.com
    print("\n💼 Testing Investing.com Scraper...")
    inv = InvestingComScraper()
    
    try:
        print("  → Fetching gainers...")
        gainers = inv.get_gainers(3)
        print(f"  → Found {len(gainers)} gainers")
        for i, stock in enumerate(gainers, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ Investing.com gainers error: {e}")
    
    try:
        print("  → Fetching most active...")
        active = inv.get_most_active(3)
        print(f"  → Found {len(active)} most active stocks")
        for i, stock in enumerate(active, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ Investing.com most active error: {e}")
    
    # Test Yahoo Finance Screener
    print("\n📊 Testing Yahoo Finance Screener...")
    yahoo = YahooFinanceScreenerScraper()
    
    try:
        print("  → Fetching gainers...")
        gainers = yahoo.get_gainers(3)
        print(f"  → Found {len(gainers)} gainers")
        for i, stock in enumerate(gainers, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ Yahoo gainers error: {e}")
    
    # Test CNBC
    print("\n💼 Testing CNBC Scraper...")
    cnbc = CNBCScraper()
    
    try:
        print("  → Fetching gainers...")
        gainers = cnbc.get_gainers(3)
        print(f"  → Found {len(gainers)} gainers")
        for i, stock in enumerate(gainers, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ CNBC gainers error: {e}")
    
    # Test Fallback Data
    print("\n🔧 Testing Fallback Data Provider...")
    fallback = FallbackDataProvider()
    
    try:
        print("  → Getting fallback gainers...")
        gainers = fallback.get_gainers(3)
        print(f"  → Found {len(gainers)} gainers")
        for i, stock in enumerate(gainers, 1):
            print(f"    {i}. {stock['symbol']}: {stock['change_pct']:+.2f}% (${stock['price']:.2f}) - {stock['source']}")
    except Exception as e:
        print(f"  ❌ Fallback error: {e}")

def test_aggregator():
    """Test the combined aggregator"""
    print("\n" + "="*60)
    print("🔄 TESTING COMBINED AGGREGATOR")
    print("="*60)
    
    aggregator = StockDataAggregator()
    
    # Test different source preferences including new ones
    for source in ['both', 'marketwatch', 'investing', 'yahoo', 'cnbc', 'fallback']:
        print(f"\n🎯 Testing with source: {source}")
        try:
            dropdown_options, symbols, mapping = aggregator.get_formatted_dropdown_data(source)
            print(f"  → Total dropdown options: {len(dropdown_options)}")
            print(f"  → Total symbols: {len(symbols)}")
            print(f"  → Sample options:")
            for i, option in enumerate(dropdown_options[:5]):
                print(f"    {i+1}. {option}")
            
            if mapping:
                print(f"  → Symbol mapping working: {len(mapping)} entries")
            else:
                print("  ⚠️ No symbol mapping created")
                
        except Exception as e:
            print(f"  ❌ Error with {source}: {e}")
            
        print("  " + "-"*40)  # Separator between sources

def test_dropdown_integration():
    """Test how the data would integrate with the Streamlit dropdown"""
    print("\n" + "="*60)
    print("📋 TESTING DROPDOWN INTEGRATION")
    print("="*60)
    
    aggregator = StockDataAggregator()
    
    try:
        dropdown_options, symbols, mapping = aggregator.get_formatted_dropdown_data('both')
        
        print(f"📊 Generated {len(dropdown_options)} dropdown options")
        print(f"🎯 Generated {len(symbols)} symbols")
        print(f"🔗 Created {len(mapping)} display-to-symbol mappings")
        
        print("\n📝 Sample dropdown content:")
        for i, option in enumerate(dropdown_options[:10]):
            print(f"  {i+1:2d}. {option}")
            
        # Test symbol lookup
        print(f"\n🔍 Testing symbol lookup:")
        test_options = [opt for opt in dropdown_options if not opt.startswith('---')][:3]
        for option in test_options:
            symbol = mapping.get(option, "NOT FOUND")
            print(f"  '{option[:30]}...' → {symbol}")
            
    except Exception as e:
        print(f"❌ Dropdown integration error: {e}")

def main():
    """Run all tests"""
    print("🚀 YFINANCE ALTERNATIVE - WEB SCRAPING DEBUG")
    print("Testing MarketWatch and Investing.com scrapers")
    print("=" * 60)
    
    # Check if requests and beautifulsoup4 are available
    try:
        import requests
        import bs4
        print("✅ Required packages available (requests, beautifulsoup4)")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Install with: pip install requests beautifulsoup4")
        return
    
    # Run tests
    test_individual_scrapers()
    test_aggregator() 
    test_dropdown_integration()
    
    print("\n" + "="*60)
    print("🎉 DEBUG COMPLETE")
    print("If you see stock data above, web scraping is working!")
    print("You can now use the 'Web Scraping' option in your Streamlit app.")
    print("="*60)

if __name__ == "__main__":
    main()