import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json
import os

# Import analysis functions
from analysis import (
    calculate_bollinger_bands, 
    calculate_dpo_9, 
    calculate_dpo_20,
    analyze_fast_lane_stock,
    bulk_analyze_stocks_with_live_updates,
    bulk_analyze_stocks,
    load_cache,
    save_cache,
    get_today_key,
    is_cache_valid
)

# Configure Streamlit page
st.set_page_config(page_title="Stock Analysis with Bollinger Bands & DPO", layout="wide")

# Data directory (useful for Docker volume persistence)
DATA_DIR = os.getenv("FASTLANE_DATA_DIR", ".")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    DATA_DIR = "."

# Cache / persistence file paths
CACHE_FILE = os.path.join(DATA_DIR, "stock_analysis_cache.json")

# My stocks (simple holdings tracker)
MYSTOCKS_FILE = os.path.join(DATA_DIR, "mystocks.json")


def load_mystocks():
    try:
        if os.path.exists(MYSTOCKS_FILE):
            with open(MYSTOCKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and isinstance(data.get("holdings"), list):
                    return data["holdings"]
    except Exception:
        pass
    return []


def save_mystocks(holdings):
    try:
        payload = {
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "holdings": holdings,
        }
        with open(MYSTOCKS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        st.sidebar.error(f"Failed saving {MYSTOCKS_FILE}: {e}")


def _normalize_symbol(sym: str) -> str:
    sym = (sym or "").strip().upper()
    if sym.startswith("$"):
        sym = sym[1:]
    return sym


def fetch_live_quote_price(symbol: str):
    """Best-effort quote price fetch.

    Uses yfinance fast_info when available (usually cheaper), otherwise falls
    back to a 1m candle for today.
    Returns: (price|None, timestamp|None)
    """
    symbol = _normalize_symbol(symbol)
    if not symbol:
        return None, None

    # Prefer fast_info
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if isinstance(fi, dict):
            for key in ("last_price", "lastPrice", "last", "regularMarketPrice"):
                val = fi.get(key)
                if val is not None:
                    try:
                        return float(val), datetime.datetime.now()
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback to intraday candle
    try:
        t = yf.Ticker(symbol)
        intraday = t.history(period="1d", interval="1m")
        if intraday is None or intraday.empty:
            return None, None
        last_row = intraday.iloc[-1]
        return float(last_row["Close"]), intraday.index[-1]
    except Exception:
        return None, None


def get_live_prices(symbols, ttl_seconds: int = 60):
    """Get live prices for a set of symbols with session-state throttling."""
    now = datetime.datetime.now()
    cache = st.session_state.get("portfolio_live_prices", {})
    out = {}

    for sym in symbols:
        s = _normalize_symbol(sym)
        if not s:
            continue

        cached = cache.get(s)
        if cached:
            try:
                ts = datetime.datetime.fromisoformat(cached.get("ts"))
                if (now - ts).total_seconds() <= ttl_seconds:
                    out[s] = cached
                    continue
            except Exception:
                pass

        price, ts = fetch_live_quote_price(s)
        entry = {
            "price": price,
            "ts": (ts.isoformat(timespec="seconds") if hasattr(ts, "isoformat") else str(ts)) if ts else None,
        }
        cache[s] = entry
        out[s] = entry

    st.session_state["portfolio_live_prices"] = cache
    return out

def load_cache():
    """Load analysis cache from JSON file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.warning(f"Cache loading error: {e}")
    return {}

def save_cache(cache_data):
    """Save analysis cache to JSON file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        st.sidebar.warning(f"Cache saving error: {e}")

def get_today_key():
    """Get today's date as cache key"""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _get_today_fast_lane_symbols_from_cache(cache: dict) -> set:
    """Best-effort set of symbols that are currently passing Fast Lane today."""
    passing = set()
    today_key = get_today_key()

    try:
        bulk = (cache or {}).get("bulk_analysis", {}).get(today_key) or {}
        results = bulk.get("results") or []
        for r in results:
            if not isinstance(r, dict):
                continue
            sym = _normalize_symbol(r.get("symbol"))
            if not sym:
                continue
            if r.get("error", False):
                continue
            if r.get("is_fast_lane", False):
                passing.add(sym)
    except Exception:
        pass

    # Also consider per-symbol cache entries (covers manual search caching)
    try:
        for k, v in (cache or {}).items():
            if k in ("bulk_analysis", "_meta"):
                continue
            sym = _normalize_symbol(k)
            if not sym:
                continue
            day_entry = (v or {}).get(today_key) if isinstance(v, dict) else None
            data = (day_entry or {}).get("data") if isinstance(day_entry, dict) else None
            if isinstance(data, dict) and data.get("is_fast_lane", False) and not data.get("error", False):
                passing.add(sym)
    except Exception:
        pass

    return passing


def _decorate_dropdown_with_fast_lane(display_options: list, symbol_options: list, passing_syms: set) -> list:
    """Return display strings with 🚀 for passing, 🍋 for not passing.

    Always strips any previous 🚀/🍋 prefix first so the emoji updates cleanly
    whenever cached results change.
    """
    out = []
    for disp, sym in zip(display_options or [], symbol_options or []):
        d = str(disp) if disp is not None else ""
        # Remove any existing status prefix (rocket/lemon), possibly repeated.
        while d.startswith("🚀 ") or d.startswith("🍋 "):
            d = d[2:]
        s = _normalize_symbol(sym)
        if s and s in (passing_syms or set()):
            d = f"🚀 {d}"
        else:
            d = f"🍋 {d}"
        out.append(d)
    return out

def is_cache_valid(cache_entry, hours_valid=24):
    """Check if cache entry is still valid"""
    try:
        cached_time = datetime.datetime.fromisoformat(cache_entry['timestamp'])
        time_diff = datetime.datetime.now() - cached_time
        return time_diff.total_seconds() < (hours_valid * 3600)
    except:
        return False

def get_most_active_stocks():
    """Fetch top movers and most active stocks"""
    try:
        # Always include portfolio symbols in the dropdown.
        portfolio_symbols = []
        try:
            holdings = load_mystocks()
            for h in holdings:
                if isinstance(h, dict):
                    s = _normalize_symbol(h.get("stock"))
                    if s:
                        portfolio_symbols.append(s)
            # Preserve order and dedupe
            seen = set()
            portfolio_symbols = [s for s in portfolio_symbols if not (s in seen or seen.add(s))]
        except Exception:
            portfolio_symbols = []

        # Get top gainers, losers, and most active from Yahoo Finance
        # Using popular tickers as a fallback since Yahoo Finance screening can be inconsistent
        popular_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
            'SPY', 'QQQ', 'AMD', 'CRM', 'ORCL', 'INTC', 'IBM', 'BABA',
            'UBER', 'LYFT', 'SNAP', 'ROKU', 'ZM', 'PLTR', 'COIN',
            'GME', 'AMC', 'BB', 'NOK', 'SNDL', 'DOGE-USD', 'BTC-USD'
        ]
        
        # Get current data for these stocks to find actual movers
        active_stocks = []
        
        for ticker in popular_tickers[:20]:  # Limit to avoid API rate limits
            try:
                stock = yf.Ticker(ticker)
                info = stock.history(period="2d")  # Get last 2 days
                if len(info) >= 2:
                    current_price = info['Close'].iloc[-1]
                    prev_price = info['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    active_stocks.append({
                        'symbol': ticker,
                        'name': ticker,  # We can enhance this with company names later
                        'change_pct': change_pct,
                        'current_price': current_price
                    })
            except:
                continue
        
        # Sort by absolute percentage change to get most active
        active_stocks = sorted(active_stocks, key=lambda x: abs(x['change_pct']), reverse=True)
        
        # Format for dropdown
        dropdown_options = []

        # Portfolio symbols first (no extra API calls)
        for sym in portfolio_symbols:
            dropdown_options.append(f"⭐ {sym} (Portfolio)")
        for stock in active_stocks:
            change_symbol = "📈" if stock['change_pct'] > 0 else "📉"
            dropdown_options.append(
                f"{change_symbol} {stock['symbol']} ({stock['change_pct']:+.2f}%)"
            )

        # Build symbol list aligned with dropdown options
        symbols_out = []
        symbols_out.extend(portfolio_symbols)
        symbols_out.extend([stock['symbol'] for stock in active_stocks])
        # Deduplicate while preserving order
        seen = set()
        symbols_out = [s for s in symbols_out if not (s in seen or seen.add(s))]

        # Keep the dropdown compact
        limit = 20
        return dropdown_options[:limit], symbols_out[:limit]
    
    except Exception as e:
        # Fallback to popular stocks if API fails
        fallback_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
                          'SPY', 'QQQ', 'AMD', 'CRM', 'ORCL', 'INTC', 'IBM', 'BABA',
                          'UBER', 'LYFT', 'SNAP', 'ROKU']
        # Still prepend portfolio symbols if possible
        try:
            holdings = load_mystocks()
            portfolio_symbols = []
            for h in holdings:
                if isinstance(h, dict):
                    s = _normalize_symbol(h.get("stock"))
                    if s:
                        portfolio_symbols.append(s)
            seen = set()
            portfolio_symbols = [s for s in portfolio_symbols if not (s in seen or seen.add(s))]
        except Exception:
            portfolio_symbols = []

        display = [f"⭐ {sym} (Portfolio)" for sym in portfolio_symbols]
        symbols = list(portfolio_symbols)
        for sym in fallback_stocks:
            if sym not in symbols:
                display.append(sym)
                symbols.append(sym)
        return display[:20], symbols[:20]

st.title("📈 Stock Analysis with Bollinger Bands & DPO")

# Sidebar for input
st.sidebar.header("Stock Analysis Parameters")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", value="ASTS").upper()

# Get most active stocks for dropdown
if 'active_stocks_display' not in st.session_state or 'active_stocks_symbols' not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading most active stocks..."):
            display_options, symbol_options = get_most_active_stocks()
            st.session_state['active_stocks_display'] = display_options
            st.session_state['active_stocks_symbols'] = symbol_options

# Always decorate the dropdown based on today's cached Fast Lane results.
try:
    _cache_for_dropdown = load_cache()
    _passing = _get_today_fast_lane_symbols_from_cache(_cache_for_dropdown)
    st.session_state['active_stocks_display'] = _decorate_dropdown_with_fast_lane(
        st.session_state.get('active_stocks_display', []),
        st.session_state.get('active_stocks_symbols', []),
        _passing,
    )
except Exception:
    pass

# Dropdown for most active stocks
st.sidebar.markdown("**Or select from most active stocks:**")
selected_active = st.sidebar.selectbox(
    "Top Movers & Active Stocks",
    options=["Select a stock..."] + st.session_state['active_stocks_display'],
    key="active_stock_dropdown"
)

# Update symbol if dropdown selection is made
if selected_active != "Select a stock..." and selected_active:
    # Extract symbol from dropdown selection
    selected_index = st.session_state['active_stocks_display'].index(selected_active)
    symbol = st.session_state['active_stocks_symbols'][selected_index]

# Refresh button for active stocks
if st.sidebar.button("🔄 Refresh Active Stocks"):
    with st.sidebar:
        with st.spinner("Refreshing most active stocks..."):
            display_options, symbol_options = get_most_active_stocks()
            st.session_state['active_stocks_display'] = display_options
            st.session_state['active_stocks_symbols'] = symbol_options
            st.rerun()

# Reserve a spot for the Search button directly below Refresh Active Stocks.
# We'll render into this container later (after helper functions are defined),
# but it will appear here in the sidebar.
search_slot = st.sidebar.container()

# Bulk analysis button
st.sidebar.markdown("---")

# Debug/Reset button for troubleshooting
if st.sidebar.button("🔧 Reset Analysis State", help="Clear any stuck analysis sessions"):
    for key in ['analysis_results', 'analysis_in_progress', 'current_analysis_index', 'total_stocks', 'stock_symbols']:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("✅ Analysis state cleared!")
    st.rerun()

# Show current analysis state for debugging
if st.sidebar.checkbox("🔍 Show Debug Info"):
    st.sidebar.write("**Debug Information:**")
    if 'analysis_in_progress' in st.session_state:
        st.sidebar.write(f"Analysis in progress: {st.session_state.analysis_in_progress}")
    if 'current_analysis_index' in st.session_state:
        st.sidebar.write(f"Current index: {st.session_state.current_analysis_index}")
    if 'total_stocks' in st.session_state:
        st.sidebar.write(f"Total stocks: {st.session_state.total_stocks}")
    st.sidebar.write(f"Cache file exists: {os.path.exists(CACHE_FILE)}")

# Bulk-analysis debug (more verbose)
st.session_state["debug_bulk"] = st.sidebar.checkbox(
    "🐛 Debug bulk analysis",
    value=bool(st.session_state.get("debug_bulk", False)),
    help="Logs bulk analysis events to session state and bulk_debug.log",
)

if st.session_state.get("debug_bulk"):
    with st.sidebar.expander("🐛 Bulk debug log (latest)", expanded=False):
        debug_events = st.session_state.get("_bulk_debug", [])
        st.write(f"Events: {len(debug_events)}")
        if debug_events:
            st.json(debug_events[-50:])
        else:
            st.write("No events yet. Start a bulk scan to populate logs.")

# Show cache status
cache = load_cache()
today_key = get_today_key()
if 'bulk_analysis' in cache and today_key in cache['bulk_analysis']:
    bulk_cache = cache['bulk_analysis'][today_key]
    if is_cache_valid(bulk_cache):
        st.sidebar.success(f"📁 Cached bulk analysis available for today ({len(bulk_cache.get('results', []))} stocks)")
        
        if st.sidebar.button("📋 Show Cached Results"):
            with st.expander("📁 Cached Bulk Analysis Results", expanded=True):
                st.markdown("### 🚀 Fast Lane Stock Scanner (Cached)")
                st.markdown(f"**Cached from:** {bulk_cache['timestamp'][:19]}")
                st.markdown("---")
                
                cached_results = bulk_cache['results']
                failed_cached = [r for r in cached_results if r and r.get('error', False)]
                ok_cached = [r for r in cached_results if r and not r.get('error', False)]
                fast_lane_cached = [r for r in ok_cached if r.get('is_fast_lane', False)]
                other_cached = [r for r in ok_cached if not r.get('is_fast_lane', False)]
                
                if fast_lane_cached:
                    st.markdown("#### 🟢 STOCKS IN THE FAST LANE! (Cached)")
                    for idx, stock in enumerate(fast_lane_cached, 1):
                        st.markdown(
                            f"""<div style='background-color: #1a4b3a; padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #00FF00;'>
<h4 style='color: #00FF00; margin: 0;'>🚀 #{idx} {stock['symbol']} - IN THE FAST LANE!</h4>
<p style='color: #ffffff; margin: 2px 0; font-size: 14px;'>💰 Price: ${stock['price']:.2f} | 📈 DPO: {stock['dpo']:.2f}</p>
</div>""", unsafe_allow_html=True
                        )

                if other_cached:
                    st.markdown("#### 📊 Other Stocks (Cached)")
                    for idx, stock in enumerate(other_cached, 1):
                        status_items = []

                        if not stock.get('positive_dpo_20', stock.get('positive_dpo', False)):
                            status_items.append("❌ Negative DPO-20")
                        else:
                            status_items.append("✅ Positive DPO-20")

                        if not stock.get('positive_dpo_9', False):
                            status_items.append("❌ Negative DPO-9")
                        else:
                            status_items.append("✅ Positive DPO-9")

                        if not stock.get('in_bb_range', False):
                            price_val = float(stock.get('price', 0) or 0)
                            bb1 = float(stock.get('bb_upper_1', 0) or 0)
                            bb3 = float(stock.get('bb_upper_3', 0) or 0)
                            if price_val <= bb1:
                                status_items.append("❌ Below 1σ upper BB")
                            elif bb3 and price_val >= bb3:
                                status_items.append("❌ Above 3σ upper BB")
                            else:
                                status_items.append("❌ Outside BB range")
                        else:
                            status_items.append("✅ In BB Range")

                        st.markdown(
                            f"""<div style='background-color: #2a2a2a; padding: 8px; margin: 3px 0; border-radius: 6px; border-left: 3px solid #FF6B6B;'>
<p style='color: #FF6B6B; margin: 0; font-weight: bold;'>#{idx} {stock.get('symbol','?')} - NOT IN THE FAST LANE</p>
<p style='color: #cccccc; margin: 2px 0; font-size: 12px;'>Price: ${float(stock.get('price', 0) or 0):.2f} | DPO-9: {float(stock.get('dpo_9', 0) or 0):.2f} | DPO-20: {float(stock.get('dpo_20', stock.get('dpo', 0)) or 0):.2f}</p>
<p style='color: #cccccc; margin: 2px 0; font-size: 12px;'>{' | '.join(status_items)}</p>
</div>""",
                            unsafe_allow_html=True,
                        )

                if failed_cached:
                    st.markdown("#### ⚠️ Failures / No Data (Cached)")
                    for idx, stock in enumerate(failed_cached, 1):
                        # Show whatever values were cached (often partial/zeroed), instead of a generic message.
                        price_val = float(stock.get('price', 0) or 0)
                        dpo9_val = float(stock.get('dpo_9', 0) or 0)
                        dpo20_val = float(stock.get('dpo_20', stock.get('dpo', 0)) or 0)
                        bb1_val = stock.get('bb_upper_1')
                        bb3_val = stock.get('bb_upper_3')
                        bb_text = ""
                        if bb1_val is not None and bb3_val is not None:
                            try:
                                bb_text = f" | BB1: ${float(bb1_val):.2f} | BB3: ${float(bb3_val):.2f}"
                            except Exception:
                                bb_text = ""

                        flags = []
                        if stock.get('positive_dpo_20', stock.get('positive_dpo', False)):
                            flags.append("DPO-20+")
                        if stock.get('positive_dpo_9', False):
                            flags.append("DPO-9+")
                        if stock.get('in_bb_range', False):
                            flags.append("InBB")
                        flags_text = (" | " + ", ".join(flags)) if flags else ""

                        st.markdown(
                            f"""<div style='background-color: #2a2a2a; padding: 8px; margin: 3px 0; border-radius: 6px; border-left: 3px solid #FFA500;'>
<p style='color: #FFA500; margin: 0; font-weight: bold;'>#{idx} {stock.get('symbol','?')} - FAILED</p>
<p style='color: #cccccc; margin: 2px 0; font-size: 12px;'>Price: ${price_val:.2f} | DPO-9: {dpo9_val:.2f} | DPO-20: {dpo20_val:.2f}{bb_text}{flags_text}</p>
</div>""",
                            unsafe_allow_html=True,
                        )
                
                # Summary (derive from displayed lists so counts always match)
                total_analyzed = len(cached_results)
                fast_lane_count = len(fast_lane_cached)
                failed_count = len(failed_cached)
                success_rate = (fast_lane_count / total_analyzed * 100) if total_analyzed else 0.0
                st.markdown(f"""<div style='background-color: #2a2a2a; padding: 10px; margin: 10px 0; border-radius: 8px;'>
<h4 style='color: #ffffff; margin: 0;'>📈 Cached Analysis Summary</h4>
<p style='color: #ffffff; margin: 3px 0;'>🔍 Total Analyzed: {total_analyzed}</p>
<p style='color: #00FF00; margin: 3px 0;'>🚀 Fast Lane: {fast_lane_count}</p>
<p style='color: #FFA500; margin: 3px 0;'>⚠️ Failures: {failed_count}</p>
<p style='color: #cccccc; margin: 0;'>📊 Success Rate: {success_rate:.1f}%</p>
</div>""", unsafe_allow_html=True)

if st.sidebar.button("🚀 Do Analysis - Find Fast Lane Stocks!", type="secondary"):
    if 'active_stocks_symbols' in st.session_state:
        # Clear any previous analysis state
        for key in ['analysis_results', 'analysis_in_progress', 'current_analysis_index']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.sidebar.info("⏳ Starting live analysis... Results will appear in real-time!")
        
        with st.expander("🔍 Live Bulk Analysis Results", expanded=True):
            st.markdown("### 🚀 Fast Lane Stock Scanner (Live Updates)")
            st.markdown("**Real-time analysis:** Results appear as each stock is processed...")
            st.markdown("---")
            
            # Start the live analysis
            bulk_analyze_stocks_with_live_updates(st.session_state['active_stocks_symbols'])
    else:
        st.sidebar.error("Please refresh the active stocks list first.")

# --- My Stocks (bottom-left) ---
st.sidebar.markdown("---")

if "show_add_stock" not in st.session_state:
    st.session_state["show_add_stock"] = False

add_clicked = st.sidebar.button("➕ Add Stock", help="Add a holding to mystocks.json")
if add_clicked:
    st.session_state["show_add_stock"] = True

if st.session_state.get("show_add_stock"):
    with st.sidebar.expander("Add stock", expanded=True):
        with st.form("add_stock_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_stock = st.text_input("Stock", placeholder="AAPL").strip().upper()
            with c2:
                new_price = st.number_input("Price", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            with c3:
                new_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.001, format="%.6f")

            submitted = st.form_submit_button("Save")
            if submitted:
                if not new_stock:
                    st.error("Stock is required")
                else:
                    holdings = load_mystocks()
                    holdings.append(
                        {
                            "stock": new_stock,
                            "price": float(new_price),
                            "quantity": float(new_qty),
                            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    save_mystocks(holdings)
                    st.success(f"Saved {new_stock} to {MYSTOCKS_FILE}")

        # Show current holdings (lightweight)
        holdings_now = load_mystocks()
        if holdings_now:
            st.caption("Current holdings")
            st.dataframe(holdings_now, use_container_width=True, hide_index=True)
        else:
            st.caption("No holdings saved yet.")

# Edit holdings (table editor)
with st.sidebar.expander("✏️ Edit holdings", expanded=False):
    current_holdings = load_mystocks()
    if not current_holdings:
        st.write("No holdings saved yet.")
    else:
        # Keep an editable copy in session state so edits persist across reruns.
        if "edited_holdings" not in st.session_state:
            st.session_state["edited_holdings"] = current_holdings

        edited = st.data_editor(
            st.session_state["edited_holdings"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "stock": st.column_config.TextColumn("Stock"),
                "price": st.column_config.NumberColumn("Price", min_value=0.0, format="%.2f"),
                "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.001, format="%.6f"),
                "ts": st.column_config.TextColumn("ts", disabled=True),
            },
            key="holdings_editor",
        )

        c_save, c_reload = st.columns(2)
        with c_save:
            if st.button("💾 Save edits", type="primary"):
                cleaned = []
                for row in (edited or []):
                    if not isinstance(row, dict):
                        continue
                    sym = _normalize_symbol(row.get("stock"))
                    if not sym:
                        continue
                    try:
                        price = float(row.get("price", 0.0))
                    except Exception:
                        price = 0.0
                    try:
                        qty = float(row.get("quantity", 0.0))
                    except Exception:
                        qty = 0.0
                    cleaned.append(
                        {
                            "stock": sym,
                            "price": float(price),
                            "quantity": float(qty),
                            "ts": row.get("ts") or datetime.datetime.now().isoformat(timespec="seconds"),
                        }
                    )

                save_mystocks(cleaned)
                st.session_state["edited_holdings"] = cleaned
                st.success(f"Updated {MYSTOCKS_FILE}")
                st.rerun()

        with c_reload:
            if st.button("↩️ Discard", help="Reload from disk"):
                st.session_state["edited_holdings"] = load_mystocks()
                st.rerun()

# Portfolio P/L (auto-updates on rerun: load/refresh/search/new stock)
holdings_for_pl = load_mystocks()
if holdings_for_pl:
    # Aggregate lots by symbol (weighted average cost)
    agg = {}
    for h in holdings_for_pl:
        sym = _normalize_symbol(h.get("stock") if isinstance(h, dict) else None)
        if not sym:
            continue
        try:
            qty = float(h.get("quantity", 0.0))
        except Exception:
            qty = 0.0
        try:
            cost = float(h.get("price", 0.0))
        except Exception:
            cost = 0.0

        if qty <= 0:
            continue

        a = agg.get(sym) or {"quantity": 0, "cost_value": 0.0}
        a["quantity"] += float(qty)
        a["cost_value"] += qty * cost
        agg[sym] = a

    symbols = list(agg.keys())
    quotes = get_live_prices(symbols, ttl_seconds=60)

    rows = []
    total_cost = 0.0
    total_value = 0.0
    for sym in symbols:
        qty = agg[sym]["quantity"]
        cost_value = float(agg[sym]["cost_value"])
        avg_cost = (cost_value / qty) if qty else 0.0

        live_price = quotes.get(sym, {}).get("price")
        live_ts = quotes.get(sym, {}).get("ts")
        live_value = (float(live_price) * qty) if live_price is not None else None

        pnl = (live_value - cost_value) if live_value is not None else None
        pnl_pct = ((pnl / cost_value) * 100) if (pnl is not None and cost_value > 0) else None

        total_cost += cost_value
        if live_value is not None:
            total_value += live_value

        rows.append(
            {
                "Stock": sym,
                "Quantity": round(float(qty), 6),
                "Avg Cost": round(avg_cost, 4),
                "Live Price": (round(float(live_price), 4) if live_price is not None else None),
                "P/L $": (round(float(pnl), 2) if pnl is not None else None),
                "P/L %": (round(float(pnl_pct), 2) if pnl_pct is not None else None),
                "Updated": live_ts,
            }
        )

    with st.sidebar.expander("📌 My Portfolio (P/L)", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        st.metric("Total Cost", f"${total_cost:,.2f}")
        st.metric("Total Value (live)", f"${total_value:,.2f}")
        st.metric("Total P/L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:+.2f}%")

# Date range
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

def fetch_intraday_price(symbol: str):
    """Fetch a single best-effort intraday price point.

    Returns: (price|None, timestamp|None, market_hours_bool)
    """
    try:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return None, None, False

        market_hours = is_market_hours()
        if not market_hours:
            return None, None, False

        ticker = yf.Ticker(symbol)
        intraday_data = ticker.history(period="1d", interval="1m")
        if intraday_data.empty:
            return None, None, market_hours

        latest_real_time = intraday_data.iloc[-1]
        price = float(latest_real_time['Close'])
        ts = intraday_data.index[-1]
        return price, ts, market_hours
    except Exception:
        return None, None, False

def fetch_stock_data(symbol, update_cache=True):
    """Fetch stock data using yfinance with real-time intraday data"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data for technical indicators (1 year)
        historical_data = ticker.history(start=start_date, end=end_date)
        
        # Pull live price ONCE when the stock is loaded (Search/selection).
        # Avoid continuous polling to keep server usage low.
        live_price, live_ts, market_hours = fetch_intraday_price(symbol)
        if len(historical_data) > 0:
            if live_price is not None:
                historical_data.iloc[-1, historical_data.columns.get_loc('Close')] = live_price
                historical_data.attrs['real_time_price'] = live_price
            else:
                historical_data.attrs['real_time_price'] = historical_data.iloc[-1]['Close']

            historical_data.attrs['real_time_timestamp'] = live_ts or historical_data.index[-1]
            historical_data.attrs['market_hours'] = bool(market_hours)
        
        if historical_data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
            
        # Calculate technical indicators
        historical_data = calculate_bollinger_bands(historical_data)
        historical_data = calculate_dpo_9(historical_data)
        historical_data = calculate_dpo_20(historical_data)
        
        # Add legacy DPO column for backward compatibility
        historical_data['DPO'] = historical_data['DPO_20']
        
        # Update cache for manual searches if requested and no data exists for today
        if update_cache:
            try:
                cache = load_cache()
                today_key = get_today_key()
                
                # Only update if no data exists for today
                if symbol not in cache or today_key not in cache.get(symbol, {}):
                    latest_data = historical_data.iloc[-1]
                    
                    # Calculate Fast Lane status for caching
                    is_positive_dpo_20 = latest_data['DPO_20'] > 0
                    is_positive_dpo_9 = latest_data['DPO_9'] > 0
                    price_between_bb1_bb3 = (latest_data['Close'] > latest_data['BB_Upper_1'] and 
                                             latest_data['Close'] < latest_data['BB_Upper_3'])
                    
                    result = {
                        'symbol': symbol,
                        'price': float(latest_data['Close']),
                        'dpo_9': float(latest_data['DPO_9']),
                        'dpo_20': float(latest_data['DPO_20']),
                        'dpo': float(latest_data['DPO_20']),
                        'bb_upper_1': float(latest_data['BB_Upper_1']),
                        'bb_upper_3': float(latest_data['BB_Upper_3']),
                        'is_fast_lane': bool(is_positive_dpo_20 and is_positive_dpo_9 and price_between_bb1_bb3),
                        'positive_dpo_20': bool(is_positive_dpo_20),
                        'positive_dpo_9': bool(is_positive_dpo_9),
                        'positive_dpo': bool(is_positive_dpo_20),
                        'in_bb_range': bool(price_between_bb1_bb3),
                        'error': False,
                        'manual_search': True  # Flag to indicate this was a manual search
                    }
                    
                    # Save to cache
                    if symbol not in cache:
                        cache[symbol] = {}
                    
                    cache[symbol][today_key] = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'data': result
                    }
                    
                    save_cache(cache)
                    st.sidebar.success(f"📁 Cached analysis for {symbol}")
                    
            except Exception as cache_error:
                st.sidebar.warning(f"Cache update failed for {symbol}: {str(cache_error)}")
        
        return historical_data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def is_market_hours():
    """Check if market is currently open (simplified)"""
    now = datetime.datetime.now()
    # Simplified check for US market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    
    # Basic check (doesn't account for holidays or exact timezone)
    if weekday < 5 and 9 <= hour < 16:  # Monday-Friday, 9 AM - 4 PM
        return True
    return False

# Search button or auto-search when dropdown selection changes
with search_slot:
    search_triggered = st.button("🔍 Search", type="primary") or (selected_active != "Select a stock..." and selected_active)

if search_triggered:
    if symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            stock_data = fetch_stock_data(symbol)
            
        if stock_data is not None:
            # Store data in session state
            st.session_state['stock_data'] = stock_data
            st.session_state['symbol'] = symbol

# Display analysis if data exists
if 'stock_data' in st.session_state and st.session_state['stock_data'] is not None:
    data = st.session_state['stock_data']
    current_symbol = st.session_state['symbol']

    # If this symbol is in the user's portfolio, show a P/L summary table at the top.
    try:
        sym_norm = _normalize_symbol(current_symbol)
        holdings_for_symbol = [
            h for h in (load_mystocks() or [])
            if isinstance(h, dict) and _normalize_symbol(h.get("stock")) == sym_norm
        ]

        if holdings_for_symbol:
            qty_total = 0.0
            cost_value = 0.0
            for h in holdings_for_symbol:
                try:
                    qty = float(h.get("quantity", 0.0) or 0.0)
                except Exception:
                    qty = 0.0
                try:
                    cost = float(h.get("price", 0.0) or 0.0)
                except Exception:
                    cost = 0.0

                if qty <= 0:
                    continue
                qty_total += qty
                cost_value += qty * cost

            if qty_total > 0:
                quotes = get_live_prices([sym_norm], ttl_seconds=60)
                live_price = quotes.get(sym_norm, {}).get("price")
                live_ts = quotes.get(sym_norm, {}).get("ts")

                # Fallback: use last known close from loaded data if live quote isn't available.
                if live_price is None:
                    try:
                        live_price = float(data.iloc[-1]["Close"])
                    except Exception:
                        live_price = None

                avg_cost = (cost_value / qty_total) if qty_total else 0.0
                live_value = (float(live_price) * qty_total) if live_price is not None else None
                pnl = (live_value - cost_value) if live_value is not None else None
                pnl_pct = ((pnl / cost_value) * 100) if (pnl is not None and cost_value > 0) else None

                st.markdown("### 📌 Portfolio P/L (This Symbol)")
                pl_df = pd.DataFrame(
                    [
                        {
                            "Stock": sym_norm,
                            "Quantity": round(qty_total, 6),
                            "Avg Cost": round(avg_cost, 4),
                            "Live Price": (round(float(live_price), 4) if live_price is not None else None),
                            "Cost Value": round(cost_value, 2),
                            "Live Value": (round(float(live_value), 2) if live_value is not None else None),
                            "P/L $": (round(float(pnl), 2) if pnl is not None else None),
                            "P/L %": (round(float(pnl_pct), 2) if pnl_pct is not None else None),
                            "Updated": live_ts,
                        }
                    ]
                )

                def _pl_color(v):
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return ""
                        x = float(v)
                    except Exception:
                        return ""
                    if x > 0:
                        return "color: #00FF00; font-weight: 700;"
                    if x < 0:
                        return "color: #FF0000; font-weight: 700;"
                    return ""

                def _fmt_currency(v):
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return ""
                        x = float(v)
                        return f"${x:,.2f}"
                    except Exception:
                        return ""

                pl_styler = pl_df.style
                if "P/L $" in pl_df.columns:
                    pl_styler = pl_styler.map(_pl_color, subset=["P/L $"])
                if "P/L %" in pl_df.columns:
                    pl_styler = pl_styler.map(_pl_color, subset=["P/L %"])

                if "P/L $" in pl_df.columns:
                    pl_styler = pl_styler.format({"P/L $": _fmt_currency})

                st.dataframe(pl_styler, hide_index=True, use_container_width=True)
                st.markdown("---")
    except Exception:
        pass

    # Manual real-time refresh (no background polling)
    refresh_col, _ = st.columns([1, 3])
    with refresh_col:
        if st.button("🔄 Refresh Real-Time Price", help="Fetch latest 1-minute price for the loaded symbol"):
            live_price, live_ts, market_hours_now = fetch_intraday_price(current_symbol)
            if live_price is not None:
                try:
                    data.iloc[-1, data.columns.get_loc('Close')] = live_price
                except Exception:
                    pass
                data.attrs['real_time_price'] = live_price
                data.attrs['real_time_timestamp'] = live_ts
                data.attrs['market_hours'] = bool(market_hours_now)
                st.session_state['stock_data'] = data
                st.rerun()
            else:
                st.info("No intraday update available right now (likely outside market hours).")
    
    # Get latest values for display
    latest_data = data.iloc[-1]
    
    # Get real-time information
    real_time_price = getattr(data, 'attrs', {}).get('real_time_price', latest_data['Close'])
    real_time_timestamp = getattr(data, 'attrs', {}).get('real_time_timestamp', data.index[-1])
    market_hours = getattr(data, 'attrs', {}).get('market_hours', False)
    
    # Display real-time information at the top
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"💹 {current_symbol} Real-Time Price",
            value=f"${real_time_price:.2f}",
            delta=f"{((real_time_price - latest_data['Close']) / latest_data['Close'] * 100):+.2f}%" if real_time_price != latest_data['Close'] else None
        )
    
    with col2:
        market_status = "🟢 Market Hours" if market_hours else "🔴 After Hours"
        st.metric(
            label="Market Status",
            value=market_status
        )
    
    with col3:
        # Time since last update
        if isinstance(real_time_timestamp, str):
            timestamp_str = real_time_timestamp[:19]
        else:
            timestamp_str = real_time_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        st.metric(
            label="⏰ Last Update",
            value=timestamp_str[-8:]  # Show just the time part
        )
    
    with col4:
        # Data freshness indicator
        now = datetime.datetime.now()
        if isinstance(real_time_timestamp, str):
            last_update = datetime.datetime.fromisoformat(real_time_timestamp.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            last_update = real_time_timestamp.replace(tzinfo=None) if real_time_timestamp.tzinfo else real_time_timestamp
        
        time_diff = now - last_update
        minutes_ago = int(time_diff.total_seconds() / 60)
        
        if minutes_ago < 1:
            freshness = "🟢 Live"
        elif minutes_ago < 15:
            freshness = f"🟡 {minutes_ago}m ago"
        else:
            freshness = f"🔴 {minutes_ago}m ago"
            
        st.metric(
            label="Data Freshness",
            value=freshness
        )
    
    st.markdown("---")
    
    # FAST LANE Signal Logic (using real-time price and both DPO periods)
    is_positive_dpo_20 = latest_data['DPO_20'] > 0
    is_positive_dpo_9 = latest_data['DPO_9'] > 0
    price_between_bb1_bb3 = (real_time_price > latest_data['BB_Upper_1'] and 
                             real_time_price < latest_data['BB_Upper_3'])
    
    # Display FAST LANE signal with enhanced criteria
    if is_positive_dpo_20 and is_positive_dpo_9 and price_between_bb1_bb3:
        st.markdown(
            "<h1 style='text-align: center; color: #00FF00; font-size: 48px; font-weight: bold; text-shadow: 2px 2px 4px #000000;'>🚀 In the FAST LANE! 🚀</h1>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #00AA00; font-size: 18px;'>✅ Enhanced criteria met: Positive momentum (9-day & 20-day DPO) + Price between 1σ and 3σ Bollinger Bands!</p>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='text-align: center; color: #FF0000; font-size: 24px;'>🚫 NO GO</h3>", 
            unsafe_allow_html=True
        )
        reasons = []
        if not is_positive_dpo_20:
            reasons.append("❌ DPO-20 is negative (below trend)")
        if not is_positive_dpo_9:
            reasons.append("❌ DPO-9 is negative (short-term below trend)")
        if not price_between_bb1_bb3:
            if real_time_price <= latest_data['BB_Upper_1']:
                reasons.append("❌ Price below 1σ upper Bollinger Band")
            else:
                reasons.append("❌ Price above 3σ upper Bollinger Band (overbought)")
        
        st.markdown(
            f"<p style='text-align: center; color: #AA0000; font-size: 14px;'>{'<br>'.join(reasons)}</p>", 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Display current values
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Bollinger Bands (1 Standard Deviation)")
        st.markdown(
            f"""
<div style="margin: 0; padding: 0.25rem 0 0.25rem 0;">
    <div style="font-size: 1rem; line-height: 1.25rem;">Upper Band (+1σ)</div>
  <div style="font-size: 1.5rem; line-height: 1.8rem; font-weight: 600; color: #FFFF00;">${latest_data['BB_Upper_1']:.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.metric("Moving Average (20-day)", f"${latest_data['MA_20']:.2f}")
        st.metric("Lower Band (-1σ)", f"${latest_data['BB_Lower_1']:.2f}")
    
    with col2:
        st.subheader("📊 Bollinger Bands (3 Standard Deviation)")
        st.metric("Upper Band (+3σ)", f"${latest_data['BB_Upper_3']:.2f}")
        st.metric("Real-Time Price", f"${real_time_price:.2f}")  # Use real-time price
        st.metric("Lower Band (-3σ)", f"${latest_data['BB_Lower_3']:.2f}")
    
    # Create synchronized subplots with three rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{current_symbol} Stock Price with Bollinger Bands', 
                       'Detrended Price Oscillator (DPO) - 9 Days', 
                       'Detrended Price Oscillator (DPO) - 20 Days'],
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Add stock price and Bollinger Bands to the first subplot
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            name='Close Price',
            line=dict(color='#1f77b4', width=4),  # Thicker, more prominent blue
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Add 20-day moving average
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['MA_20'], 
            mode='lines', 
            name='20-Day MA',
            line=dict(color='#ff7f0e', width=3, dash='dash'),  # Thicker orange dashed line
            hovertemplate='<b>Date</b>: %{x}<br><b>MA</b>: $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Add Bollinger Bands (upper region +1σ to +3σ, then 1σ channel)
    # Order matters for Plotly 'fill=tonexty':
    # - BB_Upper_3 (no fill)
    # - BB_Upper_1 (fills to BB_Upper_3 => shades between upper bands)
    # - BB_Lower_1 (fills to BB_Upper_1 => shades 1σ channel)
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Upper_3'], 
            mode='lines', 
            name='BB Upper (3σ)',
            line=dict(color='#FF4500', width=2, dash='dot'),  # Bright orange-red
            hovertemplate='<b>Date</b>: %{x}<br><b>Upper BB (3σ)</b>: $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Upper_1'],
            mode='lines',
            name='BB Upper (1σ)',
            line=dict(color='#FFFF00', width=4),  # Bright yellow
            fill='tonexty',
            fillcolor='rgba(255,255,0,0.06)',  # Light shading between +1σ and +3σ
            hovertemplate='<b>Date</b>: %{x}<br><b>Upper BB (1σ)</b>: $%{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Lower_1'], 
            mode='lines', 
            name='BB Lower (1σ)',
            line=dict(color='#FFA500', width=3),  # Bright orange
            fill='tonexty',
            fillcolor='rgba(255,255,0,0.10)',  # Light yellow fill
            hovertemplate='<b>Date</b>: %{x}<br><b>Lower BB (1σ)</b>: $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )

    # Highlight latest +1σ value on the chart in yellow (kept after filled bands)
    fig.add_trace(
        go.Scatter(
            x=[data.index[-1]],
            y=[latest_data['BB_Upper_1']],
            mode='markers+text',
            text=[f"${latest_data['BB_Upper_1']:.2f}"],
            textposition='top right',
            textfont=dict(color='#FFFF00', size=12),
            marker=dict(color='#FFFF00', size=10),
            name='BB Upper (1σ) Latest',
            showlegend=False,
            hovertemplate='<b>Date</b>: %{x}<br><b>Upper BB (1σ)</b>: $%{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['BB_Lower_3'], 
            mode='lines', 
            name='BB Lower (3σ)',
            line=dict(color='#FFD700', width=2, dash='dot'),  # Bright gold
            hovertemplate='<b>Date</b>: %{x}<br><b>Lower BB (3σ)</b>: $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Add current stock price indicator (green dot) using real-time price
    current_date = data.index[-1]
    current_price = real_time_price  # Use real-time price instead of close
    
    fig.add_trace(
        go.Scatter(
            x=[current_date], 
            y=[current_price], 
            mode='markers', 
            name='Real-Time Price',
            marker=dict(
                color='#00FF00' if market_hours else '#FFA500',  # Green if market open, orange if closed
                size=12,  # Smaller size as requested
                symbol='circle',
                line=dict(color='#000000', width=2)  # Black border
            ),
            hovertemplate=f'<b>Real-Time Price</b><br><b>Time</b>: {timestamp_str}<br><b>Price</b>: $%{{y:.2f}}<br><b>Status</b>: {"🚀 FAST LANE" if is_positive_dpo_20 and is_positive_dpo_9 and price_between_bb1_bb3 else "🛑 NO GO"}<br><b>DPO-9</b>: {latest_data["DPO_9"]:.2f}<br><b>DPO-20</b>: {latest_data["DPO_20"]:.2f}<br><b>Market</b>: {market_status}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Add 9-day DPO to the second subplot
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['DPO_9'], 
            mode='lines', 
            name='DPO (9-day)',
            line=dict(color='#ff6b35', width=3),  # Orange-red for 9-day
            hovertemplate='<b>Date</b>: %{x}<br><b>DPO-9</b>: %{y:.2f}<extra></extra>'
        ), 
        row=2, col=1
    )
    
    # Add zero line for 9-day DPO
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Add 20-day DPO to the third subplot
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['DPO_20'], 
            mode='lines', 
            name='DPO (20-day)',
            line=dict(color='#2ca02c', width=3),  # Green for 20-day
            hovertemplate='<b>Date</b>: %{x}<br><b>DPO-20</b>: %{y:.2f}<extra></extra>'
        ), 
        row=3, col=1
    )
    
    # Add zero line for 20-day DPO
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{current_symbol} Enhanced Technical Analysis (9-day & 20-day DPO)',
        height=900,  # Increased height for three subplots
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="DPO-9 Value", row=2, col=1)
    fig.update_yaxes(title_text="DPO-20 Value", row=3, col=1)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information
    st.subheader("📋 Technical Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Bollinger Band Position (1σ):**")
        bb_position_1 = ((real_time_price - latest_data['BB_Lower_1']) / 
                         (latest_data['BB_Upper_1'] - latest_data['BB_Lower_1'])) * 100
        st.write(f"Position: {bb_position_1:.1f}% (0% = Lower Band, 100% = Upper Band)")
    
    with col2:
        st.write("**Bollinger Band Position (3σ):**")
        bb_position_3 = ((real_time_price - latest_data['BB_Lower_3']) / 
                         (latest_data['BB_Upper_3'] - latest_data['BB_Lower_3'])) * 100
        st.write(f"Position: {bb_position_3:.1f}% (0% = Lower Band, 100% = Upper Band)")
    
    with col3:
        st.write("**Latest DPO Values:**")
        st.write(f"9-day DPO: {latest_data['DPO_9']:.2f}")
        st.write(f"20-day DPO: {latest_data['DPO_20']:.2f}")
        dpo_status_9 = "✅ Positive" if latest_data['DPO_9'] > 0 else "❌ Negative"
        dpo_status_20 = "✅ Positive" if latest_data['DPO_20'] > 0 else "❌ Negative"
        st.write(f"9-day Status: {dpo_status_9}")
        st.write(f"20-day Status: {dpo_status_20}")
    
    # Data table (optional)
    if st.checkbox("Show Raw Data"):
        st.subheader("📊 Data Table")
        display_columns = ['Close', 'MA_20', 'BB_Upper_1', 'BB_Lower_1', 'BB_Upper_3', 'BB_Lower_3', 'DPO_9', 'DPO_20']
        st.dataframe(data[display_columns].tail(20))

else:
    st.info("👆 Enter a stock symbol in the sidebar and click 'Search' to begin analysis, or select from the most active stocks dropdown.")
    
    # Show example usage
    st.subheader("📖 How to Use This App")
    st.write("""
    **Individual Stock Analysis:**
    1. **Enter a stock symbol** (e.g., AAPL, TSLA, MSFT) in the sidebar, OR
    2. **Select from the dropdown** of most active stocks (top and bottom movers)
    3. **Click the Search button** or the dropdown will auto-analyze
    
    **Bulk Analysis (New!):**
    4. **Click "Do Analysis"** to scan all stocks in the dropdown for Fast Lane opportunities
    5. **View results** showing which stocks meet the Fast Lane criteria
    
    **Charts & Data:**
    6. **View the charts** showing:
       - Stock price with Bollinger Bands (1σ and 3σ)
       - 20-day moving average
       - Detrended Price Oscillator (DPO) below the main chart
    7. **Check the numerical values** displayed above the charts
    8. **Hover over the charts** for detailed information at specific dates
    9. **Use the refresh button** to update the most active stocks list
    
    **Understanding the Indicators:**
    - **Bollinger Bands**: Show volatility and potential support/resistance levels
    - **DPO**: Removes the trend to show cyclical patterns in price movements
    - **Fast Lane Criteria**: Positive DPO + Price between 1σ and 3σ Bollinger Bands
    
    **Most Active Stocks Features:**
    - 📈 Green arrows indicate gainers, 📉 red arrows indicate losers
    - List updates with current market movers
    - Shows percentage change for quick reference
    - Bulk analysis scans all stocks with server-friendly delays
    """)