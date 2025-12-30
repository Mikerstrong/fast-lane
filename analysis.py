import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import json
import os
import traceback


DEBUG_LOG_FILE = "bulk_debug.log"


def fetch_live_price(symbol: str):
    """Best-effort live price fetch for a single symbol.

    Returns: (price|None, timestamp|None)
    """
    symbol = (symbol or "").strip().upper()
    if symbol.startswith("$"):
        symbol = symbol[1:]
    if not symbol:
        return None, None

    # Try fast_info first (usually cheaper than intraday candles).
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

    # Fallback: last 1-minute candle from today.
    try:
        t = yf.Ticker(symbol)
        intraday = t.history(period="1d", interval="1m")
        if intraday is None or intraday.empty:
            return None, None
        last_row = intraday.iloc[-1]
        return float(last_row["Close"]), intraday.index[-1]
    except Exception:
        return None, None


def refresh_result_with_price(result: dict, price: float | None, ts) -> dict:
    """Update an existing cached result with a new price and recompute lane status.

    This intentionally does NOT recompute indicators (DPO/BB) to avoid expensive
    history fetches. It uses cached band levels + DPO signs and updates:
    - price
    - in_bb_range
    - is_fast_lane
    """
    if not result:
        return result

    if price is None:
        return result

    try:
        result = dict(result)
        result["price"] = float(price)
        result["live_price_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        bb1 = result.get("bb_upper_1")
        bb3 = result.get("bb_upper_3")
        if bb1 is not None and bb3 is not None:
            in_range = (float(price) > float(bb1)) and (float(price) < float(bb3))
            result["in_bb_range"] = bool(in_range)
        else:
            # Can't recompute band-range; leave as-is.
            in_range = result.get("in_bb_range", False)

        pos20 = bool(result.get("positive_dpo_20", result.get("positive_dpo", False)))
        pos9 = bool(result.get("positive_dpo_9", False))
        result["is_fast_lane"] = bool(pos20 and pos9 and bool(in_range))
        return result
    except Exception:
        return result


def upsert_symbol_cache(cache: dict, symbol: str, today_key: str, data: dict) -> None:
    """Write/update a per-symbol cache entry (best-effort)."""
    try:
        if not symbol:
            return
        symbol = symbol.strip().upper()
        if symbol.startswith("$"):
            symbol = symbol[1:]
        if not symbol:
            return

        if symbol not in cache:
            cache[symbol] = {}

        cache[symbol][today_key] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data,
        }
    except Exception:
        return


def upsert_bulk_cache(cache: dict, today_key: str, results: list) -> None:
    """Write/update the bulk_analysis cache entry for today (best-effort)."""
    try:
        bulk_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_analyzed': len(results),
                'fast_lane_count': len([r for r in results if isinstance(r, dict) and r.get('is_fast_lane', False)]),
                # Include all symbols so UI counts match the results list.
                'symbols_analyzed': [r.get('symbol') for r in results if isinstance(r, dict) and r.get('symbol')],
            },
        }

        if 'bulk_analysis' not in cache:
            cache['bulk_analysis'] = {}
        cache['bulk_analysis'][today_key] = bulk_results
    except Exception:
        return


def _debug_is_enabled() -> bool:
    """Best-effort flag controlled by Streamlit session state."""
    try:
        return bool(st.session_state.get("debug_bulk", False))
    except Exception:
        return False


def debug_log(event: str, **data) -> None:
    """Log debug events to session_state and an on-disk JSONL file.

    This is intentionally safe/best-effort: it should never break the app.
    """
    if not _debug_is_enabled():
        return

    try:
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **data,
        }

        # In-memory ring buffer (for UI display)
        buf = st.session_state.get("_bulk_debug", [])
        buf.append(entry)
        st.session_state["_bulk_debug"] = buf[-200:]

        # On-disk JSONL (for post-mortem)
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        # Never let logging break analysis.
        return

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands with 1 and 3 standard deviations"""
    # Calculate moving average
    df['MA_20'] = df['Close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    df['STD_20'] = df['Close'].rolling(window=window).std()
    
    # Calculate Bollinger Bands for 1 std dev
    df['BB_Upper_1'] = df['MA_20'] + (df['STD_20'] * 1)
    df['BB_Lower_1'] = df['MA_20'] - (df['STD_20'] * 1)
    
    # Calculate Bollinger Bands for 3 std dev
    df['BB_Upper_3'] = df['MA_20'] + (df['STD_20'] * 3)
    df['BB_Lower_3'] = df['MA_20'] - (df['STD_20'] * 3)
    
    return df

def calculate_dpo(df, period=20):
    """Calculate Detrended Price Oscillator (DPO)"""
    # DPO = Close - Simple Moving Average of (period/2 + 1) periods ago
    shift_period = int(period / 2) + 1
    df[f'DPO_{period}'] = df['Close'] - df['Close'].rolling(window=period).mean().shift(shift_period)
    return df

def calculate_dpo_9(df):
    """Calculate 9-day Detrended Price Oscillator (DPO)"""
    return calculate_dpo(df, period=9)

def calculate_dpo_20(df):
    """Calculate 20-day Detrended Price Oscillator (DPO)"""
    return calculate_dpo(df, period=20)

def load_cache():
    """Load analysis cache from JSON file"""
    cache_file = "stock_analysis_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.warning(f"Cache loading error: {e}")
    return {}

def save_cache(cache_data):
    """Save analysis cache to JSON file"""
    cache_file = "stock_analysis_cache.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        st.sidebar.warning(f"Cache saving error: {e}")

def get_today_key():
    """Get today's date as cache key"""
    return datetime.datetime.now().strftime("%Y-%m-%d")

def is_cache_valid(cache_entry, hours_valid=24):
    """Check if cache entry is still valid"""
    try:
        cached_time = datetime.datetime.fromisoformat(cache_entry['timestamp'])
        time_diff = datetime.datetime.now() - cached_time
        return time_diff.total_seconds() < (hours_valid * 3600)
    except:
        return False

def analyze_fast_lane_stock(symbol, use_cache=True):
    """Analyze a single stock for Fast Lane criteria with caching"""
    try:
        # Normalize symbols coming from screeners (sometimes prefixed with '$')
        original_symbol = symbol
        symbol = (symbol or "").strip().upper()
        if symbol.startswith("$"):
            symbol = symbol[1:]
        if not symbol:
            debug_log("analyze_invalid_symbol", symbol=original_symbol)
            return None

        debug_log("analyze_start", symbol=symbol, use_cache=use_cache)
        cache = load_cache()
        today_key = get_today_key()
        
        # Check cache first
        if use_cache and symbol in cache and today_key in cache[symbol]:
            cached_result = cache[symbol][today_key]
            if is_cache_valid(cached_result):
                debug_log("analyze_cache_hit", symbol=symbol)
                return cached_result['data']
        
        # Add small delay to be server-friendly
        time.sleep(0.5)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=datetime.datetime.now() - datetime.timedelta(days=365), 
                             end=datetime.datetime.now())
        
        if data.empty or len(data) < 20:
            st.warning(f"Insufficient data for {symbol}")
            debug_log("analyze_insufficient_data", symbol=symbol, rows=int(len(data)))
            return None
            
        # Calculate technical indicators
        data = calculate_bollinger_bands(data)
        data = calculate_dpo_9(data)
        data = calculate_dpo_20(data)
        
        # Add legacy DPO column for backward compatibility
        data['DPO'] = data['DPO_20']
        
        # Get latest values
        latest_data = data.iloc[-1]
        
        # Enhanced Fast Lane criteria (Task #12)
        is_positive_dpo_20 = latest_data['DPO_20'] > 0
        is_positive_dpo_9 = latest_data['DPO_9'] > 0
        price_between_bb1_bb3 = (latest_data['Close'] > latest_data['BB_Upper_1'] and 
                                 latest_data['Close'] < latest_data['BB_Upper_3'])
        
        result = {
            'symbol': symbol,
            'price': float(latest_data['Close']),
            'dpo_9': float(latest_data['DPO_9']),
            'dpo_20': float(latest_data['DPO_20']),
            'dpo': float(latest_data['DPO_20']),  # For backward compatibility
            'bb_upper_1': float(latest_data['BB_Upper_1']),
            'bb_upper_3': float(latest_data['BB_Upper_3']),
            'is_fast_lane': bool(is_positive_dpo_20 and is_positive_dpo_9 and price_between_bb1_bb3),
            'positive_dpo_20': bool(is_positive_dpo_20),
            'positive_dpo_9': bool(is_positive_dpo_9),
            'positive_dpo': bool(is_positive_dpo_20),  # For backward compatibility
            'in_bb_range': bool(price_between_bb1_bb3),
            'error': False
        }
        
        # Save to cache
        if symbol not in cache:
            cache[symbol] = {}
        
        cache[symbol][today_key] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data': result
        }
        
        save_cache(cache)
        
        debug_log(
            "analyze_success",
            symbol=symbol,
            is_fast_lane=result.get("is_fast_lane", False),
            price=result.get("price"),
        )
        return result
    except Exception as e:
        # yfinance will raise for delisted / missing price data.
        # We want bulk scans to keep going instead of showing scary errors.
        message = str(e)
        lower = message.lower()
        if (
            "possibly delisted" in lower
            or "no price data found" in lower
            or "no data found" in lower
            or "symbol may be delisted" in lower
        ):
            debug_log(
                "analyze_skip_delisted",
                symbol=symbol,
                error=message,
                traceback=traceback.format_exc(limit=20),
            )
            return None

        st.error(f"Error analyzing {symbol}: {message}")
        debug_log(
            "analyze_exception",
            symbol=symbol,
            error=message,
            traceback=traceback.format_exc(limit=50),
        )
        return None

def bulk_analyze_stocks_with_live_updates(stock_symbols):
    """Analyze all stocks with proper live updates using session state"""
    try:
        debug_log(
            "bulk_enter",
            provided_count=len(stock_symbols) if stock_symbols is not None else None,
        )
        # Initialize session state for live updates
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'analysis_in_progress' not in st.session_state:
            st.session_state.analysis_in_progress = False
        if 'current_analysis_index' not in st.session_state:
            st.session_state.current_analysis_index = 0
        
        # Start analysis if not in progress
        if not st.session_state.analysis_in_progress:
            # Pre-filter using today's cache to avoid unnecessary yfinance calls.
            raw_symbols = list(stock_symbols or [])
            cache = load_cache()
            today_key = get_today_key()

            cached_results = []
            remaining_symbols = []
            cached_symbols = []

            for sym in raw_symbols:
                normalized = (sym or "").strip().upper()
                if normalized.startswith("$"):
                    normalized = normalized[1:]

                if not normalized:
                    continue

                try:
                    if normalized in cache and today_key in cache.get(normalized, {}):
                        cached_entry = cache[normalized][today_key]
                        if is_cache_valid(cached_entry):
                            cached_data = cached_entry.get("data")
                            if cached_data:
                                cached_results.append(cached_data)
                                cached_symbols.append(normalized)
                                continue
                except Exception:
                    # If cache entry is malformed, fall back to analyzing.
                    pass

                remaining_symbols.append(normalized)

            # Refresh cached results with live prices (cheap) so "Do Analysis" always
            # reflects current price vs bands and updates JSON.
            refreshed_cached_results = []
            if cached_symbols:
                debug_log("bulk_refresh_cached_start", count=int(len(cached_symbols)))
                for r in cached_results:
                    sym = (r.get("symbol") or "").strip().upper()
                    if sym.startswith("$"):
                        sym = sym[1:]
                    price, ts = fetch_live_price(sym)
                    refreshed = refresh_result_with_price(r, price, ts)
                    refreshed_cached_results.append(refreshed)
                    upsert_symbol_cache(cache, sym, today_key, refreshed)
                # Persist refreshed per-symbol entries AND bulk snapshot immediately,
                # so "Show Cached Results" matches what the live view shows.
                upsert_bulk_cache(cache, today_key, refreshed_cached_results)
                save_cache(cache)
                debug_log("bulk_refresh_cached_done", count=int(len(cached_symbols)))
            else:
                refreshed_cached_results = cached_results

                # Still ensure bulk snapshot exists if we only have cached data.
                try:
                    upsert_bulk_cache(cache, today_key, refreshed_cached_results)
                    save_cache(cache)
                except Exception:
                    pass

            st.session_state.analysis_results = refreshed_cached_results
            st.session_state.current_analysis_index = 0
            st.session_state.stock_symbols = remaining_symbols
            st.session_state.total_stocks = len(remaining_symbols)
            st.session_state.original_total_stocks = len(raw_symbols)
            st.session_state.skipped_from_cache = len(refreshed_cached_results)

            debug_log(
                "bulk_start",
                original_total=int(len(raw_symbols)),
                skipped_from_cache=int(len(cached_results)),
                remaining_total=int(len(remaining_symbols)),
                remaining_symbols=remaining_symbols[:50],
            )

            if remaining_symbols:
                st.session_state.analysis_in_progress = True
                st.info(
                    f"🚀 Starting bulk analysis: {len(remaining_symbols)} to fetch, {len(refreshed_cached_results)} refreshed from cache (today)."
                )
            else:
                # Nothing to do; everything is already cached for today.
                st.session_state.analysis_in_progress = False
                st.success(
                    f"✅ All {len(refreshed_cached_results)} symbols were refreshed from today's cache. No new history fetches needed."
                )
        
        # Display current results
        if st.session_state.analysis_results:
            fast_lane_stocks = [r for r in st.session_state.analysis_results if r and r.get('is_fast_lane', False)]
            other_stocks = [r for r in st.session_state.analysis_results if r and not r.get('is_fast_lane', False)]
            
            # Display Fast Lane stocks
            if fast_lane_stocks:
                st.markdown("#### 🟢 STOCKS IN THE FAST LANE!")
                for idx, stock in enumerate(fast_lane_stocks, 1):
                    st.markdown(
                        f"""<div style='background-color: #1a4b3a; padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #00FF00;'>
<h4 style='color: #00FF00; margin: 0;'>🚀 #{idx} {stock['symbol']} - IN THE FAST LANE!</h4>
<p style='color: #ffffff; margin: 2px 0; font-size: 14px;'>💰 Price: ${stock['price']:.2f} | 📈 DPO-9: {stock.get('dpo_9', 0):.2f} | DPO-20: {stock.get('dpo_20', stock.get('dpo', 0)):.2f}</p>
</div>""", unsafe_allow_html=True
                    )
            
            # Display other stocks
            if other_stocks:
                st.markdown("#### 📊 Other Stocks")
                for idx, stock in enumerate(other_stocks, 1):
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
                        if stock.get('price', 0) <= stock.get('bb_upper_1', 0):
                            status_items.append("❌ Below 1σ BB")
                        else:
                            status_items.append("❌ Above 3σ BB")
                    else:
                        status_items.append("✅ In BB Range")
                    
                    st.markdown(
                        f"""<div style='background-color: #2a2a2a; padding: 8px; margin: 3px 0; border-radius: 6px; border-left: 3px solid #FF6B6B;'>
<p style='color: #FF6B6B; margin: 0; font-weight: bold;'>#{idx} {stock['symbol']} - NOT IN THE FAST LANE</p>
<p style='color: #cccccc; margin: 2px 0; font-size: 12px;'>{' | '.join(status_items)}</p>
</div>""", unsafe_allow_html=True
                    )
            
            # Display summary
            total_analyzed = len(st.session_state.analysis_results)
            fast_lane_count = len(fast_lane_stocks)
            total_target = st.session_state.get("original_total_stocks", st.session_state.total_stocks)
            cached_skips = st.session_state.get("skipped_from_cache", 0)
            st.markdown(f"""<div style='background-color: #2a2a2a; padding: 10px; margin: 10px 0; border-radius: 8px;'>
<h4 style='color: #ffffff; margin: 0;'>📊 Live Analysis Summary</h4>
<p style='color: #ffffff; margin: 3px 0;'>🔍 Analyzed: {total_analyzed}/{total_target} (cached skipped: {cached_skips})</p>
<p style='color: #00FF00; margin: 3px 0;'>🚀 Fast Lane: {fast_lane_count}</p>
<p style='color: #cccccc; margin: 0;'>📊 Success Rate: {(fast_lane_count/total_analyzed*100):.1f}%</p>
</div>""", unsafe_allow_html=True)
        
        # Continue analysis if in progress
        if (st.session_state.analysis_in_progress and 
            st.session_state.current_analysis_index < st.session_state.total_stocks):
            
            current_symbol = st.session_state.stock_symbols[st.session_state.current_analysis_index]
            debug_log(
                "bulk_analyze_one_start",
                index=int(st.session_state.current_analysis_index),
                total=int(st.session_state.total_stocks),
                symbol=current_symbol,
            )
            
            # Show current analysis status
            progress_placeholder = st.empty()
            progress_placeholder.info(f"Analyzing {current_symbol}... ({st.session_state.current_analysis_index + 1}/{st.session_state.total_stocks})")
            
            # Analyze current stock
            try:
                result = analyze_fast_lane_stock(current_symbol, use_cache=True)
                if result:
                    # Refresh the returned result with a live price (if available)
                    # and persist back to JSON so the cache stays current.
                    price, ts = fetch_live_price(current_symbol)
                    result = refresh_result_with_price(result, price, ts)
                    try:
                        cache = load_cache()
                        today_key = get_today_key()
                        upsert_symbol_cache(cache, current_symbol, today_key, result)
                        save_cache(cache)
                    except Exception:
                        pass

                    st.session_state.analysis_results.append(result)
                    progress_placeholder.success(f"✅ Completed {current_symbol} - {'🚀 FAST LANE' if result.get('is_fast_lane', False) else '🛑 NO GO'}")
                    debug_log(
                        "bulk_analyze_one_success",
                        symbol=current_symbol,
                        is_fast_lane=result.get("is_fast_lane", False),
                    )
                else:
                    st.session_state.analysis_results.append({
                        'symbol': current_symbol,
                        'error': True,
                        'price': 0,
                        'dpo': 0,
                        'dpo_9': 0,
                        'dpo_20': 0,
                        'is_fast_lane': False,
                        'positive_dpo': False,
                        'positive_dpo_9': False,
                        'positive_dpo_20': False,
                        'in_bb_range': False
                    })
                    progress_placeholder.warning(f"⚠️ Could not analyze {current_symbol} - data unavailable")
                    debug_log("bulk_analyze_one_none", symbol=current_symbol)
            except Exception as e:
                st.error(f"Error analyzing {current_symbol}: {str(e)}")
                debug_log(
                    "bulk_analyze_one_exception",
                    symbol=current_symbol,
                    error=str(e),
                    traceback=traceback.format_exc(limit=50),
                )
                # Add error entry to results
                st.session_state.analysis_results.append({
                    'symbol': current_symbol,
                    'error': True,
                    'price': 0,
                    'dpo': 0,
                    'dpo_9': 0,
                    'dpo_20': 0,
                    'is_fast_lane': False,
                    'positive_dpo': False,
                    'positive_dpo_9': False,
                    'positive_dpo_20': False,
                    'in_bb_range': False
                })
            
            # Move to next stock
            st.session_state.current_analysis_index += 1
            debug_log(
                "bulk_advance_index",
                next_index=int(st.session_state.current_analysis_index),
                total=int(st.session_state.total_stocks),
            )

            # Persist partial bulk results so viewing live data always updates JSON.
            try:
                cache = load_cache()
                today_key = get_today_key()
                upsert_bulk_cache(cache, today_key, st.session_state.analysis_results)
                save_cache(cache)
            except Exception:
                pass
            
            # If more stocks to analyze, schedule next analysis
            if st.session_state.current_analysis_index < st.session_state.total_stocks:
                # Show countdown and auto-rerun
                countdown_placeholder = st.empty()
                for countdown in range(60, 0, -1):  # 1 minute cooldown between symbols
                    next_symbol = st.session_state.stock_symbols[st.session_state.current_analysis_index]
                    countdown_placeholder.info(f"⏳ Cooling down... {countdown}s - Next: {next_symbol}")
                    time.sleep(1)
                countdown_placeholder.empty()
                progress_placeholder.empty()
                debug_log(
                    "bulk_rerun",
                    next_symbol=st.session_state.stock_symbols[st.session_state.current_analysis_index],
                )
                st.rerun()
            else:
                # Analysis complete
                st.session_state.analysis_in_progress = False
                st.success("✅ Analysis complete!")
                debug_log(
                    "bulk_complete",
                    total_analyzed=len(st.session_state.analysis_results),
                    fast_lane_count=len([r for r in st.session_state.analysis_results if r.get('is_fast_lane', False)]),
                )
                
                # Save final results to cache
                try:
                    cache = load_cache()
                    today_key = get_today_key()
                    
                    bulk_results = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'results': st.session_state.analysis_results,
                        'summary': {
                            'total_analyzed': len(st.session_state.analysis_results),
                            'fast_lane_count': len([r for r in st.session_state.analysis_results if r.get('is_fast_lane', False)]),
                            'symbols_analyzed': [r['symbol'] for r in st.session_state.analysis_results if not r.get('error', False)]
                        }
                    }
                    
                    if 'bulk_analysis' not in cache:
                        cache['bulk_analysis'] = {}
                    cache['bulk_analysis'][today_key] = bulk_results
                    save_cache(cache)
                    
                    st.info(f"💾 Results saved to stock_analysis_cache.json")
                    
                except Exception as e:
                    st.warning(f"Could not save results to cache: {str(e)}")
                
                progress_placeholder.empty()
    
    except Exception as e:
        # IMPORTANT: Streamlit's st.rerun()/st.stop() can raise internal exceptions
        # to control execution flow. If we catch them here, bulk analysis will
        # stop after the first item.
        exc_name = type(e).__name__
        if exc_name in {"RerunException", "StopException"}:
            raise

        st.error(f"Bulk analysis error: {str(e)}")
        debug_log(
            "bulk_fatal_exception",
            error=str(e),
            exc_type=exc_name,
            traceback=traceback.format_exc(limit=50),
        )
        st.session_state.analysis_in_progress = False

def bulk_analyze_stocks(stock_symbols):
    """Legacy function - redirects to new live update version"""
    return bulk_analyze_stocks_with_live_updates(stock_symbols)