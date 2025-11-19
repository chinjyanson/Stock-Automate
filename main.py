"""
Main Analysis Module - Contains all analysis logic for stocks and markets
Uses Stock and Market classes as pure data providers
"""

import os
from datetime import datetime
from stock import Stock
from market import Market
from typing import Dict, List

# Ensure logs directory exists
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


# ======================= LOGGING HELPER FUNCTIONS =======================

def write_to_log(ticker: str, content: str) -> str:
    """
    Write analysis content to a log file for a specific ticker.

    Args:
        ticker: Stock ticker symbol
        content: Content to write to the log file

    Returns:
        str: Path to the log file
    """
    filename = f"{ticker}.txt"
    filepath = os.path.join(LOGS_DIR, filename)

    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def generate_flowchart_analysis_text(stock: Stock) -> str:
    """Generate flowchart analysis as text for logging."""
    analysis = analyze_stock_flowchart(stock)

    lines = []
    lines.append("=" * 70)
    lines.append(f"FLOWCHART ANALYSIS: {analysis['ticker']} - {analysis['company_name']}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("METRICS:")
    lines.append("-" * 70)
    for metric, value in analysis['metrics'].items():
        metric_display = metric.replace('_', ' ').title()
        if value is not None:
            lines.append(f"  {metric_display:25s}: {value}")
        else:
            lines.append(f"  {metric_display:25s}: N/A")
    lines.append("")

    lines.append("DECISION TREE RESULTS:")
    lines.append("-" * 70)
    for decision, passed in analysis['decisions'].items():
        decision_display = decision.replace('_', ' ').title()
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {decision_display:30s}: {status}")
    lines.append("")

    lines.append("REASONING:")
    lines.append("-" * 70)
    for reason in analysis['reasoning']:
        lines.append(f"  {reason}")
    lines.append("")

    lines.append("FINAL RECOMMENDATION:")
    lines.append("-" * 70)
    recommendation = analysis['recommendation']
    if recommendation == 'INVEST':
        lines.append(f"  {recommendation} - This stock meets all investment criteria")
    elif recommendation == 'PASS':
        lines.append(f"  {recommendation} - This stock does not meet investment criteria")
    else:
        lines.append(f"  {recommendation} - Unable to fully analyze this stock")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_comprehensive_analysis_text(stock: Stock) -> str:
    """Generate comprehensive analysis as text for logging."""
    metrics = stock.get_comprehensive_metrics()

    lines = []
    lines.append("=" * 70)
    lines.append(f"COMPREHENSIVE STOCK ANALYSIS: {stock.ticker}")
    lines.append(f"Company: {stock.info.get('longName', 'Unknown')}")
    lines.append("=" * 70)
    lines.append("")

    # Basic Metrics
    lines.append("BASIC METRICS:")
    lines.append("-" * 70)
    basic = metrics['basic_metrics']
    lines.append(f"  Current Price:            ${basic.get('current_price', 'N/A')}")
    lines.append(f"  P/E Ratio:                {basic.get('pe_ratio', 'N/A')}")
    lines.append(f"  PEG Ratio:                {basic.get('peg_ratio', 'N/A')}")
    lines.append(f"  Revenue Growth:           {basic.get('revenue_growth', 'N/A')}%")
    lines.append("")

    # Profitability
    lines.append("PROFITABILITY:")
    lines.append("-" * 70)
    prof = metrics['profitability']
    lines.append(f"  ROE:                      {prof.get('roe', 'N/A')}%")
    lines.append(f"  ROA:                      {prof.get('roa', 'N/A')}%")
    margins = prof.get('profit_margins', {})
    lines.append(f"  Gross Margin:             {margins.get('gross_margin', 'N/A')}%")
    lines.append(f"  Operating Margin:         {margins.get('operating_margin', 'N/A')}%")
    lines.append(f"  Net Margin:               {margins.get('net_margin', 'N/A')}%")
    lines.append("")

    # Liquidity & Leverage
    lines.append("LIQUIDITY & LEVERAGE:")
    lines.append("-" * 70)
    liq = metrics['liquidity']
    lev = metrics['leverage']
    lines.append(f"  Current Ratio:            {liq.get('current_ratio', 'N/A')}")
    lines.append(f"  Quick Ratio:              {liq.get('quick_ratio', 'N/A')}")
    lines.append(f"  Debt-to-Equity:           {lev.get('debt_to_equity', 'N/A')}")
    lines.append("")

    # Valuation
    lines.append("VALUATION:")
    lines.append("-" * 70)
    val = metrics['valuation']
    if val.get('market_cap'):
        lines.append(f"  Market Cap:               ${val['market_cap']:,.0f}")
    if val.get('enterprise_value'):
        lines.append(f"  Enterprise Value:         ${val['enterprise_value']:,.0f}")
    lines.append(f"  Price-to-Sales:           {val.get('price_to_sales', 'N/A')}")
    lines.append(f"  Price-to-Book:            {val.get('price_to_book', 'N/A')}")
    lines.append(f"  EV/Revenue:               {val.get('ev_to_revenue', 'N/A')}")
    lines.append(f"  EV/EBITDA:                {val.get('ev_to_ebitda', 'N/A')}")
    lines.append("")

    # Growth
    lines.append("GROWTH:")
    lines.append("-" * 70)
    growth = metrics['growth']
    lines.append(f"  Earnings Growth:          {growth.get('earnings_growth', 'N/A')}%")
    lines.append(f"  Revenue Growth (TTM):     {growth.get('revenue_growth_ttm', 'N/A')}%")
    lines.append(f"  Earnings Growth (QoQ):    {growth.get('earnings_quarterly_growth', 'N/A')}%")
    lines.append("")

    # Dividends
    lines.append("DIVIDENDS:")
    lines.append("-" * 70)
    div = metrics['dividends']
    lines.append(f"  Dividend Yield:           {div.get('dividend_yield', 'N/A')}%")
    lines.append(f"  Dividend Rate:            ${div.get('dividend_rate', 'N/A')}")
    lines.append(f"  Payout Ratio:             {div.get('payout_ratio', 'N/A')}%")
    lines.append(f"  5Y Avg Yield:             {div.get('five_year_avg_yield', 'N/A')}%")
    lines.append("")

    # Cash Flow
    lines.append("CASH FLOW:")
    lines.append("-" * 70)
    cf = metrics['cash_flow']
    if cf.get('operating_cash_flow'):
        lines.append(f"  Operating Cash Flow:      ${cf['operating_cash_flow']:,.0f}")
    if cf.get('free_cash_flow'):
        lines.append(f"  Free Cash Flow:           ${cf['free_cash_flow']:,.0f}")
    lines.append("")

    # Ownership
    lines.append("OWNERSHIP:")
    lines.append("-" * 70)
    own = metrics['ownership']
    lines.append(f"  Insider Ownership:        {own.get('insider_ownership', 'N/A')}%")
    lines.append(f"  Institutional Ownership:  {own.get('institutional_ownership', 'N/A')}%")
    lines.append("")

    # Valuation Trap Check
    trap_analysis = detect_valuation_trap(stock)
    lines.append("VALUATION TRAP ANALYSIS:")
    lines.append("-" * 70)
    if trap_analysis['is_valuation_trap']:
        lines.append("  POTENTIAL VALUATION TRAP DETECTED!")
        for reason in trap_analysis['reasons']:
            lines.append(f"    - {reason}")
    else:
        lines.append("  No obvious valuation traps detected")
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


# ======================= MARKET ANALYSIS FUNCTIONS =======================

def analyze_market_sentiment(market_metrics: Dict) -> str:
    """
    Determine overall market sentiment based on various indicators.

    Args:
        market_metrics: Dictionary containing market metrics from Market.get_all_metrics()

    Returns:
        str: Market sentiment ('BULLISH', 'BEARISH', 'NEUTRAL', or 'INSUFFICIENT_DATA')
    """
    try:
        returns = market_metrics['returns']
        technical = market_metrics['technical']

        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # Check 1-month returns
        if returns['returns_1m'] is not None:
            total_signals += 1
            if returns['returns_1m'] > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Check 3-month returns
        if returns['returns_3m'] is not None:
            total_signals += 1
            if returns['returns_3m'] > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Check moving average position
        ma_data = technical.get('moving_averages')
        if ma_data is not None:
            total_signals += 1
            if ma_data['distance_from_ma_50'] > 0 and ma_data['distance_from_ma_200'] > 0:
                bullish_signals += 1
            elif ma_data['distance_from_ma_50'] < 0 and ma_data['distance_from_ma_200'] < 0:
                bearish_signals += 1

        # Check RSI
        rsi = technical.get('rsi')
        if rsi is not None:
            total_signals += 1
            if rsi > 50:
                bullish_signals += 1
            elif rsi < 50:
                bearish_signals += 1

        if total_signals == 0:
            return "INSUFFICIENT_DATA"

        # Determine sentiment based on majority
        if bullish_signals > bearish_signals * 1.5:
            return "BULLISH"
        elif bearish_signals > bullish_signals * 1.5:
            return "BEARISH"
        else:
            return "NEUTRAL"

    except Exception as e:
        print(f"Error determining market sentiment: {e}")
        return "INSUFFICIENT_DATA"


def print_market_analysis(market: Market) -> None:
    """
    Print a formatted market analysis report.

    Args:
        market: Market instance to analyze
    """
    metrics = market.get_all_metrics()
    sentiment = analyze_market_sentiment(metrics)

    print("=" * 70)
    print(f"MARKET ANALYSIS: {metrics['ticker']} - {metrics['name']}")
    print("=" * 70)
    print()

    print("RETURNS:")
    print("-" * 70)
    returns = metrics['returns']
    for key, value in returns.items():
        period_name = key.replace('returns_', '').upper()
        if value is not None:
            direction = "↑" if value > 0 else "↓"
            print(f"  {period_name:15s}: {direction} {value:>8.2f}%")
        else:
            print(f"  {period_name:15s}: N/A")
    print()

    print("RISK METRICS:")
    print("-" * 70)
    risk = metrics['risk']
    risk_labels = {
        'volatility_1y': 'Volatility (1Y)',
        'max_drawdown_1y': 'Max Drawdown (1Y)',
        'sharpe_ratio': 'Sharpe Ratio'
    }
    for key, label in risk_labels.items():
        value = risk.get(key)
        if value is not None:
            print(f"  {label:25s}: {value:>8.2f}")
        else:
            print(f"  {label:25s}: N/A")
    print()

    print("TECHNICAL INDICATORS:")
    print("-" * 70)
    technical = metrics['technical']
    rsi = technical.get('rsi')
    if rsi is not None:
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        print(f"  RSI (14-day):             {rsi:>8.2f} ({rsi_status})")
    else:
        print(f"  RSI (14-day):             N/A")

    ma_data = technical.get('moving_averages')
    if ma_data:
        print(f"  Current Price:            ${ma_data['current_price']:>8.2f}")
        print(f"  50-day MA:                ${ma_data['ma_50']:>8.2f} ({ma_data['distance_from_ma_50']:+.2f}%)")
        print(f"  200-day MA:               ${ma_data['ma_200']:>8.2f} ({ma_data['distance_from_ma_200']:+.2f}%)")
    print()

    print("MARKET SENTIMENT:")
    print("-" * 70)
    emoji = {
        'BULLISH': '🟢',
        'BEARISH': '🔴',
        'NEUTRAL': '🟡',
        'INSUFFICIENT_DATA': '⚪'
    }.get(sentiment, '❓')
    print(f"  {emoji} {sentiment}")
    print("=" * 70)
    print()


def compare_markets(tickers: List[str]) -> None:
    """
    Compare multiple market indices.

    Args:
        tickers: List of market index tickers to compare
    """
    print("=" * 70)
    print("MARKET COMPARISON")
    print("=" * 70)
    print()

    results = []
    for ticker in tickers:
        try:
            market = Market(ticker)
            metrics = market.get_all_metrics()
            sentiment = analyze_market_sentiment(metrics)
            results.append({
                'ticker': ticker,
                'metrics': metrics,
                'sentiment': sentiment
            })
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    if not results:
        print("No market data available for comparison.")
        return

    # Print comparison table
    print(f"{'Index':<12} {'1M Return':<12} {'3M Return':<12} {'1Y Return':<12} {'Sentiment':<12}")
    print("-" * 70)

    for result in results:
        ticker = result['ticker']
        returns = result['metrics']['returns']
        sentiment = result['sentiment']

        returns_1m = returns['returns_1m']
        returns_3m = returns['returns_3m']
        returns_1y = returns['returns_1y']

        returns_1m_str = f"{returns_1m:>6.2f}%" if returns_1m is not None else "N/A"
        returns_3m_str = f"{returns_3m:>6.2f}%" if returns_3m is not None else "N/A"
        returns_1y_str = f"{returns_1y:>6.2f}%" if returns_1y is not None else "N/A"

        print(f"{ticker:<12} {returns_1m_str:<12} {returns_3m_str:<12} {returns_1y_str:<12} {sentiment:<12}")

    print("=" * 70)
    print()


def display_market_overview() -> None:
    """Display overview of major market indices."""
    print("\n" + "=" * 70)
    print("MARKET OVERVIEW")
    print("=" * 70)
    print()

    indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
    compare_markets(indices)


# ======================= STOCK ANALYSIS FUNCTIONS =======================

def analyze_stock_flowchart(stock: Stock) -> Dict:
    """
    Analyze stock using the fundamental analysis flowchart.

    Args:
        stock: Stock instance to analyze

    Returns:
        Dict: Analysis results including metrics, decisions, and recommendation
    """
    metrics = stock.get_comprehensive_metrics()

    results = {
        'ticker': stock.ticker,
        'company_name': stock.info.get('longName', 'Unknown'),
        'metrics': {},
        'decisions': {},
        'recommendation': None,
        'reasoning': []
    }

    # Step 1: Revenue Growth
    revenue_growth = metrics['basic_metrics']['revenue_growth']
    results['metrics']['revenue_growth'] = revenue_growth

    if revenue_growth is None:
        results['recommendation'] = 'INSUFFICIENT_DATA'
        results['reasoning'].append('Unable to determine revenue growth')
        return results

    results['decisions']['revenue_growing_10%'] = revenue_growth >= 10

    if revenue_growth < 10:
        results['recommendation'] = 'PASS'
        results['reasoning'].append(f'Low revenue growth: {revenue_growth}% (need ≥10%)')
        return results

    results['reasoning'].append(f'✓ Revenue growing at {revenue_growth}% per year')

    # Step 2: P/E Ratio
    pe_ratio = metrics['basic_metrics']['pe_ratio']
    results['metrics']['pe_ratio'] = pe_ratio

    if pe_ratio is None:
        results['recommendation'] = 'INSUFFICIENT_DATA'
        results['reasoning'].append('Unable to determine P/E ratio')
        return results

    results['decisions']['pe_below_25'] = pe_ratio < 25

    if pe_ratio >= 25:
        results['recommendation'] = 'PASS'
        results['reasoning'].append(f'Likely overvalued: P/E ratio {pe_ratio} (need <25)')
        return results

    results['reasoning'].append(f'✓ Reasonable valuation: P/E ratio {pe_ratio}')

    # Step 3: PEG Ratio
    peg_ratio = metrics['basic_metrics']['peg_ratio']
    results['metrics']['peg_ratio'] = peg_ratio

    if peg_ratio is None:
        results['recommendation'] = 'INSUFFICIENT_DATA'
        results['reasoning'].append('Unable to determine PEG ratio')
        return results

    results['decisions']['peg_below_2'] = peg_ratio < 2

    if peg_ratio >= 2:
        results['recommendation'] = 'PASS'
        results['reasoning'].append(f'Low profit growth: PEG ratio {peg_ratio} (need <2)')
        return results

    results['reasoning'].append(f'✓ Good growth relative to price: PEG ratio {peg_ratio}')

    # Step 4: ROE
    roe = metrics['profitability']['roe']
    results['metrics']['roe'] = roe

    if roe is None:
        results['recommendation'] = 'INSUFFICIENT_DATA'
        results['reasoning'].append('Unable to determine ROE')
        return results

    results['decisions']['roe_above_5%'] = roe > 5

    if roe <= 5:
        results['recommendation'] = 'PASS'
        results['reasoning'].append(f'Weak profitability: ROE {roe}% (need >5%)')
        return results

    results['reasoning'].append(f'✓ Strong profitability: ROE {roe}%')

    # Step 5: Quick Ratio
    quick_ratio = metrics['liquidity']['quick_ratio']
    results['metrics']['quick_ratio'] = quick_ratio

    if quick_ratio is None:
        results['recommendation'] = 'INSUFFICIENT_DATA'
        results['reasoning'].append('Unable to determine quick ratio')
        return results

    results['decisions']['quick_ratio_above_1.5'] = quick_ratio > 1.5

    if quick_ratio <= 1.5:
        results['recommendation'] = 'PASS'
        results['reasoning'].append(f'Liquidity issues: Quick ratio {quick_ratio} (need >1.5)')
        return results

    results['reasoning'].append(f'✓ Good liquidity: Quick ratio {quick_ratio}')

    # All checks passed!
    results['recommendation'] = 'INVEST'
    results['reasoning'].append('Stock passes all criteria - Worth considering for investment!')

    return results


def detect_valuation_trap(stock: Stock) -> Dict:
    """
    Detect potential valuation traps in a stock.

    Args:
        stock: Stock instance to analyze

    Returns:
        Dict: Valuation trap analysis results
    """
    metrics = stock.get_comprehensive_metrics()

    is_trap = False
    reasons = []

    # Check forward vs trailing P/E
    pe_ratio = metrics['basic_metrics']['pe_ratio']
    forward_pe = stock.info.get('forwardPE')

    if pe_ratio is not None and forward_pe is not None:
        if pe_ratio < 10 and forward_pe > pe_ratio:
            is_trap = True
            reasons.append("Low current P/E but higher forward P/E (declining earnings)")

    # Check earnings growth
    earnings_growth = metrics['growth'].get('earnings_growth')
    if earnings_growth is not None and earnings_growth < 0:
        is_trap = True
        reasons.append("Negative earnings growth")

    # Check revenue growth
    revenue_growth_ttm = metrics['growth'].get('revenue_growth_ttm')
    if revenue_growth_ttm is not None and revenue_growth_ttm < 0:
        is_trap = True
        reasons.append("Negative revenue growth, indicating potential demand decline")

    # Check debt levels
    debt_to_equity = metrics['leverage'].get('debt_to_equity')
    if debt_to_equity is not None and debt_to_equity > 200:
        is_trap = True
        reasons.append(f"High debt-to-equity ratio ({debt_to_equity}), indicating financial risk")

    # Check profitability
    roe = metrics['profitability']['roe']
    if roe is not None and roe < 5:
        is_trap = True
        reasons.append(f"Low return on equity (ROE {roe}%), indicating weak profitability")

    # Check vs industry (if available)
    industry_pe = stock.info.get('industryPE')
    if industry_pe is not None and pe_ratio is not None and pe_ratio < industry_pe * 0.5:
        is_trap = True
        reasons.append("Stock significantly cheaper than industry peers (possible hidden risks)")

    return {
        'is_valuation_trap': is_trap,
        'reasons': reasons
    }


def print_stock_flowchart_analysis(stock: Stock) -> None:
    """
    Print formatted flowchart analysis for a stock.

    Args:
        stock: Stock instance to analyze
    """
    analysis = analyze_stock_flowchart(stock)

    print("=" * 70)
    print(f"FLOWCHART ANALYSIS: {analysis['ticker']} - {analysis['company_name']}")
    print("=" * 70)
    print()

    print("METRICS:")
    print("-" * 70)
    for metric, value in analysis['metrics'].items():
        metric_display = metric.replace('_', ' ').title()
        if value is not None:
            print(f"  {metric_display:25s}: {value}")
        else:
            print(f"  {metric_display:25s}: N/A")
    print()

    print("DECISION TREE RESULTS:")
    print("-" * 70)
    for decision, passed in analysis['decisions'].items():
        decision_display = decision.replace('_', ' ').title()
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {decision_display:30s}: {status}")
    print()

    print("REASONING:")
    print("-" * 70)
    for reason in analysis['reasoning']:
        print(f"  {reason}")
    print()

    print("FINAL RECOMMENDATION:")
    print("-" * 70)
    recommendation = analysis['recommendation']
    if recommendation == 'INVEST':
        print(f"  🟢 {recommendation} - This stock meets all investment criteria")
    elif recommendation == 'PASS':
        print(f"  🔴 {recommendation} - This stock does not meet investment criteria")
    else:
        print(f"  🟡 {recommendation} - Unable to fully analyze this stock")
    print("=" * 70)
    print()


def print_comprehensive_stock_analysis(stock: Stock) -> None:
    """
    Print comprehensive analysis with all accounting metrics.

    Args:
        stock: Stock instance to analyze
    """
    metrics = stock.get_comprehensive_metrics()

    print("=" * 70)
    print(f"COMPREHENSIVE STOCK ANALYSIS: {stock.ticker}")
    print(f"Company: {stock.info.get('longName', 'Unknown')}")
    print("=" * 70)
    print()

    # Basic Metrics
    print("BASIC METRICS:")
    print("-" * 70)
    basic = metrics['basic_metrics']
    print(f"  Current Price:            ${basic.get('current_price', 'N/A')}")
    print(f"  P/E Ratio:                {basic.get('pe_ratio', 'N/A')}")
    print(f"  PEG Ratio:                {basic.get('peg_ratio', 'N/A')}")
    print(f"  Revenue Growth:           {basic.get('revenue_growth', 'N/A')}%")
    print()

    # Profitability
    print("PROFITABILITY:")
    print("-" * 70)
    prof = metrics['profitability']
    print(f"  ROE:                      {prof.get('roe', 'N/A')}%")
    print(f"  ROA:                      {prof.get('roa', 'N/A')}%")
    margins = prof.get('profit_margins', {})
    print(f"  Gross Margin:             {margins.get('gross_margin', 'N/A')}%")
    print(f"  Operating Margin:         {margins.get('operating_margin', 'N/A')}%")
    print(f"  Net Margin:               {margins.get('net_margin', 'N/A')}%")
    print()

    # Liquidity & Leverage
    print("LIQUIDITY & LEVERAGE:")
    print("-" * 70)
    liq = metrics['liquidity']
    lev = metrics['leverage']
    print(f"  Current Ratio:            {liq.get('current_ratio', 'N/A')}")
    print(f"  Quick Ratio:              {liq.get('quick_ratio', 'N/A')}")
    print(f"  Debt-to-Equity:           {lev.get('debt_to_equity', 'N/A')}")
    print()

    # Valuation
    print("VALUATION:")
    print("-" * 70)
    val = metrics['valuation']
    if val.get('market_cap'):
        print(f"  Market Cap:               ${val['market_cap']:,.0f}")
    if val.get('enterprise_value'):
        print(f"  Enterprise Value:         ${val['enterprise_value']:,.0f}")
    print(f"  Price-to-Sales:           {val.get('price_to_sales', 'N/A')}")
    print(f"  Price-to-Book:            {val.get('price_to_book', 'N/A')}")
    print(f"  EV/Revenue:               {val.get('ev_to_revenue', 'N/A')}")
    print(f"  EV/EBITDA:                {val.get('ev_to_ebitda', 'N/A')}")
    print()

    # Growth
    print("GROWTH:")
    print("-" * 70)
    growth = metrics['growth']
    print(f"  Earnings Growth:          {growth.get('earnings_growth', 'N/A')}%")
    print(f"  Revenue Growth (TTM):     {growth.get('revenue_growth_ttm', 'N/A')}%")
    print(f"  Earnings Growth (QoQ):    {growth.get('earnings_quarterly_growth', 'N/A')}%")
    print()

    # Dividends
    print("DIVIDENDS:")
    print("-" * 70)
    div = metrics['dividends']
    print(f"  Dividend Yield:           {div.get('dividend_yield', 'N/A')}%")
    print(f"  Dividend Rate:            ${div.get('dividend_rate', 'N/A')}")
    print(f"  Payout Ratio:             {div.get('payout_ratio', 'N/A')}%")
    print(f"  5Y Avg Yield:             {div.get('five_year_avg_yield', 'N/A')}%")
    print()

    # Cash Flow
    print("CASH FLOW:")
    print("-" * 70)
    cf = metrics['cash_flow']
    if cf.get('operating_cash_flow'):
        print(f"  Operating Cash Flow:      ${cf['operating_cash_flow']:,.0f}")
    if cf.get('free_cash_flow'):
        print(f"  Free Cash Flow:           ${cf['free_cash_flow']:,.0f}")
    print()

    # Ownership
    print("OWNERSHIP:")
    print("-" * 70)
    own = metrics['ownership']
    print(f"  Insider Ownership:        {own.get('insider_ownership', 'N/A')}%")
    print(f"  Institutional Ownership:  {own.get('institutional_ownership', 'N/A')}%")
    print()

    # Valuation Trap Check
    trap_analysis = detect_valuation_trap(stock)
    print("VALUATION TRAP ANALYSIS:")
    print("-" * 70)
    if trap_analysis['is_valuation_trap']:
        print("  ⚠️  POTENTIAL VALUATION TRAP DETECTED!")
        for reason in trap_analysis['reasons']:
            print(f"    - {reason}")
    else:
        print("  ✅ No obvious valuation traps detected")
    print()

    print("=" * 70)
    print()


def analyze_multiple_stocks(tickers: List[str]) -> None:
    """
    Analyze multiple stocks using the flowchart method.
    Writes detailed analysis to log files and prints summary to terminal.

    Args:
        tickers: List of stock ticker symbols
    """
    results_summary = []
    total = len(tickers)

    print(f"\nAnalyzing {total} stocks (logs will be saved to {LOGS_DIR})...\n")

    for i, ticker in enumerate(tickers, 1):
        try:
            stock = Stock(ticker)
            analysis = analyze_stock_flowchart(stock)

            # Generate log content
            log_content = []
            log_content.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_content.append("")
            log_content.append(generate_flowchart_analysis_text(stock))
            log_content.append("")
            log_content.append(generate_comprehensive_analysis_text(stock))

            # Write to log file
            log_path = write_to_log(ticker, "\n".join(log_content))

            results_summary.append({
                'ticker': ticker,
                'recommendation': analysis['recommendation'],
                'log_path': log_path
            })

            # Print progress
            status_icon = {
                'INVEST': '🟢',
                'PASS': '🔴',
                'INSUFFICIENT_DATA': '🟡'
            }.get(analysis['recommendation'], '❓')
            print(f"  [{i}/{total}] {status_icon} {ticker:10s}: {analysis['recommendation']}")

        except Exception as e:
            print(f"  [{i}/{total}] ⚠️  {ticker:10s}: ERROR - {e}")
            results_summary.append({
                'ticker': ticker,
                'recommendation': 'ERROR',
                'log_path': None
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL ANALYZED STOCKS")
    print("=" * 70)

    # Group by recommendation
    invest = [r for r in results_summary if r['recommendation'] == 'INVEST']
    pass_stocks = [r for r in results_summary if r['recommendation'] == 'PASS']
    insufficient = [r for r in results_summary if r['recommendation'] == 'INSUFFICIENT_DATA']
    errors = [r for r in results_summary if r['recommendation'] == 'ERROR']

    print(f"\n  🟢 INVEST ({len(invest)}):")
    for r in invest:
        print(f"      {r['ticker']}")

    print(f"\n  🔴 PASS ({len(pass_stocks)}):")
    for r in pass_stocks:
        print(f"      {r['ticker']}")

    if insufficient:
        print(f"\n  🟡 INSUFFICIENT DATA ({len(insufficient)}):")
        for r in insufficient:
            print(f"      {r['ticker']}")

    if errors:
        print(f"\n  ⚠️  ERRORS ({len(errors)}):")
        for r in errors:
            print(f"      {r['ticker']}")

    print(f"\n  Logs saved to: {LOGS_DIR}")
    print("=" * 70)


def show_fundamental_analysis_flowchart() -> None:
    """Display the fundamental analysis flowchart."""
    print("\n" + "=" * 70)
    print("FUNDAMENTAL ANALYSIS FLOWCHART")
    print("=" * 70)
    print("""
    1. Revenue Growth ≥ 10%?
       ├─ No  → PASS (Insufficient growth)
       └─ Yes → Continue

    2. P/E Ratio < 25?
       ├─ No  → PASS (Possibly overvalued)
       └─ Yes → Continue

    3. PEG Ratio < 2?
       ├─ No  → PASS (Low profit growth)
       └─ Yes → Continue

    4. ROE > 5%?
       ├─ No  → PASS (Weak profitability)
       └─ Yes → Continue

    5. Quick Ratio > 1.5?
       ├─ No  → PASS (Liquidity issues)
       └─ Yes → INVEST (Passes all criteria!)
    """)
    print("=" * 70)


# ======================= MAIN FUNCTION =======================

def analyse():
    """Main function to run stock analysis."""
    print("\n" + "=" * 70)
    print("STOCK & MARKET ANALYSIS TOOL")
    print("=" * 70)

    # Display market overview
    display_market_overview()

    # Show fundamental analysis flowchart
    show_fundamental_analysis_flowchart()

    # Analyze multiple stocks - detailed logs saved to files
    tickers_to_analyze = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "CRM", "AMD", "ORCL",
        "TSM", "INTC", "ASML", "MU", "TXN",
        "XOM", "CVX", "BP", "SHEL", "NEP", "ENPH", "FSLR",
        "JPM", "BAC", "WFC", "V", "MA", "BLK",
        "PG", "KO", "PEP", "COST", "WMT",
        "TSLA", "HD", "MCD", "NKE", "SBUX",
        "JNJ", "PFE", "MRK", "UNH", "ABBV",
        "VZ", "T", "TMUS",
        "CAT", "BA", "GE", "DE",
        "NEE", "DUK", "SO",
        "AMT", "PLD", "O",
        "SPY", "VOO", "QQQ", "VTI", "VXUS"
    ]
    analyze_multiple_stocks(tickers_to_analyze)


if __name__ == "__main__":
    analyse()
