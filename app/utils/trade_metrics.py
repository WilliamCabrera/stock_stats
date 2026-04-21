import os
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.utils.massive import fetch_candles, _raw_data_to_dataframe

pd.options.display.float_format = '{:.3f}'.format



# ============================================================
# BENCHMARKS
# ============================================================
def buy_and_hold_benchmark(ticker, _from, _to, initial_capital):
    
    #data = utils.fetch_ticker_data_daily(ticker=ticker,start_date=_from, end_date=_to)
    data = fetch_candles(ticker=ticker,from_date=_from, to_date=_to)
    df =  _raw_data_to_dataframe(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    prices = df['close']
    first_price = df['open']

    equity = initial_capital * (prices / first_price.iloc[0])
    returns = prices.pct_change()
    
    result = pd.DataFrame(
        {
            "equity": equity,
            "returns": returns,
        },
        index=df.index,
    )
    result['equity'] = result['equity'].round(2)
 
    return result
    


# ============================================================
# EQUITY DESDE R (% FIJO DE LA CUENTA)
# ============================================================

def equity_from_r(trades: pd.DataFrame, initial_capital: float, risk_pct: float) -> pd.DataFrame:
    trades = trades.copy()

    # Asegurarse de que entry y exit sean datetime
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    trades = trades.sort_values("exit_time")
    trades["R"] = r_multiple(trades)

    capital = initial_capital
    equity = []

    for r in trades["R"]:
        pnl = capital * risk_pct * r
        capital += pnl
        equity.append(capital)

    equity_df = pd.DataFrame({
        "equity": equity,
        "entry_time": trades["entry_time"].values
    }, index=trades["exit_time"])

    # Asegurarse de que el índice sea datetime
    equity_df.index = pd.to_datetime(equity_df.index)

    return equity_df


def equity_returns(equity: pd.Series) -> pd.Series:
    returns = equity.pct_change().dropna()
    returns.name = "returns"
    return returns


def cumulative_returns(equity: pd.Series) -> float:
    return equity.iloc[-1] / equity.iloc[0] - 1


def annual_return(
    equity: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> float:
    years = (end_date - start_date).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annual_volatility(returns: pd.Series, freq: int = 252) -> float:
    return returns.std() * np.sqrt(freq)


# ============================================================
# DRAWDOWN
# ============================================================

def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_series(equity).min()


# ============================================================
# RATIOS
# ============================================================

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: int = 252
) -> float:
    excess = returns - risk_free_rate / freq
    return excess.mean() / excess.std() * np.sqrt(freq)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: int = 252
) -> float:
    downside = returns[returns < 0]
    return (returns.mean() - risk_free_rate / freq) / downside.std() * np.sqrt(freq)


def calmar_ratio(annual_ret: float, max_dd: float) -> float:
    return annual_ret / abs(max_dd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns < threshold]).sum()
    return gains / losses


def tail_ratio(returns: pd.Series, q: float = 0.95) -> float:
    right = returns.quantile(q)
    left = abs(returns.quantile(1 - q))
    return right / left


def stability(equity: pd.Series) -> float:
    log_eq = np.log(equity)
    x = np.arange(len(log_eq))
    slope, intercept = np.polyfit(x, log_eq, 1)
    fitted = slope * x + intercept
    ss_res = ((log_eq - fitted) ** 2).sum()
    ss_tot = ((log_eq - log_eq.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# ============================================================
# DISTRIBUTION
# ============================================================

def skewness(returns: pd.Series) -> float:
    return returns.skew()


def kurtosis(returns: pd.Series) -> float:
    return returns.kurtosis()


def daily_var(returns: pd.Series, level: float = 0.05) -> float:
    return returns.quantile(level)


# ============================================================
# TRADE STATS (EN R)
# ============================================================

def win_rate(trades: pd.DataFrame) -> float:
    return trades["is_profit"].mean()


def expectancy_r(trades: pd.DataFrame) -> float:
    return r_multiple(trades).mean()


def avg_r_win(trades: pd.DataFrame) -> float:
    r = r_multiple(trades)
    return r[r > 0].mean()



# ============================================================
# R MULTIPLE (EDGE PURO)
# ============================================================

def r_multiple(trades: pd.DataFrame) -> pd.Series:
    """
    Calcula el R-multiple por trade.
    Long : (exit - entry) / (entry - stop)
    Short: (entry - exit) / (stop - entry)
    """
    r = np.where(
        trades["type"].astype(str).str.lower()  == "long",
        (trades["exit_price"] - trades["entry_price"]) /
        (trades["entry_price"] - trades["stop_loss_price"]),
        (trades["entry_price"] - trades["exit_price"]) /
        (trades["stop_loss_price"] - trades["entry_price"])
    )
    
    inf_mask = np.isinf(r)  
    
    res =  pd.Series(r, index=trades.index, name="R").round(2)

    return  res


# ============================================================
# EQUITY DESDE R (% FIJO DE LA CUENTA)
# ============================================================


def equity_from_r(trades: pd.DataFrame, initial_capital: float, risk_pct: float) -> pd.DataFrame:
    """
    Construye la curva de equity usando R y un % fijo de la cuenta por trade.
    Devuelve un DataFrame con índice exit_time y columnas equity + entry_time
    risk_pct: entre 0 y 1, ex: 50% = 0.5
    """
    trades = trades.copy().sort_values("exit_time")
    trades["R"] = r_multiple(trades)
    #print("******** equity_from_r ********")
    #print(trades[['entry_time','entry_price','exit_price','stop_loss_price','R']])
    

    capital = initial_capital
    equity = []

    for r in trades["R"]:
        pnl = capital * risk_pct * r
        capital += pnl
        equity.append(capital)
        #print(f' === {pnl}, {capital}, {risk_pct}, {r}')
        

    equity_df = pd.DataFrame({
        "equity": equity,
        "entry_time": trades["entry_time"].values
    }, index=trades["exit_time"])
    

 

    return equity_df


def equity_returns(equity: pd.Series) -> pd.Series:
    returns = equity.pct_change().dropna()
    returns.name = "returns"
    return returns


def cumulative_returns(equity: pd.Series) -> float:
    return equity.iloc[-1] / equity.iloc[0] - 1


def annual_return(equity: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    
    years = (end_date - start_date).days / 365.25
    print('********* annual_return ********')
    #print(years)
    #print(equity.iloc[-1] / equity.iloc[0])
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annual_volatility(returns: pd.Series, freq: int = 252) -> float:
    return returns.std() * np.sqrt(freq)

def win_loss_streak(trades):
    
    trades['is_profit'] = trades['pnl'] > 0
    s = trades['is_profit']

    # Identifica cambios True ↔ False
    groups = (s != s.shift()).cumsum()
    
    streaks = s.groupby(groups).agg(
    value='first',
    length='size'
    )
    
    max_positive_streak = streaks.loc[streaks['value'] == True, 'length'].max()
    max_negative_streak = streaks.loc[streaks['value'] == False, 'length'].max()
    
    return max_positive_streak, max_negative_streak
# ============================================================
# DRAWDOWN
# ============================================================

def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_series(equity).min()


# ============================================================
# RATIOS
# ============================================================

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, freq: int = 252) -> float:
    excess = returns - risk_free_rate / freq
    return excess.mean() / excess.std() * np.sqrt(freq)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, freq: int = 252) -> float:
    downside = returns[returns < 0]
    return (returns.mean() - risk_free_rate / freq) / downside.std() * np.sqrt(freq)


def calmar_ratio(annual_ret: float, max_dd: float) -> float:
    return annual_ret / abs(max_dd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns < threshold]).sum()
    return gains / losses


def tail_ratio(returns: pd.Series, q: float = 0.95) -> float:
    right = returns.quantile(q)
    left = abs(returns.quantile(1 - q))
    return right / left


def stability(equity: pd.Series) -> float:
    log_eq = np.log(equity)
    x = np.arange(len(log_eq))
    slope, intercept = np.polyfit(x, log_eq, 1)
    fitted = slope * x + intercept
    ss_res = ((log_eq - fitted) ** 2).sum()
    ss_tot = ((log_eq - log_eq.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# ============================================================
# DISTRIBUTION
# ============================================================

def skewness(returns: pd.Series) -> float:
    return returns.skew()


def kurtosis(returns: pd.Series) -> float:
    return returns.kurtosis()


def daily_var(returns: pd.Series, level: float = 0.05) -> float:
    return returns.quantile(level)

def returns_distribution(returns):
    
    if not isinstance(returns, np.ndarray):
        returns = returns.to_numpy()
        
    mean = returns.mean()
    median = np.median(returns)
    p1, p5, p50, p95, p99 = np.percentile(
    returns, [1, 5, 50, 95, 99]
    )

  

    plt.figure(figsize=(10,6))
    plt.hist(returns, bins=50, density=True)

    # Dibujar líneas
    plt.axvline(p1)
    plt.axvline(p5)
    plt.axvline(p50)
    plt.axvline(p95)
    plt.axvline(p99)
    plt.axvline(mean)
    plt.axvline(median)

    plt.title("Distribución de Returns con Percentiles")
    plt.show()
    
    return
# ============================================================
# TRADE STATS (EN R)
# ============================================================

def win_rate(trades: pd.DataFrame) -> float:
    return trades["is_profit"].mean()


def expectancy_r(trades: pd.DataFrame) -> float:
    return r_multiple(trades).mean()


def avg_r_win(trades: pd.DataFrame) -> float:
    r = r_multiple(trades)
    return r[r > 0].mean()


def avg_r_loss(trades: pd.DataFrame) -> float:
    r = r_multiple(trades)
    return r[r < 0].mean()


def profit_factor(trades: pd.DataFrame) -> float:
    r = r_multiple(trades)
    
    wins = r[r > 0]
    losses = r[r <= 0]
    
    return round(wins.sum()/losses.abs().sum(),2)

def general_stas_in_R(trades):
    
    trades = trades.copy()
    if "is_profit" not in trades.columns:
        trades["is_profit"] = trades["pnl"] > 0
    
    r = r_multiple(trades)
    
    wins = r[r > 0]
    losses = r[r <= 0]
    
    s = trades['is_profit']

    # Identifica cambios True ↔ False
    groups = (s != s.shift()).cumsum()
    
    streaks = s.groupby(groups).agg(
    value='first',
    length='size'
    )
    
    
    
    expectancy_r =  r.mean()
    avg_r_win = r[r > 0].mean()
    avg_r_loss = r[r <= 0].mean()
    win_rate = trades["is_profit"].mean()
    profit_factor = round(wins.sum()/losses.abs().sum(),2)
    total_profits_R =  wins.sum()
    total_loss_R =  losses.sum()
    max_positive_streak = streaks.loc[streaks['value'] == True, 'length'].max()
    max_negative_streak = streaks.loc[streaks['value'] == False, 'length'].max()
    
    
    return expectancy_r, avg_r_win, avg_r_loss, win_rate, profit_factor, total_profits_R, total_loss_R, max_positive_streak

# ============================================================
# MARKET RELATION
# ============================================================

def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, freq: int = 252):
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    beta = cov / df.iloc[:, 1].var()
    alpha = (df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * freq
    return alpha, beta

# ============================================================
# SUMMARY REPORT (R + EQUITY REALISTA)
# ============================================================

def summary_report(trades: pd.DataFrame, initial_capital: float, risk_pct: float, benchmark_returns: pd.Series | None = None) -> pd.Series:
    equity_df = equity_from_r(trades, initial_capital, risk_pct)
    equity = equity_df["equity"]  # Tomar solo la columna

    returns = equity_returns(equity)
    ann_ret = annual_return(equity, trades.entry_time.min(), trades.exit_time.max())
    max_dd = max_drawdown(equity)
    
    trades = trades.copy()
    if "is_profit" not in trades.columns:
        trades["is_profit"] = trades["pnl"] > 0
    
    r = r_multiple(trades)
    
    #print("******** r **********")
    #print(r)
    
    wins = r[r > 0]
    losses = r[r <= 0]
    
    s = trades['is_profit']

    # Identifica cambios True ↔ False
    groups = (s != s.shift()).cumsum()
    
    streaks = s.groupby(groups).agg(
    value='first',
    length='size'
    )
    # ===== general stas =======
    expectancy_r =  r.mean()
    avg_r_win = r[r > 0].mean()
    avg_r_loss = r[r <= 0].mean()
    win_rate = trades["is_profit"].mean()
    profit_factor = round(wins.sum()/losses.abs().sum(),2)
    total_profits_R =  wins.sum()
    total_loss_R =  losses.sum()
    max_positive_streak = streaks.loc[streaks['value'] == True, 'length'].max()
    max_negative_streak = streaks.loc[streaks['value'] == False, 'length'].max()
    
    # =========================
  
    

    report = {
        "Initial capital": initial_capital,
        "Risk per R (%)": risk_pct * 100,
        "Final equity": equity.iloc[-1],
        "Cumulative return": cumulative_returns(equity),
        "Annual return": ann_ret,
        "Annual volatility": annual_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns),
        "Sortino ratio": sortino_ratio(returns),
        "Calmar ratio": calmar_ratio(ann_ret, max_dd),
        "Stability": stability(equity),
        "Max drawdown": max_dd,
        "Omega ratio": omega_ratio(returns),
        "Tail ratio": tail_ratio(returns),
        "Skew": skewness(returns),
        "Kurtosis": kurtosis(returns),
        "Daily VaR": daily_var(returns),
        "Win rate": win_rate,
        "profit_factor": profit_factor,
        "Expectancy (R)": expectancy_r,
        "Avg R win": avg_r_win,
        "Avg R loss": avg_r_loss,
        "max_positive_streak": max_positive_streak,
        "max_negative_streak": max_negative_streak,
    }

    if benchmark_returns is not None:
        alpha, beta = alpha_beta(returns, benchmark_returns)
        report["Alpha"] = alpha
        report["Beta"] = beta
        
   
    return pd.Series(report)


# ============================================================
# PLOTTING
# ============================================================

def analysis_and_plot(trades: pd.DataFrame, initial_capital: float, risk_pct: float):
   

    # ============================================================
    # Construir equity y comprimir a diario
    # ============================================================
    equity_df = equity_from_r(trades, initial_capital=initial_capital, risk_pct=risk_pct)
    equity_df.index = pd.to_datetime(equity_df.index)
    equity = equity_df["equity"].resample("D").last().dropna()

    # ============================================================
    # Resumen
    # ============================================================
    report = summary_report(trades, initial_capital=initial_capital, risk_pct=risk_pct)
    print(report)
    # ============================================================
    # Calcular métricas
    # ============================================================
    returns = equity_returns(equity)
    drawdowns = drawdown_series(equity)
    rolling_vol = returns.rolling(20).std() * (252**0.5)

    # CAGR
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    equity_cagr = equity.iloc[0] * (1 + cagr) ** np.linspace(0, years, len(equity))

    # Daily VaR 5%
    daily_var_5 = returns.quantile(0.05)

    # ============================================================
    # Graficar dashboard
    # ============================================================
    # ============================================================
    # Crear layout 2x2 con gridspec
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ----------------------------
    # 1️⃣ Equity Curve
    # ----------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(equity.index, equity, color='blue', label='Equity')
    ax0.plot(equity.index, equity_cagr, color='green', linestyle='--', label=f'CAGR {cagr*100:.2f}%')
    ax0.fill_between(equity.index, equity*(1+drawdowns), equity, color='red', alpha=0.2, label='Drawdown')
    ax0.set_ylabel('Capital')
    ax0.set_title('Equity Curve con CAGR y Drawdowns')
    ax0.legend()
    ax0.grid(True)

    # ----------------------------
    # 2️⃣ Drawdowns
    # ----------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(drawdowns.index, drawdowns, color='red', label='Drawdown')
    ax1.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax1.set_ylabel('Drawdown')
    ax1.set_title('Drawdowns')
    ax1.grid(True)
    ax1.legend()

    # ----------------------------
    # 3️⃣ Volatilidad Rolling
    # ----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rolling_vol.index, rolling_vol, color='orange', label='Volatilidad Rolling 20 días')
    ax2.set_ylabel('Volatilidad')
    ax2.set_title('Volatilidad Rolling (anualizada)')
    ax2.grid(True)
    ax2.legend()

    # ----------------------------
    # 4️⃣ Histograma de Retornos con VaR
    # ----------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns, bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax3.axvline(daily_var_5, color='red', linestyle='--', label=f'VaR 5%: {daily_var_5:.2%}')
    ax3.set_xlabel('Retorno Diario')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Retornos Diarios')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close()
    webbrowser.open(f"file://{tmp.name}")
    return


def analysis_and_plot_with_benchmark(
    trades: pd.DataFrame,
    initial_capital: float,
    risk_pct: float,
    benchmark_ticker = "SPY"
):
    
    trades_copy =  trades.copy()
    trades_copy = trades_copy.sort_values("entry_time")
    first_entry = trades_copy.iloc[0]['entry_time'].strftime('%Y-%m-%d') 
    trades_copy = trades_copy.sort_values("exit_time")
    last_exit = trades_copy.iloc[-1]['exit_time'].strftime('%Y-%m-%d') 
    benchmark = buy_and_hold_benchmark(benchmark_ticker,first_entry, last_exit, 100000)
    
    
    # ============================================================
    # Construir equity estrategia
    # ============================================================
    eq_df = equity_from_r(trades, initial_capital=initial_capital, risk_pct=risk_pct)
    eq_df.index.name = "entry_time"
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["equity"]
    returns = equity_returns(equity)
    drawdowns = drawdown_series(equity)
    rolling_vol = returns.rolling(20).std() * (252**0.5)
    daily_var_5 = returns.quantile(0.05)
    years_total = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years_total) - 1
    equity_cagr = equity.iloc[0] * ((equity.iloc[-1] / equity.iloc[0]) ** ((equity.index - equity.index[0]).days / 365.25))

    # ============================================================
    # Construir equity benchmark
    # ============================================================
    benchmark_eq = benchmark["equity"]
    benchmark_returns = benchmark_eq.pct_change().dropna()
    benchmark_dd = drawdown_series(benchmark_eq)
    benchmark_vol = benchmark_returns.rolling(20).std() * (252**0.5)
    benchmark_var5 = benchmark_returns.quantile(0.05)
    years_total_b = (benchmark_eq.index[-1] - benchmark_eq.index[0]).days / 365.25
    benchmark_cagr = benchmark_eq.iloc[0] * ((benchmark_eq.iloc[-1] / benchmark_eq.iloc[0]) ** ((benchmark_eq.index - benchmark_eq.index[0]).days / 365.25))

    # ============================================================
    # Número de gráficos y layout
    # ============================================================
    n_graphs = 4  # equity, drawdowns, vol, histograma
    fig, axes = plt.subplots(
        n_graphs, 2, figsize=(18, 4*n_graphs), sharex=False
    )

    # ============================================================
    # Columna izquierda: Estrategia
    # ============================================================
    # 1️⃣ Equity
    axes[0,0].plot(equity.index, equity, color='blue', label='Equity')
    axes[0,0].plot(equity.index, equity_cagr, color='green', linestyle='--', label=f'CAGR {cagr*100:.2f}%')
    #axes[0,0].fill_between(equity.index, equity*(1+drawdowns), equity, color='red', alpha=0.2, label='Drawdown')
    axes[0,0].set_title('Estrategia: Equity')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # 2️⃣ Drawdowns
    axes[1,0].plot(drawdowns.index, drawdowns, color='red', label='Drawdown')
    axes[1,0].fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    axes[1,0].set_title('Estrategia: Drawdowns')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # 3️⃣ Volatilidad Rolling
    axes[2,0].plot(rolling_vol.index, rolling_vol, color='orange', label='Volatilidad Rolling 20 días')
    axes[2,0].set_title('Estrategia: Volatilidad Rolling')
    axes[2,0].legend()
    axes[2,0].grid(True)

    # 4️⃣ Histograma Retornos
    # axes[3,0].hist(returns, bins=50, color='gray', edgecolor='black', alpha=0.7)
    # axes[3,0].axvline(daily_var_5, color='red', linestyle='--', label=f'VaR 5%: {daily_var_5:.2%}')
    # axes[3,0].set_title('Estrategia: Distribución de Retornos')
    # axes[3,0].legend()
    # axes[3,0].grid(True)

    # ============================================================
    # Columna derecha: Benchmark
    # ============================================================
    # 1️⃣ Equity
    axes[0,1].plot(benchmark_eq.index, benchmark_eq, color='blue', label='Benchmark Equity')
    #axes[0,1].plot(benchmark_eq.index, benchmark_cagr, color='green', linestyle='--', label='CAGR')
    axes[0,1].fill_between(benchmark_eq.index, benchmark_eq*(1+benchmark_dd), benchmark_eq, color='red', alpha=0.2, label='Drawdown')
    axes[0,1].set_title('Benchmark: Equity')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # 2️⃣ Drawdowns
    axes[1,1].plot(benchmark_dd.index, benchmark_dd, color='red', label='Drawdown')
    axes[1,1].fill_between(benchmark_dd.index, benchmark_dd, 0, color='red', alpha=0.3)
    axes[1,1].set_title('Benchmark: Drawdowns')
    axes[1,1].legend()
    axes[1,1].grid(True)

    # 3️⃣ Volatilidad Rolling
    axes[2,1].plot(benchmark_vol.index, benchmark_vol, color='orange', label='Volatilidad Rolling 20 días')
    axes[2,1].set_title('Benchmark: Volatilidad Rolling')
    axes[2,1].legend()
    axes[2,1].grid(True)

    # 4️⃣ Histograma Retornos
    axes[3,1].hist(benchmark_returns, bins=50, color='gray', edgecolor='black', alpha=0.7)
    axes[3,1].axvline(benchmark_var5, color='red', linestyle='--', label=f'VaR 5%: {benchmark_var5:.2%}')
    axes[3,1].set_title('Benchmark: Distribución de Retornos')
    axes[3,1].legend()
    axes[3,1].grid(True)

    plt.tight_layout()
    plt.show()
    
def get_mae_mfe(trades, data):
    trades = trades[:5]
    trades_data = trades.copy()
    # Asegurar datetime
    trades_data['entry_time'] = pd.to_datetime(trades_data['entry_time'])
    trades_data['exit_time'] = pd.to_datetime(trades_data['exit_time'])
    data['date'] = pd.to_datetime(data['date'])

    mae_list = []
    mfe_list = []

    for row in trades_data.itertuples():

        ticker = row.ticker
        entry_time = row.entry_time
        exit_time = row.exit_time
        entry_price = row.entry_price
        side = row.type

        # Filtrar datos del trade
        trade_data = data[
            (data['ticker'] == ticker) &
            (data['date'] >= entry_time) &
            (data['date'] <= exit_time)
        ]

        if trade_data.empty:
            mae_list.append(None)
            mfe_list.append(None)
            continue

        highest_high = trade_data['high'].max()
        lowest_low = trade_data['low'].min()

        if side.lower() == "long":
            mae = lowest_low - entry_price
            mfe = highest_high - entry_price
        else:  # short
            mae = entry_price - highest_high
            mfe = entry_price - lowest_low

        mae_list.append(mae)
        mfe_list.append(mfe)

    trades_data['MAE'] = mae_list
    trades_data['MFE'] = mfe_list
        

    return trades_data


# ===========================
# MONTECARLO SIMALATION
#============================

def monte_carlo_final_equity_dd_sim(trades, f=0.01, n_sims=10000, show_graphic=False, dd_threshold=0.15):
    
    """
    Genera un array con el final_equity de cada simulación y un array con el max DD para cada simulación.
    Crea un gráfico con histogramas para ambos casos mostrando cómo están distribuidos los resultados y DD.
    Agrega líneas de media, mediana y percentiles 1,5, 95.
    
    trades: dataframe of trades [ticker, entry_time, exit_time, pnl,....]
    f: risk per trade (% valuer entre 0 y 1)
    n_sims: total de simulaciones
    show_graphic: si quiere mostrar histograma de distribucion de resultados
    dd_threshold: usado para ver que probabilidad hay de tener un DD > que ese paramatro
    """
    trades = trades.copy().sort_values("exit_time")
    r_mult = r_multiple(trades)  # función que devuelve retornos por trade
    n_trades = len(r_mult)
    
    final_equity = []
    max_dds = []
    
    for _ in range(n_sims):
        shuffled = np.random.choice(r_mult, size=n_trades, replace=True)
        
        equity = 1.0
        peak = 1.0
        max_dd = 0
        
        for r in shuffled:
            equity *= (1 + f * r)
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)
        
        final_equity.append(equity)
        max_dds.append(max_dd)
        
    final_equity = np.array(final_equity)
    max_dds = np.array(max_dds)
    
    
    
    
    # Calcular estadísticas
    stats = {}
    for arr, name in zip([final_equity, max_dds], ["Equity Final", "Drawdown Máx"]):
        stats[name] = {
            "media": np.mean(arr),
            "mediana": np.median(arr),
            "p1": np.percentile(arr, 1),
            "p5": np.percentile(arr, 5),
            "p95": np.percentile(arr, 95)
        }
        

    # max_dds es un array negativo o positivo? 
    # En tu función, dd = (equity - peak)/peak → es negativo
    # Por lo tanto, para > 15% usamos -0.15

    threshold = -1 * dd_threshold  # -15%
    
    """
    max_dds < threshold → genera un array booleano donde True = drawdown peor que -15%.
    np.mean(...) → como True=1 y False=0, el promedio es la proporción de veces que ocurre, es decir, la probabilidad.
    """
    prob_dd_gt_15 = np.mean(max_dds < threshold)

    print(f"Probabilidad de DD > 15%: {prob_dd_gt_15:.2%}")
    stats['DD_gt_threshold_%'] = prob_dd_gt_15
    
    if show_graphic:
        # Crear la figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        
        # --- Equity Final ---
        axes[0].hist(final_equity, bins=100, color='skyblue', edgecolor='black')
        axes[0].set_title('Monte Carlo (Equity Final in R)')
        axes[0].set_xlabel('Equity Final in R')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(alpha=0.3)
        
        # Líneas estadísticas
        axes[0].axvline(stats["Equity Final"]["media"], color='red', linestyle='--', label='Media')
        axes[0].axvline(stats["Equity Final"]["mediana"], color='green', linestyle='-', label='Mediana')
        axes[0].axvline(stats["Equity Final"]["p1"], color='purple', linestyle=':', label='Percentil 1')
        axes[0].axvline(stats["Equity Final"]["p5"], color='black', linestyle=':', label='Percentil 5')
        axes[0].axvline(stats["Equity Final"]["p95"], color='orange', linestyle=':', label='Percentil 95')
        axes[0].legend()
        
        # --- Drawdown Máximo ---
        axes[1].hist(max_dds, bins=100, color='salmon', edgecolor='black')
        axes[1].set_title('Monte Carlo (Drawdown Máximo in R)')
        axes[1].set_xlabel('Drawdown Máximo in R')
        axes[1].set_ylabel('Frecuencia')
        axes[1].grid(alpha=0.3)
        
        # Líneas estadísticas
        axes[1].axvline(stats["Drawdown Máx"]["media"], color='red', linestyle='--', label='Media')
        axes[1].axvline(stats["Drawdown Máx"]["mediana"], color='green', linestyle='-', label='Mediana')
        axes[1].axvline(stats["Drawdown Máx"]["p1"], color='purple', linestyle=':', label='Percentil 1')
        axes[1].axvline(stats["Drawdown Máx"]["p5"], color='black', linestyle=':', label='Percentil 5')
        axes[1].axvline(stats["Drawdown Máx"]["p95"], color='orange', linestyle=':', label='Percentil 95')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
    
    
    return stats