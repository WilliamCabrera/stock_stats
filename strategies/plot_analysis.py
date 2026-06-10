import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet("backside_short_lower_low_fix_stop_iterative_15m_trades.parquet")
df = df.sort_values("entry_time").reset_index(drop=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def profit_factor(x):
    w = x[x > 0].sum(); l = abs(x[x < 0].sum())
    return w / l if l != 0 else np.inf

def max_dd_series(pnl):
    cum = pnl.cumsum()
    roll_max = cum.cummax()
    return cum - roll_max

# ── Derived columns ───────────────────────────────────────────────────────────
df["cum_pnl"]  = df["pnl"].cumsum()
df["drawdown"] = max_dd_series(df["pnl"])
df["ym"]       = df["entry_time"].dt.to_period("M")
df["year"]     = df["entry_time"].dt.year
df["hour"]     = df["entry_time"].dt.hour
df["ret_pct"]  = df["Return"] * 100
df["winner"]   = df["pnl"] > 0

# Monthly aggregation
monthly = df.groupby("ym").agg(
    net_pnl  =("pnl",    "sum"),
    win_rate =("winner", "mean"),
    trades   =("pnl",    "count"),
    pf       =("pnl",    profit_factor),
    avg_ret  =("ret_pct","mean"),
).reset_index()
monthly["ym_str"] = monthly["ym"].astype(str)

# ── Style ─────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
CARD    = "#1a1d27"
GREEN   = "#26a69a"
RED     = "#ef5350"
BLUE    = "#42a5f5"
YELLOW  = "#ffca28"
PURPLE  = "#ab47bc"
GRAY    = "#546e7a"
TEXT    = "#e0e0e0"
SUBTEXT = "#90a4ae"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    "#2e3547",
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#2e3547",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 26), facecolor=BG)
fig.suptitle(
    "Backside Short Lower Low — Fix Stop Iterative  |  15m  |  OOS: Jan 2024 – Jun 2026",
    fontsize=14, fontweight="bold", color=TEXT, y=0.995
)

gs = gridspec.GridSpec(
    5, 3,
    figure=fig,
    hspace=0.48,
    wspace=0.32,
    top=0.975,
    bottom=0.04,
    left=0.06,
    right=0.97,
)

# ── 0. KPI header row ─────────────────────────────────────────────────────────
kpis = [
    ("Total Trades", f"{len(df):,}",            TEXT),
    ("Win Rate",     f"{df['winner'].mean()*100:.1f}%", GREEN),
    ("Net PnL",      f"+${df['pnl'].sum():.2f}", GREEN),
    ("Profit Factor",f"{profit_factor(df['pnl']):.2f}", BLUE),
    ("Sharpe",       f"{(df['Return'].mean()/df['Return'].std())*np.sqrt(252):.2f}", YELLOW),
    ("Max DD",       f"${df['drawdown'].min():.2f}", RED),
    ("Avg Return",   f"{df['ret_pct'].mean():.2f}%", BLUE),
    ("Avg Hold",     f"{((df['exit_time']-df['entry_time']).dt.total_seconds()/3600).mean():.1f}h", PURPLE),
    ("Avg RVOL",     f"{df['rvol_daily'].mean():.2f}x", YELLOW),
]
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_xlim(0, len(kpis))
ax_kpi.set_ylim(0, 1)
ax_kpi.axis("off")
ax_kpi.set_facecolor(BG)
for i, (label, value, color) in enumerate(kpis):
    x = i + 0.5
    rect = plt.Rectangle((i + 0.05, 0.05), 0.9, 0.9, color=CARD, zorder=0,
                          transform=ax_kpi.transData)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 0.65, value, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color, zorder=1)
    ax_kpi.text(x, 0.25, label, ha="center", va="center",
                fontsize=8, color=SUBTEXT, zorder=1)

# ── 1. Equity curve + drawdown ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
ax1.set_title("Equity Curve  &  Drawdown", fontweight="bold")
ax1.fill_between(range(len(df)), df["cum_pnl"], alpha=0.15, color=GREEN)
ax1.plot(df["cum_pnl"], color=GREEN, linewidth=1.4, label="Cumulative PnL")
ax1_r = ax1.twinx()
ax1_r.fill_between(range(len(df)), df["drawdown"], 0, alpha=0.35, color=RED)
ax1_r.plot(df["drawdown"], color=RED, linewidth=0.8, alpha=0.7, label="Drawdown")
ax1_r.tick_params(colors=SUBTEXT)
ax1_r.set_ylabel("Drawdown ($)", color=RED, fontsize=8)
ax1.set_ylabel("Cumulative PnL ($)", color=GREEN, fontsize=8)
ax1.set_xlabel("Trade #")
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
lines = [Line2D([0],[0],color=GREEN,lw=1.4), Line2D([0],[0],color=RED,lw=1.4)]
ax1.legend(lines, ["Cum. PnL","Drawdown"], loc="upper left", fontsize=8,
           facecolor=CARD, edgecolor="#2e3547")

# ── 2. Monthly PnL bar ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2, :2])
ax2.set_title("Net PnL por Mes", fontweight="bold")
colors_m = [GREEN if v >= 0 else RED for v in monthly["net_pnl"]]
bars = ax2.bar(monthly["ym_str"], monthly["net_pnl"], color=colors_m, width=0.7, zorder=2)
ax2.axhline(0, color=SUBTEXT, linewidth=0.7)
ax2.set_ylabel("Net PnL ($)")
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
# Rotate x labels
ax2.set_xticks(range(len(monthly)))
ax2.set_xticklabels(monthly["ym_str"], rotation=45, ha="right", fontsize=7)
# Add value labels on bars
for bar, val in zip(bars, monthly["net_pnl"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (2 if val >= 0 else -8),
             f"{val:.0f}", ha="center", va="bottom" if val >= 0 else "top",
             fontsize=6, color=TEXT)

# ── 3. Monthly win rate line ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 2])
ax3.set_title("Win Rate por Mes", fontweight="bold")
wr_vals = monthly["win_rate"] * 100
colors_wr = [GREEN if v >= 50 else RED for v in wr_vals]
ax3.bar(range(len(monthly)), wr_vals, color=colors_wr, width=0.7, zorder=2)
ax3.axhline(50, color=YELLOW, linewidth=1, linestyle="--", alpha=0.7)
ax3.axhline(wr_vals.mean(), color=BLUE, linewidth=1, linestyle=":", alpha=0.8)
ax3.set_ylabel("Win Rate (%)")
ax3.set_ylim(30, 85)
ax3.yaxis.grid(True); ax3.set_axisbelow(True)
ax3.set_xticks(range(len(monthly)))
ax3.set_xticklabels(monthly["ym_str"], rotation=90, fontsize=6)
ax3.text(len(monthly)-1, wr_vals.mean()+1, f"avg {wr_vals.mean():.1f}%",
         ha="right", fontsize=7, color=BLUE)

# ── 4. Return distribution ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
ax4.set_title("Distribución de Retornos", fontweight="bold")
bins = np.linspace(-100, 80, 60)
wins_r = df.loc[df["winner"], "ret_pct"]
loss_r = df.loc[~df["winner"], "ret_pct"]
ax4.hist(loss_r, bins=bins, color=RED,   alpha=0.75, label="Losers",  zorder=2)
ax4.hist(wins_r, bins=bins, color=GREEN, alpha=0.75, label="Winners", zorder=2)
ax4.axvline(df["ret_pct"].mean(), color=YELLOW, linewidth=1.2, linestyle="--",
            label=f"Mean {df['ret_pct'].mean():.1f}%")
ax4.axvline(df["ret_pct"].median(), color=BLUE, linewidth=1.2, linestyle=":",
            label=f"Median {df['ret_pct'].median():.1f}%")
ax4.set_xlabel("Return (%)")
ax4.set_ylabel("Trades")
ax4.yaxis.grid(True); ax4.set_axisbelow(True)
ax4.legend(fontsize=7, facecolor=CARD, edgecolor="#2e3547")

# ── 5. MAE vs MFE scatter ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 1])
ax5.set_title("MAE % vs MFE %", fontweight="bold")
sample = df.sample(min(1500, len(df)), random_state=42)
sc_colors = [GREEN if w else RED for w in sample["winner"]]
ax5.scatter(sample["mae_pct"], sample["mfe_pct"],
            c=sc_colors, alpha=0.35, s=8, zorder=2)
ax5.plot([0, 80], [0, 80], color=SUBTEXT, linewidth=0.7, linestyle="--", alpha=0.5)
ax5.set_xlabel("MAE % (adverse)")
ax5.set_ylabel("MFE % (favorable)")
ax5.set_xlim(0, 80); ax5.set_ylim(0, 80)
ax5.yaxis.grid(True); ax5.xaxis.grid(True); ax5.set_axisbelow(True)
legend_els = [Patch(color=GREEN, label="Winner"), Patch(color=RED, label="Loser")]
ax5.legend(handles=legend_els, fontsize=7, facecolor=CARD, edgecolor="#2e3547")

# ── 6. PnL by hour of entry ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 2])
ax6.set_title("Net PnL por Hora de Entrada", fontweight="bold")
by_hour = df.groupby("hour").agg(
    net_pnl  =("pnl",    "sum"),
    trades   =("pnl",    "count"),
    win_rate =("winner", "mean"),
).reset_index()
colors_h = [GREEN if v >= 0 else RED for v in by_hour["net_pnl"]]
ax6.bar(by_hour["hour"], by_hour["net_pnl"], color=colors_h, width=0.7, zorder=2)
ax6.axhline(0, color=SUBTEXT, linewidth=0.7)
ax6.set_xlabel("Hour (ET)")
ax6.set_ylabel("Net PnL ($)")
ax6.set_xticks(by_hour["hour"])
ax6.yaxis.grid(True); ax6.set_axisbelow(True)
ax6_r = ax6.twinx()
ax6_r.plot(by_hour["hour"], by_hour["win_rate"]*100, color=YELLOW,
           linewidth=1.5, marker="o", markersize=4, label="Win Rate %")
ax6_r.set_ylabel("Win Rate %", color=YELLOW, fontsize=8)
ax6_r.tick_params(colors=SUBTEXT)
ax6_r.set_ylim(0, 100)

# ── 7. Yearly summary bar ─────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[4, 0])
ax7.set_title("Net PnL por Año", fontweight="bold")
by_year = df.groupby("year").agg(
    net_pnl  =("pnl", "sum"),
    win_rate =("winner","mean"),
    trades   =("pnl","count"),
).reset_index()
colors_y = [GREEN if v >= 0 else RED for v in by_year["net_pnl"]]
bars_y = ax7.bar(by_year["year"].astype(str), by_year["net_pnl"],
                 color=colors_y, width=0.5, zorder=2)
for bar, val in zip(bars_y, by_year["net_pnl"]):
    ax7.text(bar.get_x()+bar.get_width()/2,
             bar.get_height() + (3 if val >= 0 else -10),
             f"${val:.0f}", ha="center", fontsize=8, fontweight="bold", color=TEXT)
ax7.axhline(0, color=SUBTEXT, linewidth=0.7)
ax7.set_ylabel("Net PnL ($)")
ax7.yaxis.grid(True); ax7.set_axisbelow(True)
ax7_r = ax7.twinx()
ax7_r.plot(range(len(by_year)), by_year["win_rate"]*100,
           color=YELLOW, marker="D", markersize=6, linewidth=1.5)
ax7_r.set_ylabel("Win Rate %", color=YELLOW, fontsize=8)
ax7_r.tick_params(colors=SUBTEXT)
ax7_r.set_ylim(40, 70)

# ── 8. RVOL vs Return scatter ─────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[4, 1])
ax8.set_title("RVOL Daily vs Return (%)", fontweight="bold")
rvol_cap = df["rvol_daily"].clip(0, 15)
sc_colors2 = [GREEN if w else RED for w in df["winner"]]
ax8.scatter(rvol_cap, df["ret_pct"].clip(-60, 60),
            c=sc_colors2, alpha=0.2, s=6, zorder=2)
# Running mean
rv_bins = np.linspace(0, 15, 16)
df["rvol_bin"] = pd.cut(rvol_cap, bins=rv_bins)
rv_mean = df.groupby("rvol_bin", observed=False)["ret_pct"].mean()
bin_centers = [(b.left + b.right) / 2 for b in rv_mean.index]
ax8.plot(bin_centers, rv_mean.values, color=YELLOW, linewidth=2, zorder=3, label="Avg Ret")
ax8.axhline(0, color=SUBTEXT, linewidth=0.7)
ax8.set_xlabel("RVOL Daily (capped 15x)")
ax8.set_ylabel("Return % (capped ±60)")
ax8.yaxis.grid(True); ax8.xaxis.grid(True); ax8.set_axisbelow(True)
legend_els2 = [Patch(color=GREEN, label="Winner"), Patch(color=RED, label="Loser"),
               Line2D([0],[0], color=YELLOW, lw=2, label="Avg Ret")]
ax8.legend(handles=legend_els2, fontsize=7, facecolor=CARD, edgecolor="#2e3547")

# ── 9. Profit Factor by month ─────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[4, 2])
ax9.set_title("Profit Factor por Mes", fontweight="bold")
pf_vals = monthly["pf"].clip(0, 5)
colors_pf = [GREEN if v >= 1 else RED for v in pf_vals]
ax9.bar(range(len(monthly)), pf_vals, color=colors_pf, width=0.7, zorder=2)
ax9.axhline(1, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.8)
ax9.set_ylabel("Profit Factor (capped 5)")
ax9.set_ylim(0, 5.5)
ax9.yaxis.grid(True); ax9.set_axisbelow(True)
ax9.set_xticks(range(len(monthly)))
ax9.set_xticklabels(monthly["ym_str"], rotation=90, fontsize=6)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "backside_short_lower_low_fix_stop_iterative_15m_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
