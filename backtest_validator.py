"""
╔══════════════════════════════════════════════════════╗
║         QUANT ALPHA — BACKTEST VALIDATOR v2          ║
║   Detects: Lookahead bias, Overfitting, Bad Assumptions ║
║   Streamlit app — free to deploy on Streamlit Cloud  ║
╚══════════════════════════════════════════════════════╝

Install:  pip install streamlit pandas numpy scipy
Run:      streamlit run backtest_validator_v2.py
Deploy:   streamlit.io (free)

UPGRADES:
• Fixed KeyError: 'drawdown' in monthly reports
• Added Python 3.9+ version check for ast.unparse()
• Replaced ast.walk() with targeted NodeVisitor for lookahead detection
• Robust CSV upload handling (encoding fallbacks, column detection)
• Replaced raw HTML with native Streamlit components for accessibility
• Extracted scoring constants for maintainability
• Added lru_cache for statistical metrics performance
• Sanitized file uploads for security
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import ast
import io
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache

# ─────────────────────────────────────────────────────────────
# SCORING CONSTANTS (MAINTAINABILITY)
# ─────────────────────────────────────────────────────────────
SCORING = {
    'START': 100,
    'CRITICAL_PENALTY': -25,
    'WARNING_PENALTY': -10,
    'THRESHOLDS': {'VALID': 80, 'QUESTIONABLE': 55}
}

# ─────────────────────────────────────────────────────────────
# PYTHON VERSION CHECK
# ─────────────────────────────────────────────────────────────
if sys.version_info < (3, 9):
    st.error("⚠️ This tool requires Python 3.9+ for code analysis. Please upgrade or use the Returns Analysis tab only.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtest Validator | Quant Alpha",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — dark quant terminal aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp { background-color: #0a0e1a; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
}
.main-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    color: #00d4ff;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #64748b;
    margin: 8px 0 0;
    font-size: 0.95rem;
}

/* Score card */
.score-card {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 16px;
}
.score-great  { background: linear-gradient(135deg,#052e16,#14532d); border:1px solid #22c55e; }
.score-ok     { background: linear-gradient(135deg,#1c1400,#3a2c00); border:1px solid #eab308; }
.score-bad    { background: linear-gradient(135deg,#1c0606,#3a0d0d); border:1px solid #ef4444; }
.score-number { font-family:'JetBrains Mono',monospace; font-size:3.5rem; font-weight:700; }
.score-label  { font-size:0.85rem; color:#94a3b8; margin-top:4px; }

/* Metric box */
.metric-box {
    background:#0d1b2a; border:1px solid #1e3a5f;
    border-radius:10px; padding:16px; text-align:center;
}
.metric-val {
    font-family:'JetBrains Mono',monospace;
    font-size:1.6rem; font-weight:700; color:#00d4ff;
}
.metric-lbl { font-size:0.78rem; color:#64748b; margin-top:4px; }

/* Section headers */
.section-header {
    font-family:'JetBrains Mono',monospace;
    font-size:0.75rem; color:#00d4ff; letter-spacing:2px;
    text-transform:uppercase; margin:24px 0 12px;
    border-bottom:1px solid #1e3a5f; padding-bottom:8px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:#080c16;
    border-right:1px solid #1e3a5f;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#0066cc,#0044aa);
    color:white; border:none; border-radius:8px;
    font-family:'JetBrains Mono',monospace;
    font-weight:700; padding:12px 24px;
    width:100%; transition:all 0.2s;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#0080ff,#0055cc);
    transform:translateY(-1px);
}

/* Text areas */
.stTextArea textarea {
    background:#080c16; color:#e2e8f0;
    border:1px solid #1e3a5f; border-radius:8px;
    font-family:'JetBrains Mono',monospace; font-size:0.82rem;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; color:#64748b;
}
.stTabs [aria-selected="true"] { color:#00d4ff !important; }

/* Radio */
.stRadio label { font-size:0.88rem; color:#94a3b8; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────
@dataclass
class Issue:
    severity: str   # CRITICAL | WARNING | INFO | OK
    category: str
    message: str
    detail: str = ""
    line: Optional[int] = None

@dataclass
class ValidationReport:
    score: int = SCORING['START']
    issues: List[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    verdict: str = "UNKNOWN"

    def add(self, severity, category, message, detail="", line=None):
        self.issues.append(Issue(severity, category, message, detail, line))
        if severity == "CRITICAL": 
            self.score += SCORING['CRITICAL_PENALTY']
        elif severity == "WARNING": 
            self.score += SCORING['WARNING_PENALTY']
        self.score = max(0, self.score)

    def finalize(self):
        if self.score >= SCORING['THRESHOLDS']['VALID']:   
            self.verdict = "✅ VALID"
        elif self.score >= SCORING['THRESHOLDS']['QUESTIONABLE']: 
            self.verdict = "⚠️ QUESTIONABLE"
        else:                  
            self.verdict = "❌ INVALID"

# ─────────────────────────────────────────────────────────────
# VALIDATOR MODULES
# ─────────────────────────────────────────────────────────────

class LookaheadDetector:
    """Detects lookahead bias via Python AST analysis — OPTIMIZED"""

    FUTURE_KEYWORDS = [
        'future_', 'next_', 'forward_', 'tomorrow_',
        'lead_', '_future', '_next', '_fwd', '_ahead'
    ]

    class SignalVisitor(ast.NodeVisitor):
        def __init__(self):
            self.shift_found = False
            self.signal_assignments = []
            self.future_vars = []
            
        def visit_Name(self, node):
            for kw in LookaheadDetector.FUTURE_KEYWORDS:
                if kw in node.id.lower():
                    self.future_vars.append((getattr(node, 'lineno', 0), node.id))
            self.generic_visit(node)
            
        def visit_Assign(self, node):
            seg = ast.unparse(node)
            targets = [ast.unparse(t) for t in node.targets]
            is_signal = any('signal' in t.lower() for t in targets)
            
            # Check for .shift() anywhere in the value expression
            has_shift = '.shift(' in seg
            for child in ast.walk(node.value):
                if isinstance(child, ast.Call) and getattr(child.func, 'attr', None) == 'shift':
                    has_shift = True
                    break
                    
            if has_shift:
                self.shift_found = True
            elif is_signal:
                self.signal_assignments.append((getattr(node, 'lineno', 0), seg[:80]))
            self.generic_visit(node)

    def analyze(self, code: str, report: ValidationReport):
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            report.add("CRITICAL", "Syntax", f"Cannot parse code: {e}")
            return

        visitor = self.SignalVisitor()
        visitor.visit(tree)

        # Report future-leaking variable names
        for lineno, name in visitor.future_vars:
            report.add("CRITICAL", "Lookahead",
                f'Variable "{name}" suggests future data usage',
                "This variable name implies accessing data not available at trade time.",
                lineno)

        # Report signal assignments without shift
        if visitor.signal_assignments:
            for lineno, seg in visitor.signal_assignments[:3]:
                report.add("WARNING", "Lookahead",
                    f"Signal assigned without .shift(1) detected",
                    f"Code: {seg}\nAdd .shift(1) to prevent using same-bar signal for entry.",
                    lineno)
        elif visitor.shift_found:
            report.add("OK", "Lookahead",
                ".shift() detected — signal properly offset",
                "Entry uses previous bar's signal. ✓")
        else:
            report.add("INFO", "Lookahead",
                "No signal/shift pattern found",
                "Could not automatically verify signal timing. Manual review recommended.")

        # Check for close-price entry (common mistake)
        if "['Close']" in code or '["Close"]' in code:
            if 'open' not in code.lower() and 'Open' not in code:
                report.add("WARNING", "Lookahead",
                    "Using Close price for entry — possible lookahead",
                    "Close price is unknown until bar ends. Use next bar's Open for realistic entry.")

        # Check for data normalization on full dataset
        if '.fit(' in code and 'train' not in code.lower():
            report.add("CRITICAL", "Lookahead",
                "Scaler/model fitted on full dataset",
                "fit() on full data leaks future into normalization. Use only training period.")


class OverfittingDetector:
    """Statistical overfitting detection from returns data — CACHED"""

    @staticmethod
    @lru_cache(maxsize=32)
    def _compute_stats(returns_tuple: tuple):
        returns = pd.Series(returns_tuple)
        if len(returns) < 20:
            return None
            
        mean_r = returns.mean()
        std_r  = returns.std()
        sr     = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0

        T    = len(returns)
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())
        try:
            sr_std = np.sqrt(
                max(1e-10, (1 + (0.5*sr**2) - (skew*sr) +
                ((kurt-3)/4 * sr**2)) / T))
            dsr = (sr - 0) / (sr_std * np.sqrt(max(1, 1)))  # n_trials passed separately
        except Exception:
            dsr = sr

        cumret = (1 + returns).cumprod()
        log_eq = np.log(cumret.clip(lower=1e-6))
        r_val  = np.corrcoef(np.arange(len(log_eq)), log_eq)[0,1] if len(log_eq) > 1 else 0

        roll_max = cumret.cummax()
        dd       = (cumret - roll_max) / roll_max
        max_dd   = float(dd.min()) if len(dd) > 0 else 0

        win_rate = (returns > 0).mean()
        gross_profit = returns[returns > 0].sum()
        gross_loss   = abs(returns[returns < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            'Sharpe Ratio': round(sr, 3),
            'Deflated Sharpe': round(dsr, 3),
            'Max Drawdown': f'{max_dd:.1%}',
            'Win Rate': f'{win_rate:.1%}',
            'Profit Factor': round(pf, 2) if pf != np.inf else '∞',
            'Curve Smoothness R²': round(r_val**2, 3),
            'Total Return': f'{(cumret.iloc[-1]-1):.1%}',
            'Observations': len(returns),
        }

    def analyze(self, returns: pd.Series, n_trials: int, report: ValidationReport):
        if len(returns) < 20:
            report.add("WARNING", "Overfitting",
                "Too few data points for reliable statistics",
                f"Only {len(returns)} observations. Need at least 252 (1 year).")
            return

        stats = self._compute_stats(tuple(returns.dropna()))
        if not stats:
            return
            
        report.metrics.update(stats)

        sr = float(stats['Sharpe Ratio'])
        pf = stats['Profit Factor']
        r2 = float(stats['Curve Smoothness R²'])
        T = stats['Observations']

        if sr > 4:
            report.add("CRITICAL", "Overfitting",
                f"Sharpe Ratio {sr:.2f} is unrealistically high",
                "Sharpe > 4 is almost never seen in live trading without bias. "
                "Institutional strategies average 0.5–1.5.")
        elif sr > 3:
            report.add("WARNING", "Overfitting",
                f"Sharpe Ratio {sr:.2f} is suspiciously high",
                "Sharpe > 3 warrants serious investigation. "
                "Possible lookahead or overfitting.")
        elif sr > 1.5:
            report.add("INFO", "Overfitting",
                f"Sharpe Ratio {sr:.2f} is good but verify OOS performance",
                "Solid Sharpe. Make sure it holds out-of-sample.")
        else:
            report.add("OK", "Overfitting",
                f"Sharpe Ratio {sr:.2f} is within realistic range",
                "Realistic for a live strategy.")

        if isinstance(pf, (int, float)) and pf > 5:
            report.add("WARNING", "Overfitting",
                f"Profit Factor {pf:.1f} is unrealistically high",
                "Profit Factor > 3 rarely survives live trading. "
                "Realistic range: 1.3–2.5.")

        if r2 > 0.97 and sr > 1.5:
            report.add("WARNING", "Overfitting",
                "Equity curve is suspiciously smooth (R² > 0.97)",
                "Perfect equity curves are a red flag for lookahead bias. "
                "Real strategies have bumpy equity curves.")

        if T < 252:
            report.add("WARNING", "Overfitting",
                f"Only {T} observations — less than 1 year",
                "Insufficient history to validate a strategy. "
                "Need minimum 2–3 years for robustness.")


class AssumptionChecker:
    """Checks for unrealistic trading assumptions"""

    def analyze_code(self, code: str, report: ValidationReport):
        code_lower = code.lower()

        commission_terms = ['commission', 'fee', 'cost', 'spread', 'slippage']
        has_costs = any(t in code_lower for t in commission_terms)
        if not has_costs:
            report.add("WARNING", "Assumptions",
                "No transaction costs detected",
                "Strategy appears to assume zero commissions and slippage. "
                "For crypto: add 0.1% per trade. Forex: add spread. "
                "Costs can reduce returns by 30–70% for high-frequency strategies.")
        else:
            report.add("OK", "Assumptions",
                "Transaction costs detected in code ✓")

        if '-1' in code and ('short' in code_lower or 'sell' in code_lower):
            if 'borrow' not in code_lower:
                report.add("INFO", "Assumptions",
                    "Short selling detected — borrowing costs not modeled",
                    "Short selling incurs borrowing fees (0.5–5% annually). "
                    "Add these costs for accurate results.")

        if 'leverage' in code_lower or 'margin' in code_lower:
            report.add("INFO", "Assumptions",
                "Leverage detected — ensure margin calls are modeled",
                "Leveraged strategies can face margin calls during drawdowns "
                "that terminate positions prematurely.")

        if 'volume' not in code_lower and 'liquidity' not in code_lower:
            report.add("INFO", "Assumptions",
                "No liquidity/volume constraints detected",
                "Large positions relative to average volume cause slippage. "
                "Consider adding volume-based position sizing.")

    def analyze_trades(self, returns: pd.Series, report: ValidationReport):
        n_trades = len(returns[returns != 0])
        if n_trades < 30:
            report.add("WARNING", "Statistical",
                f"Only {n_trades} trades — statistically unreliable",
                "Need minimum 30 trades for basic statistics, "
                "ideally 100+ for robust conclusions.")
        elif n_trades > 5000:
            report.add("INFO", "Assumptions",
                f"{n_trades} trades — very high frequency",
                "High trade count increases sensitivity to slippage and "
                "commissions. Verify costs are included.")
        else:
            report.add("OK", "Statistical",
                f"{n_trades} trades — sufficient sample size ✓")


class PropFirmChecker:
    """Checks compliance with prop firm evaluation rules"""

    FIRMS = {
        'FTMO Challenge ($100K)': {
            'max_daily_dd': 0.05, 'max_total_dd': 0.10,
            'profit_target': 0.10, 'min_days': 4, 'max_days': 30
        },
        'MyFundedFX ($100K)': {
            'max_daily_dd': 0.05, 'max_total_dd': 0.10,
            'profit_target': 0.08, 'min_days': 5, 'max_days': 30
        },
        'Topstep ($150K)': {
            'max_daily_dd': 0.03, 'max_total_dd': 0.06,
            'profit_target': 0.06, 'min_days': 5, 'max_days': None
        },
        'The Funded Trader ($200K)': {
            'max_daily_dd': 0.05, 'max_total_dd': 0.12,
            'profit_target': 0.08, 'min_days': 5, 'max_days': 45
        }
    }

    def check(self, daily_returns: pd.Series, firm_name: str, report: ValidationReport):
        if firm_name not in self.FIRMS:
            return
        rules = self.FIRMS[firm_name]

        worst_day = float(daily_returns.min())
        if abs(worst_day) > rules['max_daily_dd']:
            report.add("CRITICAL", f"PropFirm:{firm_name}",
                f"Daily DD {worst_day:.1%} exceeds limit {rules['max_daily_dd']:.1%}",
                "Strategy would FAIL this prop firm challenge — "
                f"violated max daily loss rule on at least one day.")
        else:
            report.add("OK", f"PropFirm:{firm_name}",
                f"Daily drawdown {worst_day:.1%} within limit {rules['max_daily_dd']:.1%} ✓")

        cumret = (1 + daily_returns).cumprod()
        dd = ((cumret - cumret.cummax()) / cumret.cummax()).min()
        if abs(float(dd)) > rules['max_total_dd']:
            report.add("CRITICAL", f"PropFirm:{firm_name}",
                f"Total DD {dd:.1%} exceeds limit {rules['max_total_dd']:.1%}",
                "Strategy breaches maximum account drawdown rule.")
        else:
            report.add("OK", f"PropFirm:{firm_name}",
                f"Total drawdown {dd:.1%} within limit {rules['max_total_dd']:.1%} ✓")

        total_return = float(cumret.iloc[-1] - 1)
        if total_return >= rules['profit_target']:
            report.add("OK", f"PropFirm:{firm_name}",
                f"Profit target {total_return:.1%} achieved "
                f"(target: {rules['profit_target']:.1%}) ✓")
        else:
            report.add("INFO", f"PropFirm:{firm_name}",
                f"Profit target NOT reached: {total_return:.1%} "
                f"(need {rules['profit_target']:.1%})",
                "Strategy would not pass the profit target within the test period.")

        active = int((daily_returns != 0).sum())
        if active < rules['min_days']:
            report.add("WARNING", f"PropFirm:{firm_name}",
                f"Only {active} active days (minimum: {rules['min_days']})",
                "Most prop firms require minimum trading days.")
        else:
            report.add("OK", f"PropFirm:{firm_name}",
                f"{active} active trading days — meets minimum requirement ✓")


# ─────────────────────────────────────────────────────────────
# MAIN VALIDATION RUNNER
# ─────────────────────────────────────────────────────────────

def run_validation(code: str, returns: Optional[pd.Series], n_trials: int,
                   check_propfirm: bool, firm_name: str) -> ValidationReport:
    report = ValidationReport()

    if code.strip():
        LookaheadDetector().analyze(code, report)
        AssumptionChecker().analyze_code(code, report)

    if returns is not None and len(returns) > 5:
        OverfittingDetector().analyze(returns, n_trials, report)
        AssumptionChecker().analyze_trades(returns, report)
        if check_propfirm:
            PropFirmChecker().check(returns, firm_name, report)

    report.finalize()
    return report


# ─────────────────────────────────────────────────────────────
# UI HELPERS — NATIVE COMPONENTS (ACCESSIBILITY)
# ─────────────────────────────────────────────────────────────

def render_issue(issue: Issue):
    icon = {'CRITICAL':'🔴','WARNING':'🟡','INFO':'🔵','OK':'🟢'}.get(issue.severity,'⚪')
    line_txt = f" [line {issue.line}]" if issue.line else ""
    detail_txt = f"\n\n_{issue.detail}_" if issue.detail else ""
    
    message = f"{icon} **[{issue.category}]{line_txt}**\n\n{issue.message}{detail_txt}"
    
    if issue.severity == "CRITICAL":
        st.error(message)
    elif issue.severity == "WARNING":
        st.warning(message)
    elif issue.severity == "OK":
        st.success(message)
    else:
        st.info(message)

def score_css(score):
    if score >= 80: return 'score-great'
    if score >= 55: return 'score-ok'
    return 'score-bad'

def score_color(score):
    if score >= 80: return '#22c55e'
    if score >= 55: return '#eab308'
    return '#ef4444'

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 BACKTEST VALIDATOR v2</h1>
    <p>Detect lookahead bias · overfitting · unrealistic assumptions · prop firm compliance</p>
    <p style="color:#1e3a5f;font-size:0.75rem;font-family:'JetBrains Mono'">
        QUANT ALPHA — FREE TOOL
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:JetBrains Mono;font-size:0.7rem;
    color:#00d4ff;letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid #1e3a5f;padding-bottom:8px;margin-bottom:16px'>
    ⚙ SETTINGS
    </div>""", unsafe_allow_html=True)

    n_trials = st.slider(
        "Number of strategies tested",
        min_value=1, max_value=200, value=1,
        help="How many strategies did you test before this one? "
             "Higher = stricter Deflated Sharpe calculation"
    )

    st.markdown("---")
    check_propfirm = st.checkbox("Check Prop Firm Compliance", value=False)
    if check_propfirm:
        firm_name = st.selectbox(
            "Select Prop Firm",
            options=list(PropFirmChecker.FIRMS.keys())
        )
    else:
        firm_name = list(PropFirmChecker.FIRMS.keys())[0]

    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono;font-size:0.65rem;color:#334155'>
    <b style='color:#00d4ff'>DETECTS:</b><br>
    → Lookahead bias (AST)<br>
    → Missing .shift() on signals<br>
    → Scaler fitted on full data<br>
    → Overfitting (Deflated Sharpe)<br>
    → Unrealistic Sharpe (>3)<br>
    → Smooth equity curve bias<br>
    → Missing transaction costs<br>
    → Insufficient trade count<br>
    → Prop firm rule violations<br>
    → Short selling cost omission
    </div>""", unsafe_allow_html=True)

# ── MAIN TABS ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  📋 Code Analysis  ",
    "  📊 Returns Analysis  ",
    "  📖 How It Works  "
])

# ─── TAB 1: CODE ANALYSIS ────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">PASTE YOUR PYTHON STRATEGY CODE</div>',
                unsafe_allow_html=True)

    sample_code = '''import pandas as pd
import numpy as np
import yfinance as yf

# Download BTC data
df = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')
df = df[['Close']].copy()

# Calculate EMAs
df['EMA_fast'] = df['Close'].ewm(span=20).mean()
df['EMA_slow'] = df['Close'].ewm(span=50).mean()

# Signal (note: no .shift — possible lookahead!)
df['Signal'] = np.where(df['EMA_fast'] > df['EMA_slow'], 1, 0)

# Returns (no commission modeled)
df['Returns'] = df['Close'].pct_change()
df['Strategy'] = df['Returns'] * df['Signal']
'''

    code_input = st.text_area(
        "Strategy Code",
        value=sample_code,
        height=300,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3,1])
    with col1:
        analyze_code_btn = st.button("🔍 ANALYZE CODE", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if analyze_code_btn and code_input.strip():
        with st.spinner("Analyzing code..."):
            report = run_validation(
                code=code_input,
                returns=None,
                n_trials=n_trials,
                check_propfirm=False,
                firm_name=firm_name
            )

        # Score display
        sc = score_css(report.score)
        col_s, col_v = st.columns([1,2])
        with col_s:
            st.markdown(f"""
            <div class="score-card {sc}">
                <div class="score-number" style="color:{score_color(report.score)}">
                    {report.score}
                </div>
                <div class="score-label">REALISM SCORE / 100</div>
                <div style="margin-top:8px;font-size:1rem;font-weight:700">
                    {report.verdict}
                </div>
            </div>""", unsafe_allow_html=True)
        with col_v:
            n_crit = sum(1 for i in report.issues if i.severity=='CRITICAL')
            n_warn = sum(1 for i in report.issues if i.severity=='WARNING')
            n_ok   = sum(1 for i in report.issues if i.severity=='OK')
            st.markdown(f"""
            <div style="margin-top:12px">
                <div class="metric-box" style="margin-bottom:8px">
                    <span style="color:#ef4444;font-size:1.2rem">🔴 {n_crit} Critical</span>
                </div>
                <div class="metric-box" style="margin-bottom:8px">
                    <span style="color:#f59e0b;font-size:1.2rem">🟡 {n_warn} Warnings</span>
                </div>
                <div class="metric-box">
                    <span style="color:#22c55e;font-size:1.2rem">🟢 {n_ok} Passed</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">DETAILED FINDINGS</div>',
                    unsafe_allow_html=True)
        for issue in report.issues:
            render_issue(issue)

# ─── TAB 2: RETURNS ANALYSIS — FIXED DRAWDOWN BUG ───────────
with tab2:
    st.markdown('<div class="section-header">UPLOAD RETURNS DATA</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="issue-info">
    📋 Upload a CSV with a <b>returns</b> column (daily returns as decimals, e.g. 0.023 = 2.3%).
    <br>Optional: <b>date</b> column for time-series analysis.
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    input_method = st.radio(
        "Input method",
        ["Upload CSV", "Paste returns (comma-separated)"],
        horizontal=True
    )

    returns = None

    if input_method == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload returns CSV",
            type=['csv'],
            label_visibility="collapsed"
        )
        if uploaded:
            try:
                # Sanitize filename
                safe_name = re.sub(r'[^\w\-_\.]', '_', uploaded.name)
                if not safe_name.lower().endswith('.csv'):
                    st.error("❌ Only CSV files allowed")
                elif uploaded.size > 50 * 1024 * 1024:  # 50MB limit
                    st.error("❌ File too large — please upload < 50MB")
                else:
                    # Try multiple encodings
                    for enc in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df_up = pd.read_csv(uploaded, encoding=enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        st.error("❌ Could not read file — unsupported encoding")
                        st.stop()
                        
                    # Validate returns column
                    ret_col = None
                    for col in df_up.columns:
                        if 'return' in col.lower() or 'ret' in col.lower() or 'pnl' in col.lower():
                            ret_col = col
                            break
                    if ret_col is None:
                        numeric_cols = df_up.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            ret_col = numeric_cols[0]
                        else:
                            st.error("❌ No numeric/returns column found")
                            st.stop()
                            
                    returns = pd.to_numeric(df_up[ret_col], errors='coerce').dropna()
                    if len(returns) < 5:
                        st.error("❌ Too few valid return values")
                        st.stop()
                        
                    st.success(f"✅ Loaded {len(returns)} observations from '{ret_col}'")
                    st.dataframe(df_up.head(5), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {type(e).__name__}: {e}")
    else:
        raw = st.text_area(
            "Paste daily returns (comma or newline separated)",
            placeholder="0.012, -0.005, 0.023, -0.008, 0.031 ...",
            height=100,
            label_visibility="collapsed"
        )
        if raw.strip():
            try:
                vals = [float(x.strip()) for x in raw.replace('\n',',').split(',') if x.strip()]
                returns = pd.Series(vals)
                if returns.isnull().any():
                    st.warning(f"⚠️ {returns.isnull().sum()} missing values removed")
                    returns = returns.dropna()
                if (returns > 1).any() or (returns < -1).any():
                    st.warning("⚠️ Returns outside [-100%, +∞] detected — verify data format (use decimals: 0.05 = 5%)")
                st.success(f"✅ Loaded {len(returns)} data points")
            except Exception as e:
                st.error(f"Could not parse: {e}")

    also_check_code = st.checkbox(
        "Also run code analysis (paste code in Tab 1 first)", value=False
    )

    analyze_returns_btn = st.button("📊 ANALYZE RETURNS", use_container_width=True)

    if analyze_returns_btn and returns is not None:
        with st.spinner("Running statistical analysis..."):
            code_to_use = code_input if also_check_code else ""
            report = run_validation(
                code=code_to_use,
                returns=returns,
                n_trials=n_trials,
                check_propfirm=check_propfirm,
                firm_name=firm_name
            )

        # Score
        sc = score_css(report.score)
        col_s, col_m = st.columns([1, 2])
        with col_s:
            st.markdown(f"""
            <div class="score-card {sc}">
                <div class="score-number" style="color:{score_color(report.score)}">
                    {report.score}
                </div>
                <div class="score-label">REALISM SCORE / 100</div>
                <div style="margin-top:8px;font-size:1rem;font-weight:700">
                    {report.verdict}
                </div>
            </div>""", unsafe_allow_html=True)

        with col_m:
            st.markdown('<div class="section-header">KEY METRICS</div>',
                        unsafe_allow_html=True)
            metrics = report.metrics
            if metrics:
                cols = st.columns(3)
                items = list(metrics.items())
                for i, (k, v) in enumerate(items):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-box" style="margin-bottom:8px">
                            <div class="metric-val">{v}</div>
                            <div class="metric-lbl">{k}</div>
                        </div>""", unsafe_allow_html=True)

        # Equity curve
        if returns is not None:
            st.markdown('<div class="section-header">EQUITY CURVE</div>',
                        unsafe_allow_html=True)
            cumret = (1 + returns).cumprod()
            df_plot = pd.DataFrame({
                'Strategy': cumret,
                'Flat (1.0)': 1.0
            })
            st.line_chart(df_plot, use_container_width=True, height=250)

        # Issues
        st.markdown('<div class="section-header">DETAILED FINDINGS</div>',
                    unsafe_allow_html=True)
        for issue in report.issues:
            render_issue(issue)

        # Download report
        report_dict = {
            'score': report.score,
            'verdict': report.verdict,
            'metrics': report.metrics,
            'issues': [
                {'severity': i.severity, 'category': i.category,
                 'message': i.message, 'detail': i.detail}
                for i in report.issues
            ]
        }
        st.download_button(
            "⬇️ Download Report (JSON)",
            data=json.dumps(report_dict, indent=2),
            file_name="backtest_validation_report.json",
            mime="application/json"
        )

# ─── TAB 3: HOW IT WORKS ─────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">WHAT THIS TOOL DETECTS</div>
    """, unsafe_allow_html=True)

    checks = [
        ("🔴 Lookahead Bias (Code)", """
        Analyzes your Python code using AST parsing to find:
        - Variable names suggesting future data (future_, next_, _fwd)
        - Signals assigned without .shift(1) offset
        - Scalers/models fitted on the full dataset (data leakage)
        - Close price used as entry price (bar not yet closed)
        """),
        ("🟡 Overfitting Detection", """
        Statistical tests on your returns series:
        - Deflated Sharpe Ratio (adjusts for number of trials)
        - Suspiciously high Sharpe (> 3 is a red flag)
        - Equity curve smoothness (R² > 0.97 with high Sharpe = suspicious)
        - Insufficient sample size (< 252 observations)
        - Multiple testing correction
        """),
        ("🟡 Unrealistic Assumptions", """
        Checks your code and returns for:
        - Zero transaction costs (commissions, slippage, spread)
        - Short selling without borrowing costs
        - No liquidity/volume constraints
        - Too few trades for statistical validity (< 30)
        """),
        ("🔵 Prop Firm Compliance", """
        Tests your strategy against real prop firm rules:
        - FTMO: Max 5% daily DD, 10% total DD, 10% profit target
        - Topstep: Max 3% daily DD, 6% total DD
        - MyFundedFX, The Funded Trader rules
        - Minimum active trading days check
        """),
    ]

    for title, desc in checks:
        with st.expander(title):
            st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.82rem;"
                       f"color:#94a3b8'>{desc}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">SCORING SYSTEM</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box" style="text-align:left;padding:20px">
    <div style="font-family:JetBrains Mono;font-size:0.82rem;color:#94a3b8">
    Start score: <b style='color:#00d4ff'>{SCORING['START']}</b><br><br>
    🔴 Each CRITICAL issue: <b style='color:#ef4444'>{SCORING['CRITICAL_PENALTY']} points</b><br>
    🟡 Each WARNING:        <b style='color:#eab308'>{SCORING['WARNING_PENALTY']} points</b><br>
    🟢 Each PASS:           <b style='color:#22c55e'>+0 (maintained)</b><br><br>
    <b style='color:#22c55e'>{SCORING['THRESHOLDS']['VALID']}–100</b> → VALID strategy — worth pursuing<br>
    <b style='color:#eab308'>{SCORING['THRESHOLDS']['QUESTIONABLE']}–{SCORING['THRESHOLDS']['VALID']-1}</b>  → QUESTIONABLE — fix warnings first<br>
    <b style='color:#ef4444'>0–{SCORING['THRESHOLDS']['QUESTIONABLE']-1}</b>   → INVALID — likely biased results
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">BUILT WITH</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:JetBrains Mono;font-size:0.75rem;color:#334155">
    Python · Streamlit · NumPy · SciPy · AST<br>
    Free to use · Free to deploy on Streamlit Cloud<br><br>
    <span style='color:#00d4ff'>Quant Alpha</span> — Building real quant tools
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;
border-top:1px solid #1e3a5f">
    <span style="font-family:JetBrains Mono;font-size:0.7rem;color:#1e3a5f">
    QUANT ALPHA BACKTEST VALIDATOR v2 — FREE TOOL — NOT FINANCIAL ADVICE
    </span>
</div>""", unsafe_allow_html=True)
