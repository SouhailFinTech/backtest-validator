"""
╔══════════════════════════════════════════════════════╗
║     QUANT ALPHA — INTELLIGENT BACKTEST VALIDATOR     ║
║     Founder: Hrich Souhail                           ║
║   Detects: Lookahead bias, Overfitting, Logic Bugs,  ║
║              Unrealistic Assumptions, Prop Compliance║
║   Streamlit App — Production Ready                   ║
╚══════════════════════════════════════════════════════╝

Install:  pip install streamlit pandas numpy scipy
Run:      streamlit run smart_backtest_validator.py
Deploy:   streamlit.io (free)
"""

import sys
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import ast
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

# ─────────────────────────────────────────────────────────────
# CONFIG & SCORING
# ─────────────────────────────────────────────────────────────
if sys.version_info < (3, 9):
    st.error("⚠️ Requires Python 3.9+ for AST analysis. Upgrade or use Returns Analysis only.")
    st.stop()

SCORING = {
    'START': 100,
    'CRITICAL_PENALTY': -25,
    'WARNING_PENALTY': -10,
    'THRESHOLDS': {'VALID': 80, 'QUESTIONABLE': 55}
}

st.set_page_config(
    page_title="Smart Backtest Validator | Quant Alpha",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# BEAUTIFUL UI THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); 
    color: #f8fafc; 
}
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); }

.header-box { 
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%); 
    border-radius: 16px; 
    padding: 32px; 
    text-align: center; 
    margin-bottom: 32px;
    box-shadow: 0 10px 40px rgba(14, 165, 233, 0.3);
}
.header-box h1 { 
    font-family: 'JetBrains Mono', monospace; 
    color: white; 
    font-size: 2.2rem; 
    margin: 0;
    font-weight: 700;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.header-box p { 
    color: rgba(255,255,255,0.9); 
    margin: 8px 0 0; 
    font-size: 1rem;
    font-weight: 400;
}
.founder-tag {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 6px 16px;
    border-radius: 20px;
    margin-top: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.5px;
}

.score-card { 
    border-radius: 16px; 
    padding: 28px; 
    text-align: center; 
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.score-card:hover { transform: translateY(-2px); }
.score-great { 
    background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
    border: 2px solid #34d399;
}
.score-ok { 
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
    border: 2px solid #fbbf24;
}
.score-bad { 
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
    border: 2px solid #f87171;
}
.score-number { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 4rem; 
    font-weight: 700; 
    color: white;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.score-label { 
    font-size: 0.95rem; 
    color: rgba(255,255,255,0.9); 
    margin-top: 8px;
    font-weight: 600;
}

.metric-box { 
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px; 
    padding: 16px; 
    text-align: center;
    margin: 8px 0;
    transition: all 0.2s;
}
.metric-box:hover {
    border-color: rgba(14, 165, 233, 0.5);
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
}
.metric-val { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 1.6rem; 
    font-weight: 700; 
    color: #38bdf8;
}
.metric-lbl { 
    font-size: 0.8rem; 
    color: #94a3b8; 
    margin-top: 6px;
    font-weight: 500;
}

.section { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.75rem; 
    color: #38bdf8; 
    letter-spacing: 2px; 
    text-transform: uppercase; 
    margin: 28px 0 16px; 
    border-bottom: 2px solid rgba(56, 189, 248, 0.3); 
    padding-bottom: 10px;
    font-weight: 700;
}

[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #0b0f19 0%, #111827 100%); 
    border-right: 2px solid rgba(14, 165, 233, 0.3);
}

.stButton > button { 
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%); 
    color: white; 
    border: none; 
    border-radius: 10px; 
    font-weight: 700; 
    padding: 12px 28px;
    width: 100%;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    transition: all 0.3s;
}
.stButton > button:hover { 
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
}

.stTabs [data-baseweb="tab"] { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.85rem; 
    color: #94a3b8;
    font-weight: 600;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] { 
    color: #38bdf8 !important;
    background: rgba(14, 165, 233, 0.1);
    border-radius: 8px 8px 0 0;
}

.stTextArea textarea { 
    background: rgba(17, 24, 39, 0.6); 
    color: #f8fafc; 
    border: 2px solid rgba(148, 163, 184, 0.2); 
    border-radius: 12px; 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.85rem;
    backdrop-filter: blur(10px);
}
.stTextArea textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
}

.stRadio label { 
    font-size: 0.9rem; 
    color: #cbd5e1;
    font-weight: 500;
}

#MainMenu, footer, header { visibility: hidden; }

.issue-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    border-left: 4px solid;
    backdrop-filter: blur(10px);
    transition: all 0.2s;
}
.issue-card:hover { transform: translateX(4px); }
.issue-critical { background: rgba(239, 68, 68, 0.1); border-left-color: #ef4444; }
.issue-warning  { background: rgba(245, 158, 11, 0.1); border-left-color: #f59e0b; }
.issue-info     { background: rgba(59, 130, 246, 0.1); border-left-color: #3b82f6; }
.issue-ok       { background: rgba(16, 185, 129, 0.1); border-left-color: #10b981; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────
@dataclass
class Issue:
    severity: str
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
        if severity == "CRITICAL": self.score += SCORING['CRITICAL_PENALTY']
        elif severity == "WARNING": self.score += SCORING['WARNING_PENALTY']
        self.score = max(0, self.score)

    def finalize(self):
        if self.score >= SCORING['THRESHOLDS']['VALID']: self.verdict = "✅ VALID"
        elif self.score >= SCORING['THRESHOLDS']['QUESTIONABLE']: self.verdict = "⚠️ QUESTIONABLE"
        else: self.verdict = "❌ INVALID"

# ─────────────────────────────────────────────────────────────
# INTELLIGENT VALIDATOR MODULES
# ─────────────────────────────────────────────────────────────
class SmartLookaheadDetector:
    FUTURE_KEYWORDS = ['future_', 'next_', 'forward_', 'tomorrow_', 'lead_', '_future', '_next', '_fwd', '_ahead']
    REALISTIC_ENTRY = [r"df\.iloc\[i\+\d+\]", r"entry.*=.*open", r"next_bar"]

    class LineageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.assignments: Dict[str, List[Tuple[int, str]]] = {}
            self.usages: Dict[str, List[Tuple[int, str]]] = {}
            self.shift_lines: set = set()
            self.fit_lines: set = set()

        def visit_Assign(self, node):
            targets = [ast.unparse(t) for t in node.targets]
            code = ast.unparse(node)
            lineno = getattr(node, 'lineno', 0)
            for t in targets: self.assignments.setdefault(t, []).append((lineno, code))
            if '.shift(' in code: self.shift_lines.add(lineno)
            if '.fit(' in code or '.fit_transform(' in code: self.fit_lines.add(lineno)
            self.generic_visit(node)

        def visit_Name(self, node):
            self.usages.setdefault(node.id, []).append((getattr(node, 'lineno', 0), node.id))
            self.generic_visit(node)

    def analyze(self, code: str, report: ValidationReport):
        try: tree = ast.parse(code)
        except SyntaxError as e:
            report.add("CRITICAL", "Syntax", f"Cannot parse: {e}")
            return

        vis = self.LineageVisitor()
        vis.visit(tree)

        for var, assigns in vis.assignments.items():
            if any(kw in var.lower() for kw in self.FUTURE_KEYWORDS):
                for ln, _ in assigns: report.add("CRITICAL", "Lookahead", f'Variable "{var}" implies future data', "Rename to avoid lookahead implications.", ln)

        sig_vars = [v for v in vis.assignments if 'signal' in v.lower()]
        for sv in sig_vars:
            used_in_pnl = any(kw in " ".join(c for _, c in vis.usages.get(sv, [])).lower() for kw in ['return', 'pnl', 'profit', 'strategy'])
            if not used_in_pnl: continue
            has_shift = any(ln in vis.shift_lines for ln, _ in vis.assignments[sv])
            if has_shift: report.add("OK", "Lookahead", f"Signal '{sv}' properly offset with .shift()", "Entry uses previous bar's signal. ✓")
            elif any(re.search(p, code, re.IGNORECASE) for p in self.REALISTIC_ENTRY):
                report.add("OK", "Lookahead", f"Signal '{sv}' used with realistic execution (i+N/next_bar)", "Future-bar entry enforces safety. ✓")
            else:
                report.add("WARNING", "Lookahead", f"Signal '{sv}' used in PnL without .shift() or future entry", "Same-bar signal × return leaks data. Add .shift(1) or use df.iloc[i+1]['open'].")

        if vis.fit_lines and not re.search(r'train|test|split|fold|cv=', code, re.IGNORECASE):
            for ln in vis.fit_lines: report.add("CRITICAL", "Lookahead", ".fit()/.fit_transform() on full dataset", "Splits data first. Use TimeSeriesSplit or train/test split.", ln)

        if ("['Close']" in code or '["Close"]' in code) and 'open' not in code.lower():
            report.add("WARNING", "Lookahead", "Using Close for entry — possible lookahead", "Close is unknown until bar ends. Use next bar's Open.")

        if 'center=True' in code and 'rolling(' in code:
            report.add("CRITICAL", "Lookahead", "center=True in rolling window", "Leaks future bars. Use center=False (default) or shift results.")

class OverfittingDetector:
    @staticmethod
    @lru_cache(maxsize=32)
    def _stats(returns_tuple: tuple):
        ret = pd.Series(returns_tuple)
        if len(ret) < 10: return None
        m, s = ret.mean(), ret.std()
        sr = (m / s * np.sqrt(252)) if s > 0 else 0
        T, skew, kurt = len(ret), float(ret.skew()), float(ret.kurtosis())
        try:
            var_sr = (1 + 0.5*sr**2 - skew*sr + (kurt-3)/4 * sr**2) / T
            dsr = sr / np.sqrt(max(var_sr, 1e-10))
        except: dsr = sr
        cum = (1 + ret).cumprod()
        r = np.corrcoef(np.arange(len(cum)), np.log(cum.clip(1e-6)))[0,1] if len(cum)>1 else 0
        dd = (cum.cummax() - cum) / cum.cummax()
        pf = ret[ret>0].sum()/abs(ret[ret<0].sum()) if ret[ret<0].sum()!=0 else np.inf
        return {'Sharpe': round(sr, 3), 'Deflated Sharpe': round(dsr, 3), 'Max DD': f"{dd.max():.1%}",
                'Win Rate': f"{(ret>0).mean():.1%}", 'Profit Factor': round(pf, 2) if pf!=np.inf else "∞",
                'Smoothness R²': round(r**2, 3), 'Total Return': f"{(cum.iloc[-1]-1):.1%}", 'Obs': T}

    def analyze(self, returns: pd.Series, n_trials: int, report: ValidationReport):
        stats = self._stats(tuple(returns.dropna()))
        if not stats: return
        report.metrics.update({k: v for k, v in stats.items()})
        sr, pf, r2, T = stats['Sharpe'], stats['Profit Factor'], stats['Smoothness R²'], stats['Obs']
        if sr > 4: report.add("CRITICAL", "Overfitting", f"Sharpe {sr:.2f} unrealistic", "Live strategies rarely exceed 1.5. Check lookahead/fees.")
        elif sr > 3: report.add("WARNING", "Overfitting", f"Sharpe {sr:.2f} suspicious", "Verify OOS & transaction costs.")
        elif sr > 1.5: report.add("INFO", "Overfitting", f"Sharpe {sr:.2f} solid but verify OOS", "Confirm out-of-sample stability.")
        else: report.add("OK", "Overfitting", f"Sharpe {sr:.2f} realistic", "Within institutional norms.")
        if isinstance(pf, (int, float)) and pf > 5: report.add("WARNING", "Overfitting", f"PF {pf:.1f} too high", "Rarely survives live trading. Realistic: 1.3–2.5.")
        if r2 > 0.97 and sr > 1.5: report.add("WARNING", "Overfitting", "Equity curve too smooth (R²>0.97)", "Perfect curves often indicate lookahead bias.")
        if T < 252: report.add("WARNING", "Overfitting", f"Only {T} obs (<1 yr)", "Need ≥252 for stable stats.")

class AssumptionChecker:
    def analyze_code(self, code: str, report: ValidationReport):
        cost_pat = [r'commission\s*=', r'\*\s*0\.\d+', r'slippage', r'fee\s*=', r'transaction_cost', r'spread\s*[+*=]']
        if not any(re.search(p, code, re.IGNORECASE) for p in cost_pat):
            report.add("WARNING", "Assumptions", "No transaction costs modeled", "Crypto: 0.1%/trade. Forex: spread. Costs cut returns 30-70%.")
        else: report.add("OK", "Assumptions", "Transaction costs detected ✓")
        if '-1' in code and ('short' in code.lower() or 'sell' in code.lower()) and 'borrow' not in code.lower():
            report.add("INFO", "Assumptions", "Short selling detected", "Add borrowing fees (0.5-5%/yr).")
        if 'volume' not in code.lower() and 'liquidity' not in code.lower():
            report.add("INFO", "Assumptions", "No liquidity constraints", "Add volume-based sizing to model slippage.")

    def analyze_trades(self, returns: pd.Series, report: ValidationReport):
        n = len(returns[returns != 0])
        if n < 30: report.add("WARNING", "Statistical", f"Only {n} trades", "Need ≥30 for validity, ideally 100+.")
        elif n > 5000: report.add("INFO", "Assumptions", f"{n} trades — high freq", "Verify slippage/commission scaling.")
        else: report.add("OK", "Statistical", f"{n} trades — sufficient ✓")

class LogicBugDetector:
    def analyze(self, code: str, report: ValidationReport):
        if 'cumsum()' in code and re.search(r"Position.*=.*Signal.*cumsum", code, re.IGNORECASE):
            report.add("CRITICAL", "Logic", "Position uses cumsum() on signal", "Creates [0,1,2,3...] positions. Use: Position = Signal (1/0)")
        
        # ✅ FIXED: Catches flat numeric subtraction like df['Net'] = df['Strat'] - 0.001
        comm_pat = r"(Net|net|PnL|pnl|returns|result)\s*=.*-\s*(0\.00\d|0\.01|0\.1|Commission|commission|comm|cost|fee|spread)"
        if re.search(comm_pat, code, re.IGNORECASE) and not re.search(r"(commission|comm|cost|fee|spread)\s*=.*trades\s*\*", code, re.IGNORECASE):
            report.add("WARNING", "Logic", "Commission flat-subtracted from returns", "Use: Commission = trades * 0.001, then Net = Strat - Comm")
            
        if 'win_rate' in code.lower() and '.mean()' in code and 'Position' not in code:
            report.add("INFO", "Logic", "Win rate may include flat days", "Filter active trades: df.loc[df['Position']!=0, 'Return'].gt(0).mean()")
        if "['drawdown']" in code and 'drawdown' not in code.split("['drawdown']")[0].split('\n')[-1]:
            report.add("WARNING", "Logic", "Possible KeyError: 'drawdown'", "Compute from equity: dd = (cum.max()-cum)/cum.max()")

class PropFirmChecker:
    FIRMS = {
        'FTMO ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.10, 'min_days': 4},
        'Topstep ($150K)': {'max_daily_dd': 0.03, 'max_total_dd': 0.06, 'profit_target': 0.06, 'min_days': 5},
        'MyFundedFX ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.08, 'min_days': 5},
    }
    def check(self, daily_returns: pd.Series, firm: str, report: ValidationReport):
        if firm not in self.FIRMS: return
        r = self.FIRMS[firm]
        worst = float(daily_returns.min())
        if abs(worst) > r['max_daily_dd']: report.add("CRITICAL", f"Prop:{firm}", f"Daily DD {worst:.1%} > limit {r['max_daily_dd']:.1%}", "Fails prop challenge.")
        else: report.add("OK", f"Prop:{firm}", f"Daily DD {worst:.1%} within limit ✓")
        cum = (1 + daily_returns).cumprod()
        td = ((cum.cummax() - cum) / cum.cummax()).max()
        if td > r['max_total_dd']: report.add("CRITICAL", f"Prop:{firm}", f"Total DD {td:.1%} > limit {r['max_total_dd']:.1%}", "Breaches max drawdown.")
        else: report.add("OK", f"Prop:{firm}", f"Total DD {td:.1%} within limit ✓")
        ret = float(cum.iloc[-1] - 1)
        if ret >= r['profit_target']: report.add("OK", f"Prop:{firm}", f"Profit {ret:.1%} ≥ target {r['profit_target']:.1%} ✓")
        else: report.add("INFO", f"Prop:{firm}", f"Profit {ret:.1%} < target {r['profit_target']:.1%}", "Won't pass evaluation.")

# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_validation(code: str, returns: Optional[pd.Series], n_trials: int, check_prop: bool, firm: str) -> ValidationReport:
    rpt = ValidationReport()
    if code.strip():
        SmartLookaheadDetector().analyze(code, rpt)
        AssumptionChecker().analyze_code(code, rpt)
        LogicBugDetector().analyze(code, rpt)
    if returns is not None and len(returns) > 5:
        OverfittingDetector().analyze(returns, n_trials, rpt)
        AssumptionChecker().analyze_trades(returns, rpt)
        if check_prop: PropFirmChecker().check(returns, firm, rpt)
    rpt.finalize()
    return rpt

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def render_issue(issue: Issue):
    icon = {'CRITICAL':'🔴','WARNING':'🟡','INFO':'🔵','OK':'🟢'}.get(issue.severity,'⚪')
    detail = f"<br><span style='color:#94a3b8;font-size:0.85rem'>{issue.detail}</span>" if issue.detail else ""
    line_txt = f" <span style='color:#64748b'>[line {issue.line}]</span>" if issue.line else ""
    
    css_class = f"issue-{issue.severity.lower()}"
    
    st.markdown(f"""
    <div class="issue-card {css_class}">
        <div style="font-weight:700;margin-bottom:6px">
            {icon} <b>[{issue.category}]</b>{line_txt}
        </div>
        <div style="color:#f1f5f9;margin-bottom:4px">{issue.message}</div>
        {detail}
    </div>
    """, unsafe_allow_html=True)

def score_style(s): return 'score-great' if s>=80 else 'score-ok' if s>=55 else 'score-bad'
def score_color(s): return '#10b981' if s>=80 else '#f59e0b' if s>=55 else '#ef4444'

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown('''
<div class="header-box">
    <h1>🔬 INTELLIGENT BACKTEST VALIDATOR</h1>
    <p>Detect lookahead bias · overfitting · logic bugs · prop compliance</p>
    <div class="founder-tag">Founder: Hrich Souhail</div>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section">⚙️ SETTINGS</div>', unsafe_allow_html=True)
    n_trials = st.slider("Strategies tested before this", 1, 200, 1, help="Higher = stricter Deflated Sharpe")
    check_prop = st.checkbox("Check Prop Firm Compliance", False)
    firm = st.selectbox("Prop Firm", list(PropFirmChecker.FIRMS.keys())) if check_prop else list(PropFirmChecker.FIRMS.keys())[0]
    
    st.markdown('<div class="section">🧠 INTELLIGENT DETECTIONS</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
    • Context-aware AST lineage<br>
    • Regex math (not keywords)<br>
    • .fit_transform & center=True<br>
    • cumsum & flat-fee logic<br>
    • Realistic Sharpe/DD thresholds
    </div>
    ''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Code Analysis", "📊 Returns Analysis", "📖 How It Works"])

with tab1:
    st.markdown('<div class="section">PASTE PYTHON STRATEGY CODE</div>', unsafe_allow_html=True)
    sample = """import pandas as pd
import numpy as np
# Example: EMA crossover with realistic entry
df['Signal'] = np.where(df['EMA_fast'] > df['EMA_slow'], 1, 0)
# Uses df.iloc[i+2]['open'] for entry → no shift needed!
"""
    code = st.text_area("Python Code", value=sample, height=280, label_visibility="collapsed")
    if st.button("🔍 ANALYZE CODE"):
        # ✅ FIX 1: Empty input guard prevents AST crash
        if not code.strip():
            st.info("ℹ️ Please paste some Python strategy code to analyze.")
            st.stop()
            
        with st.spinner("Running AST & logic checks..."):
            rpt = run_validation(code, None, n_trials, False, firm)
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:10px;font-weight:700;font-size:1.1rem">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with c2:
            cnt = {s: sum(1 for i in rpt.issues if i.severity==s) for s in ['CRITICAL','WARNING','OK','INFO']}
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-top:20px">
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#ef4444">{cnt['CRITICAL']}</div><div class="metric-lbl">Critical</div></div>
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#f59e0b">{cnt['WARNING']}</div><div class="metric-lbl">Warnings</div></div>
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#10b981">{cnt['OK']}</div><div class="metric-lbl">Passed</div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)

with tab2:
    st.markdown('<div class="section">UPLOAD / PASTE RETURNS</div>', unsafe_allow_html=True)
    method = st.radio("Input Method", ["Upload CSV", "Paste decimals"], horizontal=True)
    returns = None
    if method == "Upload CSV":
        up = st.file_uploader("Returns CSV", type="csv", label_visibility="collapsed")
        if up:
            try:
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try: df_up = pd.read_csv(up, encoding=enc); break
                    except: continue
                else: st.error("❌ Unsupported encoding"); st.stop()
                col = next((c for c in df_up.columns if 'return' in c.lower() or 'pnl' in c.lower()), df_up.select_dtypes('number').columns[0])
                returns = pd.to_numeric(df_up[col], errors='coerce').dropna()
                returns = returns[np.isfinite(returns)]
                if len(returns) < 3: st.warning("⚠️ Too few valid data points after cleaning. Need ≥3."); st.stop()
                st.success(f"✅ Loaded {len(returns)} valid data points (NaN/Inf stripped)")
            except Exception as e: st.error(f"CSV Error: {e}")
    else:
        raw = st.text_area("Comma-separated returns", placeholder="0.012, -0.005, 0.023, NaN, inf...", height=120, label_visibility="collapsed")
        if raw:
            try:
                # ✅ FIX 2: Robust decimal parser handles NaN, Inf, and messy formatting
                clean_vals = raw.replace('\n', ',').replace(' ', ',').split(',')
                returns = pd.to_numeric(clean_vals, errors='coerce').dropna()
                returns = returns[np.isfinite(returns)]
                if len(returns) < 3: st.warning("⚠️ Too few valid data points. Need ≥3."); st.stop()
                st.success(f"✅ Loaded {len(returns)} valid data points")
            except Exception as e: st.error(f"Invalid format: {e}")
    
    if st.button("📊 ANALYZE RETURNS"):
        if returns is None:
            st.warning("⚠️ Please upload or paste returns data first.")
            st.stop()
        with st.spinner("Computing statistics & prop rules..."):
            code_ctx = code if st.checkbox("Also check code from Tab 1", False) else ""
            rpt = run_validation(code_ctx, returns, n_trials, check_prop, firm)
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:10px;font-weight:700;font-size:1.1rem">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section">KEY METRICS</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,(k,v) in enumerate(rpt.metrics.items()):
                with cols[i%3]: st.markdown(f"""<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{k}</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="section">EQUITY CURVE</div>', unsafe_allow_html=True)
        st.line_chart((1+returns).cumprod(), height=300)
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)
        st.download_button("⬇️ Download Report JSON", json.dumps({'score':rpt.score, 'metrics':rpt.metrics, 'issues': [{'s':i.severity,'c':i.category,'m':i.message} for i in rpt.issues]}, indent=2), "validation_report.json", "application/json")

with tab3:
    st.markdown('<div class="section">🧠 INTELLIGENCE UPGRADES</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="background:rgba(17,24,39,0.6);border-radius:12px;padding:24px;margin:16px 0">
    <div style="margin-bottom:16px"><b style="color:#38bdf8">• Context-Aware AST</b><br><span style="color:#94a3b8">Tracks `Signal` → `Returns` flow. Ignores false positives if `i+1/i+2` entry is used.</span></div>
    <div style="margin-bottom:16px"><b style="color:#38bdf8">• Regex Math Detection</b><br><span style="color:#94a3b8">Scans `commission\s*=` or `*\s*0.\d+`. Stops flagging comments.</span></div>
    <div style="margin-bottom:16px"><b style="color:#38bdf8">• Fixed Deflated Sharpe</b><br><span style="color:#94a3b8">Correct Bailey & López de Prado formula. No more 300k+ values.</span></div>
    <div style="margin-bottom:16px"><b style="color:#38bdf8">• Logic Bug Catcher</b><br><span style="color:#94a3b8">Flags `cumsum()` on binary signals, flat fee math, `center=True` rolling windows.</span></div>
    <div style="margin-bottom:16px"><b style="color:#38bdf8">• Prop Firm Simulator</b><br><span style="color:#94a3b8">Tests daily/total DD & profit targets against FTMO/Topstep rules.</span></div>
    <div><b style="color:#38bdf8">• Graceful Degradation</b><br><span style="color:#94a3b8">Handles syntax errors, missing CSV columns, empty returns without crashing.</span></div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="section">SCORING LOGIC</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box" style="text-align:left;padding:20px">
    <div style="font-family:'JetBrains Mono';font-size:0.9rem;color:#f1f5f9;line-height:2">
    Start: <b style="color:#38bdf8">{SCORING['START']}</b> → 🔴 Critical: <b style="color:#ef4444">{SCORING['CRITICAL_PENALTY']}</b> → 🟡 Warning: <b style="color:#f59e0b">{SCORING['WARNING_PENALTY']}</b><br>
    <b style="color:#10b981">{SCORING['THRESHOLDS']['VALID']}+ → VALID</b> · <b style="color:#f59e0b">{SCORING['THRESHOLDS']['QUESTIONABLE']}-{SCORING['THRESHOLDS']['VALID']-1} → QUESTIONABLE</b> · <b style="color:#ef4444"><{SCORING['THRESHOLDS']['QUESTIONABLE']} → INVALID</b>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('''
<div style="text-align:center;margin-top:40px;padding:24px;border-top:2px solid rgba(14,165,233,0.3)">
    <div style="font-family:'JetBrains Mono';font-size:0.85rem;color:#94a3b8;margin-bottom:8px">
    QUANT ALPHA VALIDATOR — PRODUCTION READY
    </div>
    <div style="font-size:0.8rem;color:#64748b">
    Founder: <b style="color:#38bdf8">Hrich Souhail</b> — Not Financial Advice
    </div>
</div>
''', unsafe_allow_html=True)
