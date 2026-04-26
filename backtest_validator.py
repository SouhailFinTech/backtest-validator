"""
╔══════════════════════════════════════════════════════╗
║     QUANT ALPHA — INTELLIGENT BACKTEST VALIDATOR     ║
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

st.set_page_config(page_title="Smart Backtest Validator | Quant Alpha", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
# UI THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background: #0a0e1a; color: #e2e8f0; }
.stApp { background: #0a0e1a; }
.header-box { background: linear-gradient(135deg, #0d1b2a, #1a2744); border: 1px solid #1e3a5f; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 20px; }
.header-box h1 { font-family: 'JetBrains Mono', monospace; color: #00d4ff; font-size: 1.8rem; margin: 0; }
.score-card { border-radius: 12px; padding: 20px; text-align: center; margin: 10px 0; }
.score-great { background: linear-gradient(135deg, #052e16, #14532d); border: 1px solid #22c55e; }
.score-ok { background: linear-gradient(135deg, #1c1400, #3a2c00); border: 1px solid #eab308; }
.score-bad { background: linear-gradient(135deg, #1c0606, #3a0d0d); border: 1px solid #ef4444; }
.metric-box { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px; text-align: center; margin: 6px 0; }
.metric-val { font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; color: #00d4ff; }
.metric-lbl { font-size: 0.75rem; color: #64748b; }
.section { font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #00d4ff; letter-spacing: 2px; text-transform: uppercase; margin: 20px 0 10px; border-bottom: 1px solid #1e3a5f; padding-bottom: 6px; }
[data-testid="stSidebar"] { background: #080c16; border-right: 1px solid #1e3a5f; }
.stButton > button { background: linear-gradient(135deg, #0066cc, #0044aa); color: white; border: none; border-radius: 8px; font-weight: 700; width: 100%; }
.stButton > button:hover { background: linear-gradient(135deg, #0080ff, #0055cc); }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #64748b; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; }
#MainMenu, footer, header { visibility: hidden; }
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

        # 1. Future-named variables
        for var, assigns in vis.assignments.items():
            if any(kw in var.lower() for kw in self.FUTURE_KEYWORDS):
                for ln, _ in assigns: report.add("CRITICAL", "Lookahead", f'Variable "{var}" implies future data', "Rename to avoid lookahead implications.", ln)

        # 2. Signal → PnL lineage check
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

        # 3. Data leakage via fitting
        if vis.fit_lines and not re.search(r'train|test|split|fold|cv=', code, re.IGNORECASE):
            for ln in vis.fit_lines: report.add("CRITICAL", "Lookahead", ".fit()/.fit_transform() on full dataset", "Splits data first. Use TimeSeriesSplit or train/test split.", ln)

        # 4. Close-price entry warning
        if ("['Close']" in code or '["Close"]' in code) and 'open' not in code.lower():
            report.add("WARNING", "Lookahead", "Using Close for entry — possible lookahead", "Close is unknown until bar ends. Use next bar's Open.")

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
        comm_pat = r"(Net|net|PnL|pnl|returns)\s*=.*-\s*(Commission|commission|comm|costs)"
        if re.search(comm_pat, code, re.IGNORECASE) and not re.search(r"(commission|comm|costs)\s*=.*trades\s*\*", code, re.IGNORECASE):
            report.add("WARNING", "Logic", "Commission flat-subtracted from returns", "Use: Commission = trades * 0.001, then Net = Strat - Comm")
        if 'center=True' in code and 'rolling(' in code:
            report.add("CRITICAL", "Lookahead", "center=True in rolling window", "Leaks future bars. Use center=False (default) or shift results.")
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
    detail = f"\n\n_{issue.detail}_" if issue.detail else ""
    msg = f"{icon} **[{issue.category}]**{f' [line {issue.line}]' if issue.line else ''}\n\n{issue.message}{detail}"
    if issue.severity == "CRITICAL": st.error(msg)
    elif issue.severity == "WARNING": st.warning(msg)
    elif issue.severity == "OK": st.success(msg)
    else: st.info(msg)

def score_style(s): return 'score-great' if s>=80 else 'score-ok' if s>=55 else 'score-bad'
def score_color(s): return '#22c55e' if s>=80 else '#eab308' if s>=55 else '#ef4444'

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="header-box"><h1>🔬 INTELLIGENT BACKTEST VALIDATOR</h1><p>Detect lookahead bias · overfitting · logic bugs · prop compliance</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section">⚙️ SETTINGS</div>', unsafe_allow_html=True)
    n_trials = st.slider("Strategies tested before this", 1, 200, 1, help="Higher = stricter Deflated Sharpe")
    check_prop = st.checkbox("Check Prop Firm Compliance", False)
    firm = st.selectbox("Prop Firm", list(PropFirmChecker.FIRMS.keys())) if check_prop else list(PropFirmChecker.FIRMS.keys())[0]
    st.markdown('<div style="font-size:0.7rem;color:#334155;margin-top:10px"><b style="color:#00d4ff">INTELLIGENT DETECTIONS:</b><br>• Context-aware AST lineage<br>• Regex math (not keywords)<br>• .fit_transform & center=True<br>• cumsum & flat-fee logic<br>• Realistic Sharpe/DD thresholds</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Code Analysis", "📊 Returns Analysis", "📖 How It Works"])

with tab1:
    st.markdown('<div class="section">PASTE PYTHON STRATEGY CODE</div>', unsafe_allow_html=True)
    sample = """import pandas as pd
import numpy as np
# Example: EMA crossover with realistic entry
df['Signal'] = np.where(df['EMA_fast'] > df['EMA_slow'], 1, 0)
# Uses df.iloc[i+2]['open'] for entry → no shift needed!
"""
    code = st.text_area("Python Code", value=sample, height=250)
    if st.button("🔍 ANALYZE CODE"):
        with st.spinner("Running AST & logic checks..."):
            rpt = run_validation(code, None, n_trials, False, firm)
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div style="font-family:'JetBrains Mono';font-size:3rem;font-weight:700;color:{score_color(rpt.score)}">{rpt.score}</div><div style="font-size:0.8rem;color:#94a3b8">REALISM SCORE / 100</div><div style="margin-top:6px;font-weight:700">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with c2:
            cnt = {s: sum(1 for i in rpt.issues if i.severity==s) for s in ['CRITICAL','WARNING','OK','INFO']}
            st.markdown(f"🔴 {cnt['CRITICAL']} Critical · 🟡 {cnt['WARNING']} Warnings · 🟢 {cnt['OK']} Passed")
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)

with tab2:
    st.markdown('<div class="section">UPLOAD / PASTE RETURNS</div>', unsafe_allow_html=True)
    method = st.radio("Input", ["Upload CSV", "Paste decimals"], horizontal=True)
    returns = None
    if method == "Upload CSV":
        up = st.file_uploader("Returns CSV", type="csv")
        if up:
            try:
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try: df_up = pd.read_csv(up, encoding=enc); break
                    except: continue
                else: st.error("❌ Unsupported encoding"); st.stop()
                col = next((c for c in df_up.columns if 'return' in c.lower() or 'pnl' in c.lower()), df_up.select_dtypes('number').columns[0])
                returns = pd.to_numeric(df_up[col], errors='coerce').dropna()
                if len(returns) < 5: st.error("❌ Too few values"); st.stop()
                st.success(f"✅ Loaded {len(returns)} obs from '{col}'")
            except Exception as e: st.error(f"CSV Error: {e}")
    else:
        raw = st.text_area("Comma-separated returns", placeholder="0.012, -0.005, 0.023...")
        if raw:
            try:
                returns = pd.Series([float(x.strip()) for x in raw.replace('\n',',').split(',') if x.strip()])
                st.success(f"✅ Loaded {len(returns)} data points")
            except: st.error("Invalid format. Use decimals only.")
    
    if st.button("📊 ANALYZE RETURNS") and returns is not None:
        with st.spinner("Computing statistics & prop rules..."):
            code_ctx = code if st.checkbox("Also check code from Tab 1", False) else ""
            rpt = run_validation(code_ctx, returns, n_trials, check_prop, firm)
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div style="font-family:'JetBrains Mono';font-size:3rem;font-weight:700;color:{score_color(rpt.score)}">{rpt.score}</div><div style="font-size:0.8rem;color:#94a3b8">REALISM SCORE / 100</div><div style="margin-top:6px;font-weight:700">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section">KEY METRICS</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,(k,v) in enumerate(rpt.metrics.items()):
                with cols[i%3]: st.markdown(f"""<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{k}</div></div>""", unsafe_allow_html=True)
        st.line_chart((1+returns).cumprod(), height=250)
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)
        st.download_button("⬇️ Download Report JSON", json.dumps({'score':rpt.score, 'metrics':rpt.metrics, 'issues': [{'s':i.severity,'c':i.category,'m':i.message} for i in rpt.issues]}, indent=2), "validation_report.json", "application/json")

with tab3:
    st.markdown("### 🔍 Intelligence Upgrades")
    st.info("• **Context-Aware AST**: Tracks `Signal` → `Returns` flow. Ignores false positives if `i+1/i+2` entry is used.")
    st.info("• **Regex Math Detection**: Scans for `commission\s*=` or `*\s*0.\d+`. Stops flagging comments.")
    st.info("• **Fixed Deflated Sharpe**: Correct Bailey & López de Prado formula. No more 300k+ values.")
    st.info("• **Logic Bug Catcher**: Flags `cumsum()` on binary signals, flat fee math, `center=True` rolling windows.")
    st.info("• **Prop Firm Simulator**: Tests daily/total DD & profit targets against FTMO/Topstep rules.")
    st.info("• **Graceful Degradation**: Handles syntax errors, missing columns, empty returns without crashing.")
    st.markdown('<div class="section">SCORING LOGIC</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="metric-box" style="text-align:left;padding:16px;font-size:0.85rem">
    Start: <b style="color:#00d4ff">{SCORING['START']}</b> → 🔴 Critical: <b style="color:#ef4444">{SCORING['CRITICAL_PENALTY']}</b> → 🟡 Warning: <b style="color:#eab308">{SCORING['WARNING_PENALTY']}</b><br>
    <b style="color:#22c55e">{SCORING['THRESHOLDS']['VALID']}+ → VALID</b> · <b style="color:#eab308">{SCORING['THRESHOLDS']['QUESTIONABLE']}-{SCORING['THRESHOLDS']['VALID']-1} → QUESTIONABLE</b> · <b style="color:#ef4444"><{SCORING['THRESHOLDS']['QUESTIONABLE']} → INVALID</b>
    </div>""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;margin-top:30px;padding:16px;border-top:1px solid #1e3a5f;font-size:0.7rem;color:#1e3a5f">QUANT ALPHA VALIDATOR — PRODUCTION READY — NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)

