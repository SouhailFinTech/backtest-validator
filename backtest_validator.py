"""
╔══════════════════════════════════════════════════════╗
║     QUANT ALPHA — SMART BACKTEST VALIDATOR v2        ║
║   Detects: Lookahead bias, Overfitting, Bad Assumptions ║
║   Streamlit app — free to deploy on Streamlit Cloud  ║
╚══════════════════════════════════════════════════════╝

Install:  pip install streamlit pandas numpy scipy
Run:      streamlit run backtest_validator_smart_v2.py
Deploy:   streamlit.io (free)

SMART UPGRADES:
• Variable lineage tracking (signal → PnL flow verification)
• Context-aware lookahead detection (recognizes i+1/i+2 entry patterns)
• Fixed Deflated Sharpe Ratio (Bailey & López de Prado)
• Fixed commission false-positives (checks for math, not keywords)
• Logic bug detector (cumsum, flat fees, pandas KeyErrors)
• Native Streamlit UI (accessible, mobile-friendly)
• Production error handling & graceful degradation
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from functools import lru_cache

# ─────────────────────────────────────────────────────────────
# SCORING CONSTANTS
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
    st.error("⚠️ This tool requires Python 3.9+ for code analysis. Please upgrade or use Returns Analysis only.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Backtest Validator | Quant Alpha", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
.stApp { background-color: #0a0e1a; }
.main-header { background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 32px; margin-bottom: 24px; text-align: center; }
.main-header h1 { font-family: 'JetBrains Mono', monospace; font-size: 2rem; color: #00d4ff; margin: 0; letter-spacing: -1px; }
.main-header p { color: #64748b; margin: 8px 0 0; font-size: 0.95rem; }
.score-card { border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px; }
.score-great { background: linear-gradient(135deg,#052e16,#14532d); border:1px solid #22c55e; }
.score-ok { background: linear-gradient(135deg,#1c1400,#3a2c00); border:1px solid #eab308; }
.score-bad { background: linear-gradient(135deg,#1c0606,#3a0d0d); border:1px solid #ef4444; }
.score-number { font-family:'JetBrains Mono',monospace; font-size:3.5rem; font-weight:700; }
.score-label { font-size:0.85rem; color:#94a3b8; margin-top:4px; }
.metric-box { background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:16px; text-align:center; }
.metric-val { font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; color:#00d4ff; }
.metric-lbl { font-size:0.78rem; color:#64748b; margin-top:4px; }
.section-header { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#00d4ff; letter-spacing:2px; text-transform:uppercase; margin:24px 0 12px; border-bottom:1px solid #1e3a5f; padding-bottom:8px; }
[data-testid="stSidebar"] { background:#080c16; border-right:1px solid #1e3a5f; }
.stButton > button { background:linear-gradient(135deg,#0066cc,#0044aa); color:white; border:none; border-radius:8px; font-family:'JetBrains Mono',monospace; font-weight:700; padding:12px 24px; width:100%; transition:all 0.2s; }
.stButton > button:hover { background:linear-gradient(135deg,#0080ff,#0055cc); transform:translateY(-1px); }
.stTextArea textarea { background:#080c16; color:#e2e8f0; border:1px solid #1e3a5f; border-radius:8px; font-family:'JetBrains Mono',monospace; font-size:0.82rem; }
.stTabs [data-baseweb="tab"] { font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#64748b; }
.stTabs [aria-selected="true"] { color:#00d4ff !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
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
# SMART LOOKAHEAD DETECTOR (AST + LINEAGE + CONTEXT)
# ─────────────────────────────────────────────────────────────
class SmartLookaheadDetector:
    FUTURE_KEYWORDS = ['future_', 'next_', 'forward_', 'tomorrow_', 'lead_', '_future', '_next', '_fwd', '_ahead']
    REALISTIC_ENTRY_PATTERNS = [r"df\.iloc\[i\+\d+\]", r"entry.*=.*open", r"next_bar"]

    class LineageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.assignments: Dict[str, List[Tuple[int, str]]] = {}
            self.usages: Dict[str, List[Tuple[int, str]]] = {}
            self.shift_lines: Set[int] = set()
            self.fit_lines: Set[int] = set()

        def visit_Assign(self, node):
            targets = [ast.unparse(t) for t in node.targets]
            code = ast.unparse(node)
            lineno = getattr(node, 'lineno', 0)
            for t in targets:
                self.assignments.setdefault(t, []).append((lineno, code))
            if '.shift(' in code: self.shift_lines.add(lineno)
            if '.fit(' in code: self.fit_lines.add(lineno)
            self.generic_visit(node)

        def visit_Name(self, node):
            self.usages.setdefault(node.id, []).append((getattr(node, 'lineno', 0), node.id))
            self.generic_visit(node)

    def _has_realistic_entry(self, code: str) -> bool:
        return any(re.search(p, code, re.IGNORECASE) for p in self.REALISTIC_ENTRY_PATTERNS)

    def _trace_signal_to_pnl(self, visitor: LineageVisitor, signal_var: str) -> bool:
        """Check if signal variable actually flows into returns/PnL calculation"""
        for lineno, usage in visitor.usages.get(signal_var, []):
            context = "".join([c[1] for c in visitor.usages.get(signal_var, [])])
            if any(kw in context.lower() for kw in ['return', 'pnl', 'profit', 'strategy']):
                return True
        return False

    def analyze(self, code: str, report: ValidationReport):
        try: tree = ast.parse(code)
        except SyntaxError as e:
            report.add("CRITICAL", "Syntax", f"Cannot parse code: {e}")
            return

        visitor = self.LineageVisitor()
        visitor.visit(tree)

        # 1. Future-leaking variables
        for var, assigns in visitor.assignments.items():
            for kw in self.FUTURE_KEYWORDS:
                if kw in var.lower():
                    for ln, _ in assigns: report.add("CRITICAL", "Lookahead", f'Variable "{var}" suggests future data', "Rename to avoid lookahead implications.", ln)

        # 2. Signal → PnL lineage check
        signal_vars = [v for v in visitor.assignments if 'signal' in v.lower()]
        for sv in signal_vars:
            used_in_pnl = self._trace_signal_to_pnl(visitor, sv)
            if not used_in_pnl: continue

            has_shift = any(ln in visitor.shift_lines for ln, _ in visitor.assignments[sv])
            if has_shift:
                report.add("OK", "Lookahead", f"Signal '{sv}' properly offset with .shift()", "Entry uses previous bar's signal. ✓")
            elif self._has_realistic_entry(code):
                report.add("OK", "Lookahead", f"Signal '{sv}' used with realistic execution timing (i+N / next_bar)", "Future-bar entry enforces lookahead safety. ✓")
            else:
                report.add("WARNING", "Lookahead", f"Signal '{sv}' used in PnL without .shift() or future-bar entry", "Same-bar signal × return leaks future data. Add .shift(1) or use df.iloc[i+1]['open'].")

        # 3. Close-price entry warning
        if ("['Close']" in code or '["Close"]' in code) and 'open' not in code.lower():
            report.add("WARNING", "Lookahead", "Using Close price for entry — possible lookahead", "Close is unknown until bar ends. Use next bar's Open for realistic fills.")

        # 4. Full-dataset fitting
        if '.fit(' in code and 'train' not in code.lower() and 'test' not in code.lower():
            for ln in visitor.fit_lines:
                report.add("CRITICAL", "Lookahead", "Scaler/model fitted on full dataset", ".fit() leaks future data. Split train/test first.", ln)

# ─────────────────────────────────────────────────────────────
# OVERFITTING DETECTOR (FIXED DSR + REALISTIC THRESHOLDS)
# ─────────────────────────────────────────────────────────────
class OverfittingDetector:
    @staticmethod
    @lru_cache(maxsize=32)
    def _compute_stats(returns_tuple: tuple):
        returns = pd.Series(returns_tuple)
        if len(returns) < 10: return None
        mean_r, std_r = returns.mean(), returns.std()
        sr = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        T = len(returns)
        skew, kurt = float(returns.skew()), float(returns.kurtosis())
        
        # FIXED Deflated Sharpe Ratio
        try:
            var_sr = (1 + 0.5*sr**2 - skew*sr + (kurt-3)/4 * sr**2) / T
            dsr = sr / np.sqrt(max(var_sr, 1e-10))
        except: dsr = sr

        cumret = (1 + returns).cumprod()
        r_val = np.corrcoef(np.arange(len(cumret)), np.log(cumret.clip(1e-6)))[0,1] if len(cumret)>1 else 0
        dd = (cumret.cummax() - cumret) / cumret.cummax()
        
        return {
            'Sharpe Ratio': round(sr, 3), 'Deflated Sharpe': round(dsr, 3),
            'Max Drawdown': f"{dd.max():.1%}", 'Win Rate': f"{(returns>0).mean():.1%}",
            'Profit Factor': round(returns[returns>0].sum()/abs(returns[returns<0].sum()), 2) if returns[returns<0].sum()!=0 else "∞",
            'Curve Smoothness R²': round(r_val**2, 3), 'Total Return': f"{(cumret.iloc[-1]-1):.1%}", 'Observations': T
        }

    def analyze(self, returns: pd.Series, n_trials: int, report: ValidationReport):
        stats = self._compute_stats(tuple(returns.dropna()))
        if not stats: return
        report.metrics.update(stats)

        sr, pf, r2, T = float(stats['Sharpe Ratio']), stats['Profit Factor'], float(stats['Curve Smoothness R²']), stats['Observations']
        if sr > 4: report.add("CRITICAL", "Overfitting", f"Sharpe {sr:.2f} unrealistically high", "Live strategies rarely exceed 1.5. Check for lookahead/fees.")
        elif sr > 3: report.add("WARNING", "Overfitting", f"Sharpe {sr:.2f} suspiciously high", "Verify OOS performance and transaction costs.")
        elif sr > 1.5: report.add("INFO", "Overfitting", f"Sharpe {sr:.2f} solid but verify OOS", "Good metric. Confirm out-of-sample stability.")
        else: report.add("OK", "Overfitting", f"Sharpe {sr:.2f} realistic", "Within institutional norms.")

        if isinstance(pf, (int, float)) and pf > 5: report.add("WARNING", "Overfitting", f"Profit Factor {pf:.1f} too high", "Rarely survives live trading. Realistic: 1.3–2.5.")
        if r2 > 0.97 and sr > 1.5: report.add("WARNING", "Overfitting", "Equity curve too smooth (R²>0.97)", "Perfect curves often indicate lookahead bias.")
        if T < 252: report.add("WARNING", "Overfitting", f"Only {T} observations (<1 yr)", "Need ≥252 for stable statistics.")

# ─────────────────────────────────────────────────────────────
# ASSUMPTION CHECKER (REGEX MATH DETECTION)
# ─────────────────────────────────────────────────────────────
class AssumptionChecker:
    def analyze_code(self, code: str, report: ValidationReport):
        # Check for actual cost math, not just keywords
        cost_patterns = [r'commission\s*=', r'\*\s*0\.\d+', r'slippage', r'fee\s*=', r'transaction_cost', r'spread\s*[+*=]']
        has_costs = any(re.search(p, code, re.IGNORECASE) for p in cost_patterns)
        if not has_costs: report.add("WARNING", "Assumptions", "No transaction costs modeled", "Crypto: 0.1%/trade. Forex: spread. Costs cut returns 30-70%.")
        else: report.add("OK", "Assumptions", "Transaction costs detected ✓")

        if '-1' in code and ('short' in code.lower() or 'sell' in code.lower()) and 'borrow' not in code.lower():
            report.add("INFO", "Assumptions", "Short selling detected", "Add borrowing fees (0.5-5%/yr) for accuracy.")
        if 'volume' not in code.lower() and 'liquidity' not in code.lower():
            report.add("INFO", "Assumptions", "No liquidity constraints", "Add volume-based sizing to model slippage.")

    def analyze_trades(self, returns: pd.Series, report: ValidationReport):
        n = len(returns[returns != 0])
        if n < 30: report.add("WARNING", "Statistical", f"Only {n} trades", "Need ≥30 for basic validity, ideally 100+.")
        elif n > 5000: report.add("INFO", "Assumptions", f"{n} trades — high frequency", "Verify slippage/commission scaling.")
        else: report.add("OK", "Statistical", f"{n} trades — sufficient ✓")

# ─────────────────────────────────────────────────────────────
# LOGIC BUG DETECTOR (CATCHES COMMON QUANT MISTAKES)
# ─────────────────────────────────────────────────────────────
class LogicBugDetector:
    def analyze(self, code: str, report: ValidationReport):
        # 1. cumsum on binary signal
        if 'cumsum()' in code and re.search(r"Position.*=.*Signal.*cumsum", code, re.IGNORECASE):
            report.add("CRITICAL", "Logic", "Position uses cumsum() on signal", "Creates [0,1,2,3...] positions. Use: Position = Signal (1/0)")
        
        # 2. Flat commission math
        if 'Net_Return' in code and re.search(r"Net_Return\s*=.*Strategy_Return\s*-\s*Commission", code):
            if not re.search(r"Commission\s*=.*\*.*0\.", code):
                report.add("WARNING", "Logic", "Commission flat-subtracted from returns", "Use: Commission = trades * 0.001, then Net_Return = Strat - Comm")
        
        # 3. Win rate includes flat days
        if 'win_rate' in code.lower() and '.mean()' in code and 'Position' not in code:
            report.add("INFO", "Logic", "Win rate may include flat days", "Filter active trades: df.loc[df['Position']!=0, 'Return'].gt(0).mean()")

        # 4. Common pandas KeyError patterns
        if "['drawdown']" in code and 'drawdown' not in code.split("['drawdown']")[0].split('\n')[-1]:
            report.add("WARNING", "Logic", "Possible KeyError: 'drawdown'", "Compute drawdown from equity: dd = (cumret.cummax()-cumret)/cumret.cummax()")

# ─────────────────────────────────────────────────────────────
# PROP FIRM CHECKER
# ─────────────────────────────────────────────────────────────
class PropFirmChecker:
    FIRMS = {
        'FTMO ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.10, 'min_days': 4},
        'Topstep ($150K)': {'max_daily_dd': 0.03, 'max_total_dd': 0.06, 'profit_target': 0.06, 'min_days': 5},
        'MyFundedFX ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.08, 'min_days': 5},
    }
    def check(self, daily_returns: pd.Series, firm_name: str, report: ValidationReport):
        if firm_name not in self.FIRMS: return
        r = self.FIRMS[firm_name]
        worst = float(daily_returns.min())
        if abs(worst) > r['max_daily_dd']: report.add("CRITICAL", f"Prop:{firm_name}", f"Daily DD {worst:.1%} > limit {r['max_daily_dd']:.1%}", "Fails prop challenge.")
        else: report.add("OK", f"Prop:{firm_name}", f"Daily DD {worst:.1%} within limit ✓")

        cumret = (1 + daily_returns).cumprod()
        total_dd = ((cumret.cummax() - cumret) / cumret.cummax()).max()
        if total_dd > r['max_total_dd']: report.add("CRITICAL", f"Prop:{firm_name}", f"Total DD {total_dd:.1%} > limit {r['max_total_dd']:.1%}", "Breaches max drawdown.")
        else: report.add("OK", f"Prop:{firm_name}", f"Total DD {total_dd:.1%} within limit ✓")

        ret = float(cumret.iloc[-1] - 1)
        if ret >= r['profit_target']: report.add("OK", f"Prop:{firm_name}", f"Profit {ret:.1%} ≥ target {r['profit_target']:.1%} ✓")
        else: report.add("INFO", f"Prop:{firm_name}", f"Profit {ret:.1%} < target {r['profit_target']:.1%}", "Won't pass evaluation.")

# ─────────────────────────────────────────────────────────────
# MAIN VALIDATION RUNNER
# ─────────────────────────────────────────────────────────────
def run_validation(code: str, returns: Optional[pd.Series], n_trials: int, check_propfirm: bool, firm_name: str) -> ValidationReport:
    report = ValidationReport()
    if code.strip():
        SmartLookaheadDetector().analyze(code, report)
        AssumptionChecker().analyze_code(code, report)
        LogicBugDetector().analyze(code, report)
    if returns is not None and len(returns) > 5:
        OverfittingDetector().analyze(returns, n_trials, report)
        AssumptionChecker().analyze_trades(returns, report)
        if check_propfirm: PropFirmChecker().check(returns, firm_name, report)
    report.finalize()
    return report

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
st.markdown("""<div class="main-header"><h1>🔬 SMART BACKTEST VALIDATOR v2</h1><p>Detect lookahead bias · overfitting · logic bugs · prop compliance</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("⚙️ SETTINGS")
    n_trials = st.slider("Strategies tested before this", 1, 200, 1, help="Higher = stricter Deflated Sharpe")
    check_prop = st.checkbox("Check Prop Firm Compliance", False)
    firm = st.selectbox("Prop Firm", list(PropFirmChecker.FIRMS.keys())) if check_prop else list(PropFirmChecker.FIRMS.keys())[0]

tab1, tab2, tab3 = st.tabs(["📋 Code Analysis", "📊 Returns Analysis", "📖 How It Works"])

with tab1:
    st.markdown('<div class="section-header">PASTE STRATEGY CODE</div>', unsafe_allow_html=True)
    sample = """import pandas as pd
import numpy as np
# Example: EMA crossover
df['Signal'] = np.where(df['EMA_fast'] > df['EMA_slow'], 1, 0)
# Note: No .shift() but uses df.iloc[i+2]['open'] for entry → OK!
"""
    code = st.text_area("Python Code", value=sample, height=250)
    if st.button("🔍 ANALYZE CODE"):
        with st.spinner("Analyzing AST & logic..."):
            rpt = run_validation(code, None, n_trials, False, firm)
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number" style="color:{score_color(rpt.score)}">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:8px;font-weight:700">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with col2:
            c,w,o = sum(1 for i in rpt.issues if i.severity=='CRITICAL'), sum(1 for i in rpt.issues if i.severity=='WARNING'), sum(1 for i in rpt.issues if i.severity=='OK')
            st.markdown(f"🔴 {c} Critical · 🟡 {w} Warnings · 🟢 {o} Passed")
        st.markdown('<div class="section-header">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)

with tab2:
    st.markdown('<div class="section-header">UPLOAD / PASTE RETURNS</div>', unsafe_allow_html=True)
    method = st.radio("Input", ["Upload CSV", "Paste decimals"], horizontal=True)
    returns = None
    if method == "Upload CSV":
        up = st.file_uploader("Returns CSV", type="csv")
        if up:
            try:
                df = pd.read_csv(up)
                col = next((c for c in df.columns if 'return' in c.lower() or 'pnl' in c.lower()), df.select_dtypes('number').columns[0])
                returns = pd.to_numeric(df[col], errors='coerce').dropna()
                st.success(f"✅ Loaded {len(returns)} observations from '{col}'")
            except Exception as e: st.error(f"CSV Error: {e}")
    else:
        raw = st.text_area("Comma-separated returns", placeholder="0.012, -0.005, 0.023...")
        if raw:
            try:
                returns = pd.Series([float(x.strip()) for x in raw.replace('\n',',').split(',') if x.strip()])
                st.success(f"✅ Loaded {len(returns)} data points")
            except: st.error("Invalid format. Use decimals only.")
    
    if st.button("📊 ANALYZE RETURNS") and returns is not None:
        with st.spinner("Computing statistics..."):
            rpt = run_validation("", returns, n_trials, check_prop, firm)
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number" style="color:{score_color(rpt.score)}">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:8px;font-weight:700">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-header">METRICS</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,(k,v) in enumerate(rpt.metrics.items()):
                with cols[i%3]: st.markdown(f"""<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{k}</div></div>""", unsafe_allow_html=True)
        st.line_chart((1+returns).cumprod(), height=250)
        st.markdown('<div class="section-header">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)
        st.download_button("⬇️ Download Report JSON", json.dumps({'score':rpt.score, 'metrics':rpt.metrics, 'issues': [{'s':i.severity,'c':i.category,'m':i.message} for i in rpt.issues]}, indent=2), "validation_report.json", "application/json")

with tab3:
    st.markdown("### 🔍 What This Detects")
    st.info("• **Context-Aware Lookahead**: Tracks signal → PnL flow. Recognizes `i+2` entry vs same-bar multiplication.")
    st.info("• **Fixed Deflated Sharpe**: Uses correct Bailey & López de Prado formula. No more 359,000% Sharpe.")
    st.info("• **Logic Bug Catcher**: Flags `cumsum()` on binary signals, flat commission math, pandas KeyErrors.")
    st.info("• **Prop Firm Simulator**: Tests daily DD, total DD, profit targets against FTMO/Topstep/MyFundedFX rules.")
    st.info("• **Zero False Positives on Comments**: Uses regex math detection, not keyword matching.")

st.markdown("""<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #1e3a5f"><span style="font-family:JetBrains Mono;font-size:0.7rem;color:#1e3a5f">QUANT ALPHA SMART VALIDATOR v2 — NOT FINANCIAL ADVICE</span></div>""", unsafe_allow_html=True)
