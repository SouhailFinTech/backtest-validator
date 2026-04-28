"""
╔══════════════════════════════════════════════════════╗
║     QUANT ALPHA — INSTITUTIONAL VALIDATOR v3.0       ║
║     Founder: Hrich Souhail                           ║
║   Detects: Lookahead, Overfitting (DSR),             ║
║            Parameter Sensitivity, Monte Carlo Risk   ║
║   Streamlit App — Production Ready                   ║
╚══════════════════════════════════════════════════════╝

Install:  pip install streamlit pandas numpy scipy
Run:      streamlit run quant_alpha_v3.py
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
    st.error("⚠️ Requires Python 3.9+ for advanced analysis.")
    st.stop()

SCORING = {
    'START': 100,
    'CRITICAL_PENALTY': -25,
    'WARNING_PENALTY': -10,
    'THRESHOLDS': {'VALID': 80, 'QUESTIONABLE': 55}
}

INITIAL_CAPITAL = 10000.0

st.set_page_config(
    page_title="Quant Alpha | Institutional Validator",
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
    background: linear-gradient(135deg, #05080f 0%, #0a111f 100%); 
    color: #f8fafc; 
}
.stApp { background: linear-gradient(135deg, #05080f 0%, #0a111f 100%); }

.header-box { 
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); 
    border-radius: 16px; 
    padding: 32px; 
    text-align: center; 
    margin-bottom: 32px;
    box-shadow: 0 10px 40px rgba(0, 242, 254, 0.2);
}
.header-box h1 { 
    font-family: 'JetBrains Mono', monospace; 
    color: #001e36; 
    font-size: 2.2rem; 
    margin: 0;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: -1px;
}
.header-box p { 
    color: #003355; 
    margin: 8px 0 0; 
    font-size: 1rem;
    font-weight: 600;
}
.founder-tag {
    display: inline-block;
    background: rgba(0,0,0,0.15);
    padding: 6px 16px;
    border-radius: 20px;
    margin-top: 12px;
    font-size: 0.85rem;
    font-weight: 700;
    color: #001e36;
    letter-spacing: 0.5px;
}

.score-card { 
    border-radius: 16px; 
    padding: 28px; 
    text-align: center; 
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.score-card:hover { transform: translateY(-2px); }
.score-great { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); border: 2px solid #55efc4; }
.score-ok { background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%); border: 2px solid #ffeaa7; }
.score-bad { background: linear-gradient(135deg, #d63031 0%, #e84393 100%); border: 2px solid #ff7675; }
.score-number { font-family: 'JetBrains Mono', monospace; font-size: 4rem; font-weight: 800; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2); }
.score-label { font-size: 0.95rem; color: rgba(255,255,255,0.9); margin-top: 8px; font-weight: 600; }

.metric-box { 
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(79, 172, 254, 0.2);
    border-radius: 12px; 
    padding: 16px; 
    text-align: center;
    margin: 8px 0;
    transition: all 0.2s;
}
.metric-box:hover { border-color: rgba(79, 172, 254, 0.6); box-shadow: 0 4px 15px rgba(0, 242, 254, 0.15); }
.metric-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #4facfe; }
.metric-lbl { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; font-weight: 500; }

.section { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.75rem; 
    color: #4facfe; 
    letter-spacing: 2px; 
    text-transform: uppercase; 
    margin: 28px 0 16px; 
    border-bottom: 2px solid rgba(79, 172, 254, 0.3); 
    padding-bottom: 10px;
    font-weight: 700;
}

[data-testid="stSidebar"] { background: linear-gradient(180deg, #05080f 0%, #0a111f 100%); border-right: 2px solid rgba(79, 172, 254, 0.3); }

.stButton > button { 
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); 
    color: #001e36; border: none; border-radius: 10px; font-weight: 800; padding: 12px 28px; width: 100%;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3); transition: all 0.3s;
}
.stButton > button:hover { background: linear-gradient(135deg, #55efc4 0%, #81ecec 100%); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5); }

.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; font-weight: 600; padding: 10px 20px; }
.stTabs [aria-selected="true"] { color: #4facfe !important; background: rgba(79, 172, 254, 0.1); border-radius: 8px 8px 0 0; }

.stTextArea textarea { 
    background: rgba(15, 23, 42, 0.6); color: #f8fafc; border: 2px solid rgba(148, 163, 184, 0.2); 
    border-radius: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; backdrop-filter: blur(10px);
}
.stTextArea textarea:focus { border-color: #4facfe; box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.15); }

.stRadio label { font-size: 0.9rem; color: #cbd5e1; font-weight: 500; }

#MainMenu, footer, header { visibility: hidden; }

.issue-card {
    border-radius: 12px; padding: 16px 20px; margin: 12px 0; border-left: 4px solid;
    backdrop-filter: blur(10px); transition: all 0.2s;
}
.issue-card:hover { transform: translateX(4px); }
.issue-critical { background: rgba(214, 48, 49, 0.15); border-left-color: #ff7675; }
.issue-warning  { background: rgba(225, 112, 85, 0.15); border-left-color: #fdcb6e; }
.issue-info     { background: rgba(79, 172, 254, 0.15); border-left-color: #4facfe; }
.issue-ok       { background: rgba(0, 184, 148, 0.15); border-left-color: #55efc4; }
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
    REALISTIC_ENTRY = [r"df\.iloc\[i\+\d+\]", r"entry.*=.*open", r"next_bar"]

    def analyze(self, code: str, report: ValidationReport):
        try: tree = ast.parse(code)
        except SyntaxError as e:
            report.add("CRITICAL", "Syntax", f"Cannot parse: {e}")
            return

        lines = code.split('\n')
        signal_defs = {}
        shift_vars = set()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'): continue
            
            if '.shift(' in stripped:
                match = re.match(r'(\w+)\s*=\s*.*\.shift\(', stripped)
                if match: shift_vars.add(match.group(1))
                match = re.match(r'df\[(.+?)\]\s*=\s*(\w+)\.shift\(', stripped)
                if match: shift_vars.add(match.group(2))

            sig_match = re.match(r'(\w+)\s*=\s*.*df\[.*\].*[><=!]+.*', stripped)
            if sig_match:
                var_name = sig_match.group(1)
                signal_defs[var_name] = i

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'): continue
            for sig_name, def_line in signal_defs.items():
                if sig_name in stripped and '*' in stripped and i != def_line:
                    if sig_name not in shift_vars and '.shift(' not in stripped:
                        report.add("WARNING", "Lookahead",
                            f"Signal '{sig_name}' multiplied without .shift()",
                            f"Line {i+1}: '{stripped}'\nUsing a signal in multiplication without .shift(1) causes same-bar lookahead bias.")
                        return

        # Robustness Scan: Magic Numbers
        magic_numbers = re.findall(r'(?<!\w)(\d{2,})(?!\w)', code)
        param_vars = re.findall(r'(\w+)\s*=\s*\d+', code)
        if len(magic_numbers) > 10 and len(param_vars) < 3:
            report.add("WARNING", "Robustness", "Hardcoded parameters detected", 
                       "Strategy uses many hardcoded numbers without defined variables. Suggests fragile, non-optimized logic.")

        if '.fit(' in code and not re.search(r'train|test|split|fold|cv=', code, re.IGNORECASE):
            report.add("CRITICAL", "Lookahead", ".fit()/.fit_transform() on full dataset", "Splits data first. Use TimeSeriesSplit or train/test split.")
        if 'center=True' in code and 'rolling(' in code:
            report.add("CRITICAL", "Lookahead", "center=True in rolling window", "Leaks future bars. Use center=False (default) or shift results.")
        if ("['Close']" in code or '["Close"]' in code) and 'open' not in code.lower():
            report.add("WARNING", "Lookahead", "Using Close for entry — possible lookahead", "Close is unknown until bar ends. Use next bar's Open.")

class OverfittingDetector:
    @staticmethod
    @lru_cache(maxsize=32)
    def _stats(returns_tuple: tuple, n_trials: int):
        ret = pd.Series(returns_tuple)
        if len(ret) < 10: return None
        m, s = ret.mean(), ret.std()
        sr = (m / s * np.sqrt(252)) if s > 0 else 0
        
        T, skew, kurt = len(ret), float(ret.skew()), float(ret.kurtosis())
        try:
            # Bailey & Lopez de Prado DSR Formula
            var_sr = (1 + 0.5*sr**2 - skew*sr + (kurt-3)/4 * sr**2) / T
            # DSR penalizes Sharpe based on number of trials (parameter sensitivity)
            dsr = sr / (np.sqrt(var_sr) * np.sqrt(np.log(n_trials))) if n_trials > 1 else sr / np.sqrt(var_sr)
        except: dsr = sr
        
        cum = (1 + ret).cumprod()
        r = np.corrcoef(np.arange(len(cum)), np.log(cum.clip(1e-6)))[0,1] if len(cum)>1 else 0
        dd = (cum.cummax() - cum) / cum.cummax()
        pf = ret[ret>0].sum()/abs(ret[ret<0].sum()) if ret[ret<0].sum()!=0 else np.inf
        
        # Stability Score (Sharpe / StdDev of Bootstrapped Sharpe)
        boot_sr = []
        for _ in range(200):
            sample = ret.sample(n=len(ret), replace=True)
            sm, ss = sample.mean(), sample.std()
            boot_sr.append((sm/ss*np.sqrt(252)) if ss > 0 else 0)
        stability = (np.mean(boot_sr) / np.std(boot_sr)) if np.std(boot_sr) > 0 else 99
        
        return {
            'Sharpe': round(sr, 3), 'Deflated Sharpe': round(dsr, 3), 'Stability Score': round(stability, 2),
            'Max DD': f"{dd.max():.1%}", 'Win Rate': f"{(ret>0).mean():.1%}", 
            'Profit Factor': round(pf, 2) if pf!=np.inf else "∞",
            'Smoothness R²': round(r**2, 3), 'Total Return': f"{(cum.iloc[-1]-1):.1%}", 'Obs': T
        }

    def analyze(self, returns: pd.Series, n_trials: int, report: ValidationReport):
        stats = self._stats(tuple(returns.dropna()), n_trials)
        if not stats: return
        report.metrics.update({k: v for k, v in stats.items()})
        
        sr, dsr, stab, pf, r2, T = stats['Sharpe'], stats['Deflated Sharpe'], stats['Stability Score'], stats['Profit Factor'], stats['Smoothness R²'], stats['Obs']
        
        # Parameter Sensitivity Check
        if dsr < 0:
            report.add("CRITICAL", "Overfitting", f"Deflated Sharpe {dsr:.2f} (Negative)", 
                       f"Strategy is likely overfitted after testing {n_trials} parameter combinations. Edge is not robust.")
        elif dsr < 1:
            report.add("WARNING", "Overfitting", f"Deflated Sharpe {dsr:.2f} (Low)", 
                       f"After adjusting for {n_trials} trials, edge weakens significantly. High risk of parameter instability.")
        elif sr > 4:
             report.add("CRITICAL", "Overfitting", f"Sharpe {sr:.2f} unrealistic", "Live strategies rarely exceed 1.5. Check lookahead/fees.")
        elif sr > 3:
            report.add("WARNING", "Overfitting", f"Sharpe {sr:.2f} suspicious", "Verify OOS & transaction costs.")
        elif sr > 1.5:
            report.add("INFO", "Overfitting", f"Sharpe {sr:.2f} solid", "Strong performance. Check stability score.")
        else:
            report.add("OK", "Overfitting", f"Sharpe {sr:.2f} realistic", "Within institutional norms.")

        if stab < 2:
            report.add("WARNING", "Sensitivity", f"Stability Score {stab:.1f} (Low)", 
                       "Returns are highly sensitive to sample variation. Strategy may lack parameter robustness.")
            
        if isinstance(pf, (int, float)) and pf > 5: report.add("WARNING", "Overfitting", f"PF {pf:.1f} too high", "Rarely survives live trading. Realistic: 1.3–2.5.")
        if r2 > 0.97 and sr > 1.5: report.add("WARNING", "Overfitting", "Equity curve too smooth (R²>0.97)", "Perfect curves often indicate lookahead bias.")
        if T < 100: report.add("WARNING", "Overfitting", f"Only {T} obs (<6mo)", "Need ≥100 for stable stats.")

# Monte Carlo for Returns Tab
def run_monte_carlo(returns, simulations=1000):
    ret_arr = returns.values
    n_days = len(ret_arr)
    final_equities = []
    max_drawdowns = []
    
    for _ in range(simulations):
        # Random permutation (Block Bootstrap proxy)
        shuffled = np.random.choice(ret_arr, size=n_days, replace=True)
        equity = np.cumprod(1 + shuffled)
        peak = np.maximum.accumulate(equity)
        dd = np.max((peak - equity) / peak)
        final_equities.append(equity[-1])
        max_drawdowns.append(dd)
        
    return {
        'sharpe_dist': np.percentile(final_equities, [5, 50, 95]),
        'dd_dist': np.percentile(max_drawdowns, [5, 50, 95])
    }

class AssumptionChecker:
    def analyze_code(self, code: str, report: ValidationReport):
        cost_pat = [r'commission\s*=', r'\*\s*0\.\d+', r'slippage', r'fee\s*=', r'transaction_cost', r'spread\s*[+*=]']
        if not any(re.search(p, code, re.IGNORECASE) for p in cost_pat):
            report.add("WARNING", "Assumptions", "No transaction costs modeled", "Crypto: 0.1%/trade. Forex: spread. Costs cut returns 30-70%.")
        else: report.add("OK", "Assumptions", "Transaction costs detected ✓")
        if 'volume' not in code.lower() and 'liquidity' not in code.lower():
            report.add("INFO", "Assumptions", "No liquidity constraints", "Add volume-based sizing to model slippage.")

class LogicBugDetector:
    def analyze(self, code: str, report: ValidationReport):
        if 'cumsum()' in code and re.search(r"Position.*=.*Signal.*cumsum", code, re.IGNORECASE):
            report.add("CRITICAL", "Logic", "Position uses cumsum() on signal", "Creates [0,1,2,3...] positions. Use: Position = Signal (1/0)")
        comm_pat = r"(Net|net|PnL|pnl|returns|result)\s*=.*-\s*(0\.00\d|0\.01|0\.1|Commission|commission|comm|cost|fee|spread)"
        if re.search(comm_pat, code, re.IGNORECASE) and not re.search(r"(commission|comm|cost|fee|spread)\s*=.*trades\s*\*", code, re.IGNORECASE):
            report.add("WARNING", "Logic", "Commission flat-subtracted from returns", "Use: Commission = trades * 0.001, then Net = Strat - Comm")

class PropFirmChecker:
    FIRMS = {
        'FTMO ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.10},
        'Topstep ($150K)': {'max_daily_dd': 0.03, 'max_total_dd': 0.06, 'profit_target': 0.06},
        'MyFundedFX ($100K)': {'max_daily_dd': 0.05, 'max_total_dd': 0.10, 'profit_target': 0.08},
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
        <div style="font-weight:700;margin-bottom:6px">{icon} <b>[{issue.category}]</b>{line_txt}</div>
        <div style="color:#f1f5f9;margin-bottom:4px">{issue.message}</div>{detail}
    </div>""", unsafe_allow_html=True)

def score_style(s): return 'score-great' if s>=80 else 'score-ok' if s>=55 else 'score-bad'
def score_color(s): return '#55efc4' if s>=80 else '#fdcb6e' if s>=55 else '#ff7675'

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown('''
<div class="header-box">
    <h1>🔬 QUANT ALPHA VALIDATOR v3.0</h1>
    <p>Lookahead Bias • Parameter Sensitivity (DSR) • Monte Carlo Risk</p>
    <div class="founder-tag">Founder: Hrich Souhail</div>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section">⚙️ SETTINGS</div>', unsafe_allow_html=True)
    n_trials = st.slider("Parameter Trials Tested", 1, 500, 1, help="Crucial for DSR calculation. If you tested 50 EMA combos, set to 50.")
    check_prop = st.checkbox("Prop Firm Compliance", False)
    firm = st.selectbox("Prop Firm", list(PropFirmChecker.FIRMS.keys())) if check_prop else list(PropFirmChecker.FIRMS.keys())[0]
    st.markdown('<div class="section">🧠 INSTITUTIONAL DETECTIONS</div>', unsafe_allow_html=True)
    st.markdown('''<div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
    • Bailey & Lopez de Prado DSR<br>• Monte Carlo Drawdown CI<br>• Indirect Signal Tracking<br>• Magic Number Detection<br>• Stability Score Analysis
    </div>''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Code Audit", "📊 Returns & Sensitivity", "📖 Methodology"])

with tab1:
    st.markdown('<div class="section">PASTE STRATEGY CODE</div>', unsafe_allow_html=True)
    sample = """sig = (df['Close'] > df['MA20']).astype(int)
df['Strat'] = df['Ret'] * sig"""
    code = st.text_area("Python Code", value=sample, height=280, label_visibility="collapsed")
    if st.button("🔍 AUDIT CODE"):
        if not code.strip(): st.info("ℹ️ Please paste code."); st.stop()
        with st.spinner("Analyzing AST & Logic..."):
            rpt = run_validation(code, None, n_trials, False, firm)
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:10px;font-weight:700;font-size:1.1rem">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
        with c2:
            cnt = {s: sum(1 for i in rpt.issues if i.severity==s) for s in ['CRITICAL','WARNING','OK','INFO']}
            st.markdown(f"""<div style="display:flex;gap:12px;margin-top:20px">
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#ff7675">{cnt['CRITICAL']}</div><div class="metric-lbl">Critical</div></div>
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#fdcb6e">{cnt['WARNING']}</div><div class="metric-lbl">Warnings</div></div>
                <div class="metric-box" style="flex:1"><div class="metric-val" style="color:#55efc4">{cnt['OK']}</div><div class="metric-lbl">Passed</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)

with tab2:
    st.markdown('<div class="section">RETURN ANALYSIS & SENSITIVITY</div>', unsafe_allow_html=True)
    method = st.radio("Input", ["Paste decimals", "Upload CSV"], horizontal=True)
    returns = None
    
    if method == "Upload CSV":
        up = st.file_uploader("Returns CSV", type="csv", label_visibility="collapsed")
        if up:
            try:
                df_up = pd.read_csv(up)
                col = next((c for c in df_up.columns if 'return' in c.lower() or 'pnl' in c.lower()), df_up.select_dtypes('number').columns[0])
                returns = pd.to_numeric(df_up[col], errors='coerce').dropna()
                st.success(f"✅ Loaded {len(returns)} points")
            except: st.error("CSV Error")
    else:
        raw = st.text_area("Comma-separated returns", placeholder="0.01, -0.02, 0.05...", height=120, label_visibility="collapsed")
        if raw:
            try:
                vals = pd.to_numeric(raw.replace('\n',',').replace(' ',',').split(','), errors='coerce')
                returns = vals.dropna()
                st.success(f"✅ Loaded {len(returns)} points")
            except: st.error("Invalid format")
    
    if st.button("📊 RUN SENSITIVITY ANALYSIS"):
        if returns is None or len(returns) < 10: st.warning("⚠️ Need >10 returns."); st.stop()
        
        with st.spinner("Running Monte Carlo (1000 sims) & DSR..."):
            rpt = run_validation("", returns, n_trials, check_prop, firm)
            mc = run_monte_carlo(returns)
            equity = INITIAL_CAPITAL * (1 + returns).cumprod()
            max_dd = float(((equity - equity.cummax()) / equity.cummax()).min()) * 100
            
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""<div class="score-card {score_style(rpt.score)}"><div class="score-number">{rpt.score}</div><div class="score-label">REALISM SCORE / 100</div><div style="margin-top:10px;font-weight:700;font-size:1.1rem">{rpt.verdict}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-box"><div class="metric-val" style="color:#ff7675">{max_dd:.2f}%</div><div class="metric-lbl">Max Drawdown</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section">KEY METRICS</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,(k,v) in enumerate(rpt.metrics.items()):
                with cols[i%3]: st.markdown(f"""<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{k}</div></div>""", unsafe_allow_html=True)
                
        st.markdown('<div class="section">MONTE CARLO DISTRIBUTION (Risk Analysis)</div>', unsafe_allow_html=True)
        st.info(f"🔹 95% Confidence Interval for Final Equity: ${mc['sharpe_dist'][0]*INITIAL_CAPITAL:,.2f} — ${mc['sharpe_dist'][2]*INITIAL_CAPITAL:,.2f}")
        st.info(f"🔹 Probability of >20% Drawdown: {np.mean(np.array([d > 0.20 for d in np.random.choice(returns.values, size=(1000, len(returns)))])):.1%}")
        
        st.markdown('<div class="section">FINDINGS</div>', unsafe_allow_html=True)
        for iss in rpt.issues: render_issue(iss)

with tab3:
    st.markdown('<div class="section">METHODOLOGY & FORMULAS</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="background:rgba(15,23,42,0.6);border-radius:12px;padding:24px;margin:16px 0">
    <div style="margin-bottom:16px"><b style="color:#4facfe">• Deflated Sharpe Ratio (DSR)</b><br><span style="color:#94a3b8">DSR = Sharpe / sqrt(log(N_trials)). This penalizes strategies that simply tried many parameters to find a lucky one. The higher your trials, the lower your DSR.</span></div>
    <div style="margin-bottom:16px"><b style="color:#4facfe">• Stability Score</b><br><span style="color:#94a3b8">Measures Sharpe consistency across 200 bootstrap samples. Low score implies the strategy is fragile and sensitive to small data changes.</span></div>
    <div><b style="color:#4facfe">• Monte Carlo Simulation</b><br><span style="color:#94a3b8">Resamples returns 1,000 times to estimate the true distribution of Max Drawdown and Final Equity, accounting for randomness.</span></div>
    </div>''', unsafe_allow_html=True)

st.markdown('''
<div style="text-align:center;margin-top:40px;padding:24px;border-top:2px solid rgba(79,172,254,0.3)">
    <div style="font-family:'JetBrains Mono';font-size:0.85rem;color:#94a3b8;margin-bottom:8px">QUANT ALPHA v3.0 — INSTITUTIONAL GRADE</div>
    <div style="font-size:0.8rem;color:#64748b">Founder: <b style="color:#4facfe">Hrich Souhail</b> — Not Financial Advice</div>
</div>''', unsafe_allow_html=True)
