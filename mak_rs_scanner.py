"""
MAK RS Scanner  |  MAK Capital - Growing Wealth Together

Ranks every stock in universe.txt, and the sector indices, by Relative Strength
against the NIFTY 500 (always). Then it finds:
  1. Strongest sector leaders  - best stocks of the strongest sectors today
  2. Emerging sector leaders   - short-term RS strong, long-term RS still low (the next leaders)

RS is measured the same way as the TradingView RS Table: percent-rank of the
ratio (stock close / Nifty 500 close) over 5 / 21 / 63 / 126 daily bars.

Setup (once, in Command Prompt):   pip install yfinance pandas numpy openpyxl
Files in the same folder:          mak_rs_scanner.py  and  universe.txt
universe.txt, one stock per line:  RELIANCE,ENERGY      (sector after a comma is optional)
Run:                               python mak_rs_scanner.py
"""
import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd

# ------------------------------- SETTINGS -------------------------------
BENCH = "^CRSLDX"            # Nifty 500 on Yahoo Finance (always the benchmark)
BENCH_NAME = "Nifty 500"
L1, L2, L3, L4 = 5, 21, 63, 126      # same RS periods as the Pine table
LOOKBACK = 5                 # bars back for the rank-change column (about a week)
EMG_SHORT_MIN = 70.0         # emerging: 21D RS at least this ...
EMG_LONG_MAX = 65.0          # ... and 126D RS at most this
MIN_LEADER_RS = 50.0         # a stock must have at least this 21D RS to be called a leader
TOP_SECTORS = 5              # how many top sector families to show leaders for
STOCKS_PER_SECTOR = 3        # leaders listed per sector
HISTORY = "2y"               # price history to download
CHUNK = 50                   # tickers per download request
UNIVERSE_FILE = "universe.txt"
# Set by the automatic GitHub run (MAK_OUT_DIR=docs). Empty = save in the current folder, as on your PC.
OUT_DIR = os.environ.get("MAK_OUT_DIR", "")
MIN_COVERAGE = 0.5           # automatic run refuses to publish if fewer than this share of stocks downloaded

# Sector name -> Yahoo symbol of its NSE index. If a symbol is not available, the scanner
# says so and measures that sector from the median of its own stocks instead.
INDEX_MAP = {
    "BANK": "^NSEBANK",
    "PVT BANK": "NIFTY_PVT_BANK.NS",
    "PSU BANK": "^CNXPSUBANK",
    "FIN SERV": "^CNXFIN",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "HEALTH": "NIFTY_HEALTHCARE.NS",
    "AUTO": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "INFRA": "^CNXINFRA",
    "MEDIA": "^CNXMEDIA",
    "PSE": "^CNXPSE",
    "CONSUMP": "^CNXCONSUM",
    "COMMOD": "^CNXCMDT",
    "MIDCAP": "NIFTY_MIDCAP_100.NS",
    "SMALLCAP": "^CNXSC",
}

# Overlapping sectors count as one family, so you never get HDFC and ICICI as two "leaders".
FAMILY = {
    "BANK": "BANKS", "PVT BANK": "BANKS", "PSU BANK": "BANKS",
    "METAL": "METALS", "COMMOD": "METALS",
    "MIDCAP": "BROAD", "SMALLCAP": "BROAD",
}


def family(sector):
    return FAMILY.get(sector, sector)


# ------------------------------- DATA -------------------------------
def read_universe(path):
    """Returns a list of (yahoo_symbol, plain_symbol, sector)."""
    if not os.path.exists(path):
        sys.exit(f"Cannot find {path}. Put it in the same folder as this script.")
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",")]
            sym = parts[0].upper().replace("NSE:", "").replace(".NS", "")
            sec = parts[1].upper() if len(parts) > 1 and parts[1] else "UNTAGGED"
            if sym and sym not in seen:
                seen.add(sym)
                rows.append((sym + ".NS", sym, sec))
    return rows


def fetch_close(symbols):
    """Daily closes for the given Yahoo symbols (columns = symbols)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. In Command Prompt run:  pip install yfinance pandas numpy openpyxl")
    frames = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(symbols)} ...")
        df = None
        for attempt in range(3):  # Yahoo sometimes says "too many requests": wait and try again
            try:
                df = yf.download(chunk, period=HISTORY, interval="1d", auto_adjust=True,
                                 progress=False, threads=True)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print("  chunk failed:", e)
            time.sleep(5 * (attempt + 1))
        if df is None or df.empty:
            continue
        time.sleep(1)
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=chunk[0])
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    return out.sort_index()


# ------------------------------- RS MATHS -------------------------------
def pr_at(x, length, t):
    """Percent-rank like TradingView ta.percentrank: share of the previous `length`
    values that are <= the value at position t, in percent."""
    if t - length < 0 or t >= len(x):
        return np.nan
    cur = x[t]
    if np.isnan(cur):
        return np.nan
    w = x[t - length:t]
    w = w[~np.isnan(w)]
    if len(w) < 0.8 * length:
        return np.nan
    return float((w <= cur).sum()) / len(w) * 100.0


def rs_values(close_series, bench):
    """P1..P4 now, and the same as they were LOOKBACK bars ago (RS vs the benchmark)."""
    ratio = close_series.reindex(bench.index).ffill(limit=3) / bench
    x = ratio.to_numpy(dtype=float)
    n = len(x)
    t, tp = n - 1, n - 1 - LOOKBACK
    out = {}
    for name, ln in (("P1", L1), ("P2", L2), ("P3", L3), ("P4", L4)):
        out[name] = pr_at(x, ln, t)
        out[name + "_prev"] = pr_at(x, ln, tp)
    return out


def add_rank_delta(df):
    """Rank by 21D RS now and LOOKBACK bars ago. Positive = moved up that many places."""
    now = df["P2"].rank(method="min", ascending=False)
    prev = df["P2_prev"].rank(method="min", ascending=False)
    df["RankChg"] = prev - now
    df["Emerging"] = (df["P2"] >= EMG_SHORT_MIN) & (df["P4"] <= EMG_LONG_MAX)
    return df


def fmt_chg(v):
    if pd.isna(v):
        return "-"
    v = int(v)
    return f"+{v}" if v > 0 else (str(v) if v < 0 else "0")


# ------------------------------- LEADERS -------------------------------
def pick_leaders(sectors, stocks, emerging_only):
    """Walk sectors strongest-first, one sector family at a time, and list its best stocks."""
    rows, used = [], set()
    for _, s in sectors.sort_values("P2", ascending=False).iterrows():
        if len(used) >= TOP_SECTORS:
            break
        fam = family(s["Sector"])
        if fam in used or pd.isna(s["P2"]):
            continue
        if emerging_only and not s["Emerging"]:
            continue
        cand = stocks[stocks["Sector"].map(family) == fam]
        cand = cand[cand["P2"] >= MIN_LEADER_RS]
        if emerging_only:
            cand = cand[cand["Emerging"]]
        cand = cand.sort_values("P2", ascending=False).head(STOCKS_PER_SECTOR)
        if cand.empty:
            continue
        used.add(fam)
        for _, c in cand.iterrows():
            rows.append({"Sector": s["Sector"], "Sector 21D": round(s["P2"], 1),
                         "Stock": c["Symbol"], "Stock sector": c["Sector"],
                         "21D": round(c["P2"], 1), "63D": round(c["P3"], 1),
                         "126D": round(c["P4"], 1), "Rank chg": fmt_chg(c["RankChg"]),
                         "Emerging": "*" if c["Emerging"] else ""})
    return pd.DataFrame(rows)



# ------------------------------- WEB PAGE -------------------------------
HTML_CSS = """
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Roboto,Arial,sans-serif;background:#f3f6f8;color:#1b2b34}
.band{background:#fff;padding:22px 28px 14px;border-bottom:4px solid #c4742c;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.arrow{font-size:54px;color:#c4742c;font-weight:700;line-height:1}
.mak{font-size:38px;font-weight:800;color:#034a63;line-height:1}.cap{letter-spacing:7px;color:#034a63;font-size:15px}
.tag{color:#c4742c;font-weight:700;font-size:13px}
.meta{margin-left:auto;text-align:right;color:#034a63;font-size:14px}
.sign{background:#034a63;color:#fff;text-align:center;padding:6px;font-size:13px}
main{max-width:1180px;margin:0 auto;padding:18px}
h2{color:#034a63;border-left:5px solid #c4742c;padding-left:10px;font-size:18px;margin:26px 0 4px}
.note{color:#5a6b74;font-size:12.5px;margin:0 0 8px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:auto;max-height:640px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{background:#034a63;color:#fff;padding:8px 10px;text-align:center;position:sticky;top:0;cursor:pointer;white-space:nowrap}
td{padding:6px 10px;text-align:center;border-bottom:1px solid #e6edf0;white-space:nowrap}
td:first-child,th:first-child{text-align:left}
tr:hover td{background:#fbf3ea}
.s{color:#0a8f3c;font-weight:700}.w{color:#c62828;font-weight:700}.n{color:#7a5a2a}
.up{color:#0a8f3c;font-weight:700}.dn{color:#c62828;font-weight:700}
.star{color:#c4742c;font-weight:800}
input{padding:7px 10px;border:1px solid #b9c7ce;border-radius:6px;margin:6px 0;width:260px}
footer{text-align:center;color:#5a6b74;font-size:12px;padding:20px}
"""

HTML_JS = """
function sortTable(th){var t=th.closest('table'),i=[].indexOf.call(th.parentNode.children,th),
 asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';var rows=[].slice.call(t.tBodies[0].rows);
 function v(r){var x=r.cells[i].dataset.v;if(x===undefined)x=r.cells[i].innerText;var n=parseFloat(x);return isNaN(n)?x.toLowerCase():n}
 rows.sort(function(a,b){var x=v(a),y=v(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});
 rows.forEach(function(r){t.tBodies[0].appendChild(r)})}
function filt(inp,id){var q=inp.value.toLowerCase();[].forEach.call(document.getElementById(id).tBodies[0].rows,
 function(r){r.style.display=r.innerText.toLowerCase().indexOf(q)>-1?'':'none'})}
"""


def _cell(col, val):
    """One table cell: colour the RS numbers, arrows for rank change, star for emerging."""
    import html as _h
    if pd.isna(val):
        return '<td data-v="-1">-</td>'
    name = str(col)
    if name.endswith("D") and name[:-1].isdigit() or name == "Sector 21D":
        cls = "s" if val >= 80 else ("w" if val < 39 else "")
        return (f'<td class="{cls}" data-v="{val}" style="background:linear-gradient(90deg,'
                f'rgba(3,74,99,.14) {val:.0f}%,transparent {val:.0f}%)">{val:.1f}</td>')
    if name == "Rank chg":
        t = str(val)
        if t.startswith("+"):
            return f'<td class="up" data-v="{t[1:]}">&#9650;{t[1:]}</td>'
        if t.startswith("-") and len(t) > 1:
            return f'<td class="dn" data-v="{t}">&#9660;{t[1:]}</td>'
        return f'<td data-v="0">{_h.escape(t)}</td>'
    if name == "Emerging":
        return '<td class="star" data-v="1">&#9733;</td>' if val == "*" else '<td data-v="0"></td>'
    return f"<td>{_h.escape(str(val))}</td>"


def _html_table(df, tid, search=False):
    if df is None or df.empty:
        return '<p class="note">Nothing qualifies right now.</p>'
    head = "".join(f'<th onclick="sortTable(this)">{c}</th>' for c in df.columns)
    body = "".join("<tr>" + "".join(_cell(c, v) for c, v in zip(df.columns, row)) + "</tr>"
                   for row in df.itertuples(index=False, name=None))
    box = f'<input placeholder="Search..." oninput="filt(this,\'{tid}\')">' if search else ""
    return f'{box}<div class="card"><table id="{tid}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_html(path, sec_show, strongest, emerging, top_show, all_show, n_universe, last_day, xlsx_href=None):
    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    stamp = f"{ist:%d %b %Y, %I:%M %p} IST"
    dl = (f'<br><a href="{xlsx_href}" style="color:#034a63">Download Excel</a>' if xlsx_href else "")
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MAK Capital - RS Scan vs {BENCH_NAME}</title>
<style>{HTML_CSS}</style></head><body>
<div class="band"><div class="arrow">&#8599;</div><div><div class="mak">MAK</div>
<div class="cap">C A P I T A L</div><div class="tag">Growing Wealth Together</div></div>
<div class="meta"><b>Relative Strength Scan vs {BENCH_NAME}</b><br>Data up to {last_day:%d %b %Y} &middot; {n_universe} stocks scanned<br>Updated {stamp}</div></div>
<div class="sign">Mandarr A Kashikar | MAK Capital</div>
<main>
<h2>Sectors ranked by {L2}D RS</h2>
<p class="note">Green = strong (80 or more), red = weak (below 39). &#9733; = emerging (21D at least {EMG_SHORT_MIN:.0f}, 126D at most {EMG_LONG_MAX:.0f}). Rank chg = places moved in the last {LOOKBACK} bars. Click any heading to sort.</p>
{_html_table(sec_show, "t_sec")}
<h2>Emerging sector leaders &#9733;</h2>
<p class="note">Short-term strength is already visible but the long-term RS is still low: the leaders in the making.</p>
{_html_table(emerging, "t_emg")}
<h2>Strongest sector leaders</h2>
<p class="note">Best stocks of the top {TOP_SECTORS} sector families today (overlapping sectors counted once; minimum 21D RS {MIN_LEADER_RS:.0f}).</p>
{_html_table(strongest, "t_str")}
<h2>Top 25 stocks across the whole universe</h2>
{_html_table(top_show, "t_top")}
<h2>All stocks</h2>
{_html_table(all_show, "t_all", search=True)}
</main>
<footer>A screening tool, not a recommendation. Confirm every name on the chart before acting.<br>MAK Capital | Growing Wealth Together{dl}</footer>
<script>{HTML_JS}</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# ------------------------------- MAIN -------------------------------
def main():
    print(f"\nMAK RS Scanner  |  RS vs {BENCH_NAME}  |  {dt.date.today():%d-%b-%Y}\n")
    universe = read_universe(UNIVERSE_FILE)
    print(f"Universe: {len(universe)} stocks")

    print("Downloading benchmark ...")
    bench_df = fetch_close([BENCH])
    if bench_df.empty or BENCH not in bench_df.columns:
        sys.exit(f"Could not download the benchmark {BENCH} ({BENCH_NAME}). Check your internet and retry.")
    bench = bench_df[BENCH].dropna()
    last_day = bench.index[-1]
    print(f"Benchmark last date: {last_day:%d-%b-%Y}\n")

    print("Downloading sector indices ...")
    idx_close = fetch_close(list(INDEX_MAP.values()))
    print("Downloading stocks ...")
    stk_close = fetch_close([u[0] for u in universe])

    # ---- stocks ----
    srows, skipped = [], []
    for ysym, sym, sec in universe:
        if ysym not in stk_close.columns:
            skipped.append(sym)
            continue
        s = stk_close[ysym].dropna()
        if s.empty or s.index[-1] < last_day - pd.Timedelta(days=5):  # stale or suspended
            skipped.append(sym)
            continue
        r = rs_values(s, bench)
        r.update({"Symbol": sym, "Sector": sec})
        srows.append(r)
    stocks = add_rank_delta(pd.DataFrame(srows))
    stocks = stocks.dropna(subset=["P2"]).reset_index(drop=True)
    if OUT_DIR and len(stocks) < MIN_COVERAGE * len(universe):
        sys.exit(f"Only {len(stocks)} of {len(universe)} stocks downloaded (Yahoo may be blocking). "
                 "Not publishing a half-empty page; the previous page stays online.")

    # ---- sectors: use the index when it downloaded, else the median of the sector's own stocks ----
    sec_names = set(INDEX_MAP) | set(s for s in stocks["Sector"].unique() if s != "UNTAGGED")
    secrows, no_index = [], []
    for name in sorted(sec_names):
        ysym = INDEX_MAP.get(name)
        r = None
        if ysym and ysym in idx_close.columns and idx_close[ysym].dropna().size > L4 + LOOKBACK + 5:
            r = rs_values(idx_close[ysym].dropna(), bench)
            r["Source"] = "index"
        else:
            grp = stocks[stocks["Sector"] == name]
            if len(grp) >= 3:
                r = {k: grp[k].median() for k in
                     ("P1", "P2", "P3", "P4", "P1_prev", "P2_prev", "P3_prev", "P4_prev")}
                r["Source"] = f"median of {len(grp)} stocks"
            if ysym:
                no_index.append(name)
        if r:
            r["Sector"] = name
            secrows.append(r)
    sectors = add_rank_delta(pd.DataFrame(secrows)).sort_values("P2", ascending=False).reset_index(drop=True)

    # ---- leaders ----
    strongest = pick_leaders(sectors, stocks, emerging_only=False)
    emerging = pick_leaders(sectors, stocks, emerging_only=True)

    # ---- show ----
    sec_show = sectors[["Sector", "Source", "P1", "P2", "P3", "P4", "RankChg", "Emerging"]].copy()
    sec_show["RankChg"] = sec_show["RankChg"].map(fmt_chg)
    sec_show["Emerging"] = sec_show["Emerging"].map({True: "*", False: ""})
    sec_show = sec_show.rename(columns={"P1": f"{L1}D", "P2": f"{L2}D", "P3": f"{L3}D", "P4": f"{L4}D",
                                        "RankChg": "Rank chg"}).round(1)
    pd.set_option("display.width", 200)
    print(f"\n=== SECTORS ranked by {L2}D RS vs {BENCH_NAME}   (* = emerging) ===")
    print(sec_show.to_string(index=False))
    print(f"\n=== STRONGEST SECTOR LEADERS (top {TOP_SECTORS} sector families) ===")
    print(strongest.to_string(index=False) if not strongest.empty else "None qualify.")
    print("\n=== EMERGING SECTOR LEADERS (short-term strong, long-term still low) ===")
    print(emerging.to_string(index=False) if not emerging.empty else "No emerging sector leader right now.")

    top_stocks = stocks.sort_values("P2", ascending=False).head(25)
    top_show = top_stocks[["Symbol", "Sector", "P2", "P3", "P4", "RankChg", "Emerging"]].copy()
    top_show["RankChg"] = top_show["RankChg"].map(fmt_chg)
    top_show["Emerging"] = top_show["Emerging"].map({True: "*", False: ""})
    top_show = top_show.rename(columns={"P2": f"{L2}D", "P3": f"{L3}D", "P4": f"{L4}D", "RankChg": "Rank chg"})
    print(f"\n=== TOP 25 STOCKS by {L2}D RS (whole universe) ===")
    print(top_show.round(1).to_string(index=False))

    paste = list(dict.fromkeys(list(strongest.get("Stock", [])) + list(emerging.get("Stock", []))))
    if paste:
        print("\nPaste into the TradingView RS Table stock slots (Tickers 17-35):")
        print("  " + "\n  ".join("NSE:" + p for p in paste))

    if skipped:
        print(f"\nNo usable data for {len(skipped)} stocks (renamed, delisted or suspended?): "
              + ", ".join(skipped[:30]) + (" ..." if len(skipped) > 30 else ""))
    if no_index:
        print("Index not available from Yahoo, measured from its stocks instead: " + ", ".join(no_index))
    if (stocks["Sector"] == "UNTAGGED").any():
        print(f"{(stocks['Sector'] == 'UNTAGGED').sum()} stocks have no sector in universe.txt: "
              "they are ranked but cannot be sector leaders.")

    # ---- Excel ----
    all_show = stocks[["Symbol", "Sector", "P1", "P2", "P3", "P4", "RankChg", "Emerging"]].copy()
    all_show = all_show.sort_values("P2", ascending=False)
    all_show = all_show.rename(columns={"P1": f"{L1}D", "P2": f"{L2}D", "P3": f"{L3}D", "P4": f"{L4}D",
                                        "RankChg": "Rank chg"}).round(1)
    day = f"{last_day:%Y-%m-%d}"  # named after the market day, not the day the script ran

    def save_xlsx(path):
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            sec_show.to_excel(xw, sheet_name="Sectors", index=False)
            strongest.to_excel(xw, sheet_name="Strongest leaders", index=False)
            emerging.to_excel(xw, sheet_name="Emerging leaders", index=False)
            all_show.to_excel(xw, sheet_name="All stocks", index=False)

    top_html = top_show.round(1)
    if OUT_DIR:  # automatic run: docs/index.html (latest), docs/archive/<day>.html, docs/MAK_RS_Scan_latest.xlsx
        os.makedirs(os.path.join(OUT_DIR, "archive"), exist_ok=True)
        save_xlsx(os.path.join(OUT_DIR, "MAK_RS_Scan_latest.xlsx"))
        args = (sec_show, strongest, emerging, top_html, all_show, len(stocks), last_day)
        write_html(os.path.join(OUT_DIR, "index.html"), *args, xlsx_href="MAK_RS_Scan_latest.xlsx")
        write_html(os.path.join(OUT_DIR, "archive", f"MAK_RS_Scan_{day}.html"), *args,
                   xlsx_href="../MAK_RS_Scan_latest.xlsx")
        print(f"\nSaved to {OUT_DIR}: index.html, archive/MAK_RS_Scan_{day}.html, MAK_RS_Scan_latest.xlsx")
    else:  # on your PC
        out_name = f"MAK_RS_Scan_{day}.xlsx"
        for attempt in (out_name, out_name.replace(".xlsx", f"_{dt.datetime.now():%H%M%S}.xlsx")):
            try:
                save_xlsx(attempt)
                print(f"\nSaved: {attempt}")
                break
            except PermissionError:
                print(f"{attempt} is open in Excel - saving under a new name.")
        html_name = out_name.replace(".xlsx", ".html")
        try:
            write_html(html_name, sec_show, strongest, emerging, top_html, all_show, len(stocks), last_day,
                       xlsx_href=out_name)
            print(f"Saved web page: {html_name}  (double-click it to open in your browser)")
        except Exception as e:  # the page is a bonus: never lose the Excel because of it
            print("Could not write the web page:", e)
    print("\nMAK Capital | Growing Wealth Together")


if __name__ == "__main__":
    main()
