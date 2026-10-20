import io
import os
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go

st.set_page_config(
    page_title="Intrinsic Value & Stock Screener",
    page_icon="📈",
    layout="wide"
)

# Custom HTTP session with realistic browser headers to avoid 401/429 Cloud blocks
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    })
    return session


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_from_screener(ticker_symbol: str):
    """Directly scrapes Screener.in quarterly results table for Indian stocks."""
    clean_ticker = ticker_symbol.replace(".NS", "").replace(".BO", "").upper()
    url = f"https://www.screener.in/company/{clean_ticker}/consolidated/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            url = f"https://www.screener.in/company/{clean_ticker}/"
            resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        section = soup.find("section", {"id": "quarters"})
        if not section:
            return None

        table = section.find("table")
        if not table:
            return None

        headers_row = [th.text.strip() for th in table.find("thead").find_all("th") if th.text.strip()]
        data_rows = []
        for tr in table.find("tbody").find_all("tr"):
            row = [td.text.strip() for td in tr.find_all("td")]
            if row:
                data_rows.append(row)

        df_raw = pd.DataFrame(data_rows, columns=headers_row)
        df_raw.set_index(df_raw.columns[0], inplace=True)

        df_transposed = df_raw.T.reset_index()
        df_transposed.rename(columns={"index": "Quarter Ended"}, inplace=True)

        mapping = {
            "Sales": "Revenue (₹ Cr)",
            "Operating Profit": "Operating Income (₹ Cr)",
            "Net Profit": "Net Income (₹ Cr)",
            "OPM %": "OPM (%)",
        }
        cols_to_keep = ["Quarter Ended"]
        for screener_col, display_col in mapping.items():
            if screener_col in df_transposed.columns:
                df_transposed[screener_col] = (
                    df_transposed[screener_col].astype(str).str.replace(",", "").str.replace("%", "")
                )
                df_transposed[display_col] = pd.to_numeric(df_transposed[screener_col], errors="coerce").fillna(0.0)
                cols_to_keep.append(display_col)

        return df_transposed[cols_to_keep].tail(8).iloc[::-1].reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(ticker_symbol: str):
    symbol = ticker_symbol.strip().upper()
    session = get_session()
    stock = yf.Ticker(symbol, session=session)

    # 1. Fetch current price via fast_info to bypass cloud bot blocking
    current_price = None
    try:
        fast = stock.fast_info
        current_price = fast.get("lastPrice") or fast.get("regularMarketPreviousClose")
    except Exception:
        pass

    # 2. NSE retry if ticker was entered without extension
    if not current_price and "." not in symbol:
        symbol = f"{symbol}.NS"
        stock = yf.Ticker(symbol, session=session)
        try:
            fast = stock.fast_info
            current_price = fast.get("lastPrice") or fast.get("regularMarketPreviousClose")
        except Exception:
            pass

    # 3. Fallback to full info dictionary
    info = {}
    try:
        info = stock.info or {}
        if not current_price:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception:
        pass

    if not current_price:
        raise ValueError(
            f"Could not retrieve stock data for '{ticker_symbol}'. "
            f"Ensure the stock is publicly listed on NSE/BSE/US exchanges and the ticker symbol is correct."
        )

    company_name = info.get("shortName", symbol)
    currency = getattr(stock.fast_info, "currency", None) or info.get("currency", "INR" if symbol.endswith(".NS") else "USD")
    shares_outstanding = getattr(stock.fast_info, "shares", None) or info.get("sharesOutstanding") or 0

    eps_trailing = info.get("trailingEps", 0.0) or 0.0
    book_value_per_share = info.get("bookValue", 0.0) or 0.0

    # Free Cash Flow calculation
    fcf = 0.0
    try:
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            if "Free Cash Flow" in cf.index:
                fcf = cf.loc["Free Cash Flow"].iloc[0]
            elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
                fcf = cf.loc["Operating Cash Flow"].iloc[0] + cf.loc["Capital Expenditure"].iloc[0]
            elif "Total Cash From Operating Activities" in cf.index and "Capital Expenditures" in cf.index:
                fcf = cf.loc["Total Cash From Operating Activities"].iloc[0] + cf.loc["Capital Expenditures"].iloc[0]
    except Exception:
        pass

    growth_rate = info.get("earningsGrowth", 0.08) if info.get("earningsGrowth") is not None else 0.08
    growth_rate = max(0.03, min(float(growth_rate), 0.25))

    is_inr = str(currency).upper() == "INR"
    scale_factor = 10_000_000.0 if is_inr else 1_000_000.0
    unit_label = "₹ Cr" if is_inr else f"{currency} M"

    quarterly_df = pd.DataFrame()
    data_source = "Yahoo Finance"

    # Step 1: Use Screener.in for Indian tickers
    if is_inr:
        screener_df = fetch_from_screener(symbol)
        if screener_df is not None and not screener_df.empty:
            quarterly_df = screener_df
            data_source = "Screener.in (Verified)"

    # Step 2: Fallback to Yahoo Finance statements
    if quarterly_df.empty:
        try:
            q_income = stock.quarterly_income_stmt
            if q_income is None or q_income.empty:
                q_income = stock.quarterly_financials
            q_cf = stock.quarterly_cashflow

            available_dates = set()
            if q_income is not None and not q_income.empty:
                available_dates.update(q_income.columns)
            if q_cf is not None and not q_cf.empty:
                available_dates.update(q_cf.columns)

            sorted_dates = sorted(list(available_dates), key=lambda d: pd.to_datetime(d), reverse=True)
            quarterly_records = []

            for col_date in sorted_dates:
                date_str = pd.to_datetime(col_date).strftime("%Y-%m-%d")
                rev, op_inc, net_inc, ocf_q = 0.0, 0.0, 0.0, 0.0

                if q_income is not None and not q_income.empty and col_date in q_income.columns:
                    for l in ["Total Revenue", "Operating Revenue", "Revenue"]:
                        if l in q_income.index:
                            rev = q_income.loc[l, col_date]
                            break
                    for l in ["Operating Income", "Operating Profit", "EBIT"]:
                        if l in q_income.index:
                            op_inc = q_income.loc[l, col_date]
                            break
                    for l in ["Net Income", "Net Income Common Stockholders"]:
                        if l in q_income.index:
                            net_inc = q_income.loc[l, col_date]
                            break

                if q_cf is not None and not q_cf.empty and col_date in q_cf.columns:
                    for l in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                        if l in q_cf.index:
                            ocf_q = q_cf.loc[l, col_date]
                            break

                quarterly_records.append({
                    "Quarter Ended": date_str,
                    f"Revenue ({unit_label})": round(float(rev) / scale_factor, 2),
                    f"Operating Income ({unit_label})": round(float(op_inc) / scale_factor, 2),
                    f"Net Income ({unit_label})": round(float(net_inc) / scale_factor, 2),
                    f"Operating CF ({unit_label})": round(float(ocf_q) / scale_factor, 2),
                })

            quarterly_df = pd.DataFrame(quarterly_records) if quarterly_records else pd.DataFrame()
        except Exception:
            quarterly_df = pd.DataFrame()

    return {
        "ticker": symbol,
        "name": company_name,
        "currency": currency,
        "current_price": float(current_price),
        "shares_outstanding": int(shares_outstanding),
        "fcf": float(fcf),
        "eps": float(eps_trailing),
        "bvps": float(book_value_per_share),
        "estimated_growth": float(growth_rate),
        "quarterly_df": quarterly_df,
        "data_source": data_source,
        "is_inr": is_inr,
        "scale_factor": scale_factor,
        "unit_label": unit_label,
    }


def calculate_dcf(fcf: float, shares: int, growth_rate: float, discount_rate: float = 0.09, terminal_growth: float = 0.025, projection_years: int = 5):
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return 0.0, []

    projected_fcf = []
    current_fcf = fcf
    total_pv_fcf = 0.0

    for year in range(1, projection_years + 1):
        current_fcf *= (1 + growth_rate)
        pv = current_fcf / ((1 + discount_rate) ** year)
        projected_fcf.append({
            "Year": f"Year {year}",
            "Projected FCF": round(current_fcf, 2),
            "Discount Factor": round((1 + discount_rate) ** year, 4),
            "PV of FCF": round(pv, 2),
        })
        total_pv_fcf += pv

    terminal_value = current_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
    total_equity_value = total_pv_fcf + pv_terminal_value
    intrinsic_value_per_share = total_equity_value / shares

    return round(intrinsic_value_per_share, 2), projected_fcf


def calculate_graham(eps: float, growth_rate: float, bvps: float, aaa_corporate_yield: float = 4.4):
    g_percent = growth_rate * 100
    graham_revised = (eps * (8.5 + 2 * g_percent) * 4.4) / aaa_corporate_yield if eps > 0 else 0.0
    graham_number = (22.5 * eps * bvps) ** 0.5 if (eps > 0 and bvps > 0) else 0.0
    return round(graham_revised, 2), round(graham_number, 2)


def generate_excel_bytes(data, dcf_val, graham_rev, graham_num, combined_val, upside, verdict, dcf_table, discount_rate, terminal_growth):
    unit_str = "Crores" if data["is_inr"] else "Millions"
    fcf_scaled = round(data["fcf"] / data["scale_factor"], 2)

    summary_data = {
        "Metric": [
            "Company Name", "Ticker", "Currency", "Current Market Price",
            f"Base Free Cash Flow ({data['unit_label']})", "Base Free Cash Flow (Raw)",
            "Assumed 5-Year Growth Rate", "Discount Rate (WACC)", "Terminal Growth Rate",
            "Shares Outstanding", "Trailing EPS", "Book Value Per Share (BVPS)",
            "DCF Fair Value", "Graham Revised Fair Value", "Graham Number (Classic)",
            "Blended Intrinsic Value", "Margin of Safety / Upside", "Final Recommendation"
        ],
        "Value": [
            data["name"], data["ticker"], data["currency"], data["current_price"],
            f"{fcf_scaled} {unit_str}", data["fcf"],
            f"{data['estimated_growth'] * 100:.2f}%", f"{discount_rate * 100:.2f}%", f"{terminal_growth * 100:.2f}%",
            data["shares_outstanding"], data["eps"], data["bvps"],
            dcf_val if dcf_val > 0 else "N/A", graham_rev if graham_rev > 0 else "N/A", graham_num if graham_num > 0 else "N/A",
            combined_val, f"{upside:+.2f}%", verdict
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_dcf = pd.DataFrame(dcf_table) if dcf_table else pd.DataFrame({"Note": ["DCF unavailable due to non-positive FCF"]})
    df_quarterly = data["quarterly_df"]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Valuation Summary", index=False)
        if not df_quarterly.empty:
            df_quarterly.to_excel(writer, sheet_name="Quarterly Financials", index=False)
        df_dcf.to_excel(writer, sheet_name="5Y DCF Projections", index=False)
    output.seek(0)
    return output


# ---------------- UI Layout ----------------
st.title("📊 DCF & Graham Valuation Dashboard")
st.caption("Perform intrinsic valuation using Discounted Cash Flow and Benjamin Graham formulas.")

with st.sidebar:
    st.header("⚙️ Valuation Parameters")
    ticker_input = st.text_input("Enter Stock Ticker:", value="RELIANCE", help="e.g. RELIANCE, TCS, INFY, AAPL, MSFT")

    st.subheader("Model Overrides")
    override_growth = st.checkbox("Custom Growth Rate")
    custom_growth = st.slider("5Y Growth Rate (%)", min_value=1.0, max_value=40.0, value=10.0, step=0.5) if override_growth else None

    discount_rate = st.slider("Discount Rate / WACC (%)", min_value=5.0, max_value=20.0, value=9.0, step=0.5) / 100.0
    terminal_growth = st.slider("Terminal Growth Rate (%)", min_value=1.0, max_value=6.0, value=2.5, step=0.25) / 100.0
    aaa_yield = st.number_input("AAA Corporate Bond Yield (%)", value=4.4, step=0.1)

    analyze_btn = st.button("Run Valuation", type="primary", use_container_width=True)

if ticker_input:
    with st.spinner(f"Fetching financial data for {ticker_input.upper()}..."):
        try:
            data = fetch_stock_data(ticker_input)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    assumed_growth = (custom_growth / 100.0) if override_growth and custom_growth else data["estimated_growth"]

    dcf_value, dcf_table = calculate_dcf(
        fcf=data["fcf"],
        shares=data["shares_outstanding"],
        growth_rate=assumed_growth,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth
    )

    graham_revised, graham_number = calculate_graham(
        eps=data["eps"],
        growth_rate=assumed_growth,
        bvps=data["bvps"],
        aaa_corporate_yield=aaa_yield
    )

    valid_vals = [v for v in [dcf_value, graham_revised] if v > 0]
    blended_val = round(sum(valid_vals) / len(valid_vals), 2) if valid_vals else 0.0

    curr_p = data["current_price"]
    upside_downside = round(((blended_val - curr_p) / curr_p) * 100, 2) if blended_val > 0 else 0.0

    if upside_downside > 15:
        verdict = "Undervalued (Buy / Deep Value)"
        badge_color = "green"
    elif upside_downside < -15:
        verdict = "Overvalued (Caution / Sell)"
        badge_color = "red"
    else:
        verdict = "Fairly Valued"
        badge_color = "orange"

    st.subheader(f"{data['name']} ({data['ticker']})")
    st.markdown(f"**Data Source:** `{data['data_source']}` | **Currency:** `{data['currency']}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Market Price", f"{data['currency']} {curr_p:,.2f}")
    col2.metric("Blended Intrinsic Value", f"{data['currency']} {blended_val:,.2f}" if blended_val > 0 else "N/A")
    col3.metric("Margin of Safety / Upside", f"{upside_downside:+.2f}%")
    col4.markdown(f"**Assessment**<br><h3 style='margin:0;color:{badge_color};'>{verdict}</h3>", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["💡 Valuation Breakdown", "📈 5-Year DCF Projections", "📋 Quarterly Financials"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            val_df = pd.DataFrame({
                "Valuation Model": ["DCF Fair Value", "Graham Revised", "Classic Graham Number", "Blended Value", "Current Market Price"],
                "Value": [
                    f"{data['currency']} {dcf_value:.2f}" if dcf_value > 0 else "N/A",
                    f"{data['currency']} {graham_revised:.2f}" if graham_revised > 0 else "N/A",
                    f"{data['currency']} {graham_number:.2f}" if graham_number > 0 else "N/A",
                    f"{data['currency']} {blended_val:.2f}" if blended_val > 0 else "N/A",
                    f"{data['currency']} {curr_p:.2f}"
                ]
            })
            st.dataframe(val_df, use_container_width=True, hide_index=True)

            st.write("**Key Assumptions Used:**")
            st.write(f"- 5-Year Growth Rate: **{assumed_growth * 100:.2f}%**")
            st.write(f"- Discount Rate (WACC): **{discount_rate * 100:.2f}%**")
            st.write(f"- Terminal Growth Rate: **{terminal_growth * 100:.2f}%**")
            fcf_scaled = data['fcf'] / data['scale_factor']
            st.write(f"- Base Free Cash Flow: **{data['currency']} {fcf_scaled:,.2f} {data['unit_label']}**")

        with c2:
            fig = go.Figure(data=[
                go.Bar(
                    name="Valuation Models",
                    x=["DCF Value", "Graham Revised", "Current Price", "Blended Value"],
                    y=[dcf_value, graham_revised, curr_p, blended_val],
                    marker_color=["#2b5c8f", "#3b82f6", "#ef4444", "#10b981"]
                )
            ])
            fig.update_layout(title="Fair Value vs Current Market Price", yaxis_title=f"Price ({data['currency']})", height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if dcf_table:
            st.write("Projected Free Cash Flow & Discounted Values:")
            st.dataframe(pd.DataFrame(dcf_table), use_container_width=True, hide_index=True)
        else:
            st.warning("DCF cannot be computed because Base Free Cash Flow (FCF) is negative or missing.")

    with tab3:
        if not data["quarterly_df"].empty:
            st.dataframe(data["quarterly_df"], use_container_width=True, hide_index=True)
        else:
            st.info("No quarterly statements found.")

    excel_file = generate_excel_bytes(
        data=data,
        dcf_val=dcf_value,
        graham_rev=graham_revised,
        graham_num=graham_number,
        combined_val=blended_val,
        upside=upside_downside,
        verdict=verdict,
        dcf_table=dcf_table,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth
    )

    st.download_button(
        label=f"📥 Download {data['ticker']}_Valuation_Report.xlsx",
        data=excel_file,
        file_name=f"{data['ticker']}_Valuation_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
