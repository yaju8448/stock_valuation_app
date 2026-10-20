import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker_symbol: str):
    """Fetch financial data needed for DCF, Graham formulas, and quarterly breakdown."""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    # Current Price
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not current_price:
        raise ValueError(f"Could not retrieve price for ticker: {ticker_symbol}")

    company_name = info.get("shortName", ticker_symbol)
    currency = info.get("currency", "USD")
    shares_outstanding = info.get("sharesOutstanding")

    # Earnings & Book Value for Graham
    eps_trailing = info.get("trailingEps", 0.0) or 0.0
    book_value_per_share = info.get("bookValue", 0.0) or 0.0

    # Free Cash Flow calculation from Annual Cash Flow statement
    cf = stock.cashflow
    fcf = 0.0
    if cf is not None and not cf.empty:
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0]
        elif (
            "Operating Cash Flow" in cf.index
            and "Capital Expenditure" in cf.index
        ):
            ocf = cf.loc["Operating Cash Flow"].iloc[0]
            capex = cf.loc["Capital Expenditure"].iloc[0]
            fcf = ocf + capex
        elif (
            "Total Cash From Operating Activities" in cf.index
            and "Capital Expenditures" in cf.index
        ):
            ocf = cf.loc["Total Cash From Operating Activities"].iloc[0]
            capex = cf.loc["Capital Expenditures"].iloc[0]
            fcf = ocf + capex

    # Default fallback for growth rate from analyst estimates if available
    growth_rate = (
        info.get("earningsGrowth", 0.08)
        if info.get("earningsGrowth") is not None
        else 0.08
    )
    # Clamp extreme growth expectations to realistic bounds (3% - 25%)
    growth_rate = max(0.03, min(growth_rate, 0.25))

    # --- Unit scaling: Crores for INR (10^7), Millions for other currencies (10^6) ---
    is_inr = currency.upper() == "INR"
    scale_factor = 10_000_000.0 if is_inr else 1_000_000.0
    unit_label = "₹ Cr" if is_inr else f"{currency} M"

    # --- Fetch Quarterly Financials ---
    q_income = stock.quarterly_income_stmt
    q_cf = stock.quarterly_cashflow

    quarterly_records = []
    if q_income is not None and not q_income.empty:
        for col_date in q_income.columns:
            date_str = pd.to_datetime(col_date).strftime("%Y-%m-%d")

            rev = (
                q_income.loc["Total Revenue", col_date]
                if "Total Revenue" in q_income.index
                else 0.0
            )
            op_inc = (
                q_income.loc["Operating Income", col_date]
                if "Operating Income" in q_income.index
                else 0.0
            )
            net_inc = (
                q_income.loc["Net Income", col_date]
                if "Net Income" in q_income.index
                else 0.0
            )

            # Operating Cash Flow from quarterly CF statement
            ocf_q = 0.0
            if q_cf is not None and not q_cf.empty and col_date in q_cf.columns:
                if "Operating Cash Flow" in q_cf.index:
                    ocf_q = q_cf.loc["Operating Cash Flow", col_date]
                elif "Total Cash From Operating Activities" in q_cf.index:
                    ocf_q = q_cf.loc["Total Cash From Operating Activities", col_date]

            quarterly_records.append(
                {
                    "Quarter Ended": date_str,
                    f"Revenue ({unit_label})": round(float(rev) / scale_factor, 2),
                    f"Operating Income ({unit_label})": round(float(op_inc) / scale_factor, 2),
                    f"Net Income ({unit_label})": round(float(net_inc) / scale_factor, 2),
                    f"Operating CF ({unit_label})": round(float(ocf_q) / scale_factor, 2),
                }
            )

    quarterly_df = (
        pd.DataFrame(quarterly_records)
        if quarterly_records
        else pd.DataFrame()
    )

    return {
        "ticker": ticker_symbol.upper(),
        "name": company_name,
        "currency": currency,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "fcf": fcf,
        "eps": eps_trailing,
        "bvps": book_value_per_share,
        "estimated_growth": growth_rate,
        "quarterly_df": quarterly_df,
        "is_inr": is_inr,
        "scale_factor": scale_factor,
        "unit_label": unit_label,
    }


def calculate_dcf(
    fcf: float,
    shares: int,
    growth_rate: float,
    discount_rate: float = 0.09,
    terminal_growth: float = 0.025,
    projection_years: int = 5,
):
    """5-year DCF model using projected Free Cash Flow and Terminal Value."""
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return 0.0, []

    projected_fcf = []
    current_fcf = fcf

    total_pv_fcf = 0.0
    for year in range(1, projection_years + 1):
        current_fcf *= 1 + growth_rate
        pv = current_fcf / ((1 + discount_rate) ** year)
        projected_fcf.append(
            {
                "Year": f"Year {year}",
                "Projected FCF": round(current_fcf, 2),
                "Discount Factor": round((1 + discount_rate) ** year, 4),
                "PV of FCF": round(pv, 2),
            }
        )
        total_pv_fcf += pv

    terminal_value = (
        current_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    )
    pv_terminal_value = terminal_value / (
        (1 + discount_rate) ** projection_years
    )

    total_equity_value = total_pv_fcf + pv_terminal_value
    intrinsic_value_per_share = total_equity_value / shares

    return round(intrinsic_value_per_share, 2), projected_fcf


def calculate_graham(
    eps: float,
    growth_rate: float,
    bvps: float,
    aaa_corporate_yield: float = 4.4,
):
    """Benjamin Graham's Revised Intrinsic Value Formula & Classic Graham Number."""
    g_percent = growth_rate * 100

    if eps > 0:
        graham_revised = (eps * (8.5 + 2 * g_percent) * 4.4) / aaa_corporate_yield
    else:
        graham_revised = 0.0

    if eps > 0 and bvps > 0:
        graham_number = (22.5 * eps * bvps) ** 0.5
    else:
        graham_number = 0.0

    return round(graham_revised, 2), round(graham_number, 2)


def generate_valuation_report(ticker_symbol: str):
    print(f"\n[+] Fetching financial data for: {ticker_symbol.upper()}...")
    try:
        data = fetch_stock_data(ticker_symbol)
    except Exception as e:
        if "." not in ticker_symbol:
            print(f"Retrying with NSE extension: {ticker_symbol.upper()}.NS ...")
            try:
                data = fetch_stock_data(f"{ticker_symbol}.NS")
            except Exception as inner_e:
                print(f"Error fetching data: {inner_e}")
                return
        else:
            print(f"Error fetching data: {e}")
            return

    # Valuation parameters
    discount_rate = 0.09
    terminal_growth = 0.025
    growth_rate = data["estimated_growth"]

    # Compute models
    dcf_value, dcf_table = calculate_dcf(
        fcf=data["fcf"],
        shares=data["shares_outstanding"],
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )

    graham_revised, graham_number = calculate_graham(
        eps=data["eps"], growth_rate=growth_rate, bvps=data["bvps"]
    )

    valid_values = [v for v in [dcf_value, graham_revised] if v > 0]
    combined_intrinsic = (
        round(sum(valid_values) / len(valid_values), 2)
        if valid_values
        else 0.0
    )

    current_price = data["current_price"]
    upside_downside = (
        round(((combined_intrinsic - current_price) / current_price) * 100, 2)
        if combined_intrinsic > 0
        else 0.0
    )

    verdict = (
        "Undervalued (Buy / Deep Value)"
        if upside_downside > 15
        else (
            "Overvalued (Caution / Sell)"
            if upside_downside < -15
            else "Fairly Valued"
        )
    )

    # Scaled FCF display
    fcf_in_scale = data["fcf"] / data["scale_factor"]
    unit_str = "Crores" if data["is_inr"] else "Millions"

    # 1. Console Output: Valuation Parameters & Assumptions
    print("\n" + "=" * 70)
    print("KEY INPUT PARAMETERS & VALUATION ASSUMPTIONS")
    print("=" * 70)
    print(f"Base Free Cash Flow (FCF):   {data['currency']} {fcf_in_scale:,.2f} {unit_str} ({data['currency']} {data['fcf']:,.0f})")
    print(f"5-Year Growth Rate Assumed:  {growth_rate * 100:.2f}%")
    print(f"Discount Rate (WACC):        {discount_rate * 100:.2f}%")
    print(f"Terminal Growth Rate:        {terminal_growth * 100:.2f}%")
    print(f"Shares Outstanding:          {data['shares_outstanding']:,}")

    # 2. Console Output: Quarterly Financial Breakdown
    print("\n" + "=" * 70)
    print(f"QUARTERLY PERFORMANCE SUMMARY (All monetary values in {data['unit_label']})")
    print("=" * 70)
    if not data["quarterly_df"].empty:
        print(data["quarterly_df"].to_string(index=False))
    else:
        print("Quarterly financial records not available.")

    # 3. Console Output: Final Valuation Summary
    print("\n" + "=" * 70)
    print(f"VALUATION SUMMARY: {data['name']} ({data['ticker']})")
    print("=" * 70)
    print(f"Current Price:               {data['currency']} {current_price:.2f}")
    print(
        f"DCF Intrinsic Value:         {data['currency']} {dcf_value:.2f}"
        if dcf_value > 0
        else "DCF Intrinsic Value:         N/A (Negative or missing FCF)"
    )
    print(
        f"Graham Revised Value:        {data['currency']} {graham_revised:.2f}"
        if graham_revised > 0
        else "Graham Revised Value:        N/A (Negative EPS)"
    )
    print(
        f"Graham Number (Classic):     {data['currency']} {graham_number:.2f}"
        if graham_number > 0
        else "Graham Number (Classic):     N/A"
    )
    print("-" * 70)
    print(f"COMBINED INTRINSIC VALUE:    {data['currency']} {combined_intrinsic:.2f}")
    print(f"Implied Upside/Downside:      {upside_downside:+.2f}%")
    print(f"Valuation Assessment:        {verdict}")
    print("=" * 70)

    # Ask user before exporting to Excel
    user_choice = input(
        "\nType 'output' to export to Excel (or press Enter to skip): "
    ).strip().lower()
    if user_choice == "output":
        export_to_excel(
            data=data,
            dcf_val=dcf_value,
            graham_rev=graham_revised,
            graham_num=graham_number,
            combined_val=combined_intrinsic,
            upside=upside_downside,
            verdict=verdict,
            dcf_table=dcf_table,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )
    else:
        print("[!] Excel export skipped.")


def export_to_excel(
    data,
    dcf_val,
    graham_rev,
    graham_num,
    combined_val,
    upside,
    verdict,
    dcf_table,
    discount_rate,
    terminal_growth,
):
    filename = f"{data['ticker']}_Valuation_Report.xlsx"
    unit_str = "Crores" if data["is_inr"] else "Millions"
    fcf_scaled = round(data["fcf"] / data["scale_factor"], 2)

    summary_data = {
        "Metric": [
            "Company Name",
            "Ticker",
            "Currency",
            "Current Market Price",
            f"Base Free Cash Flow ({data['unit_label']})",
            "Base Free Cash Flow (Raw)",
            "Assumed 5-Year Growth Rate",
            "Discount Rate (WACC)",
            "Terminal Growth Rate",
            "Shares Outstanding",
            "Trailing EPS",
            "Book Value Per Share (BVPS)",
            "DCF Fair Value",
            "Graham Revised Fair Value",
            "Graham Number (Classic)",
            "Blended Intrinsic Value",
            "Margin of Safety / Upside",
            "Final Recommendation",
        ],
        "Value": [
            data["name"],
            data["ticker"],
            data["currency"],
            data["current_price"],
            f"{fcf_scaled} {unit_str}",
            data["fcf"],
            f"{data['estimated_growth'] * 100:.2f}%",
            f"{discount_rate * 100:.2f}%",
            f"{terminal_growth * 100:.2f}%",
            data["shares_outstanding"],
            data["eps"],
            data["bvps"],
            dcf_val if dcf_val > 0 else "N/A",
            graham_rev if graham_rev > 0 else "N/A",
            graham_num if graham_num > 0 else "N/A",
            combined_val,
            f"{upside:+.2f}%",
            verdict,
        ],
    }

    df_summary = pd.DataFrame(summary_data)
    df_dcf = (
        pd.DataFrame(dcf_table)
        if dcf_table
        else pd.DataFrame({"Note": ["DCF unavailable due to non-positive FCF"]})
    )
    df_quarterly = data["quarterly_df"]

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Valuation Summary", index=False)
        df_quarterly.to_excel(writer, sheet_name="Quarterly Financials (Cr)", index=False)
        df_dcf.to_excel(writer, sheet_name="5Y DCF Projections", index=False)

    print(f"\n[✓] Excel file generated successfully: {os.path.abspath(filename)}\n")


if __name__ == "__main__":
    ticker_input = input("Enter Stock Ticker (e.g., RELIANCE, TCS, INFY.NS, AAPL): ").strip()
    if ticker_input:
        generate_valuation_report(ticker_input)
    else:
        print("No ticker entered. Exiting.")
import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker_symbol: str):
    """Fetch financial data needed for DCF, Graham formulas, and quarterly breakdown."""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    # Current Price
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not current_price:
        raise ValueError(f"Could not retrieve price for ticker: {ticker_symbol}")

    company_name = info.get("shortName", ticker_symbol)
    currency = info.get("currency", "USD")
    shares_outstanding = info.get("sharesOutstanding")

    # Earnings & Book Value for Graham
    eps_trailing = info.get("trailingEps", 0.0) or 0.0
    book_value_per_share = info.get("bookValue", 0.0) or 0.0

    # Free Cash Flow calculation from Annual Cash Flow statement
    cf = stock.cashflow
    fcf = 0.0
    if cf is not None and not cf.empty:
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0]
        elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            ocf = cf.loc["Operating Cash Flow"].iloc[0]
            capex = cf.loc["Capital Expenditure"].iloc[0]
            fcf = ocf + capex
        elif "Total Cash From Operating Activities" in cf.index and "Capital Expenditures" in cf.index:
            ocf = cf.loc["Total Cash From Operating Activities"].iloc[0]
            capex = cf.loc["Capital Expenditures"].iloc[0]
            fcf = ocf + capex

    # Growth rate bounds
    growth_rate = info.get("earningsGrowth", 0.08) if info.get("earningsGrowth") is not None else 0.08
    growth_rate = max(0.03, min(growth_rate, 0.25))

    # Unit scaling (Crores for INR)
    is_inr = currency.upper() == "INR"
    scale_factor = 10_000_000.0 if is_inr else 1_000_000.0
    unit_label = "₹ Cr" if is_inr else f"{currency} M"

    # ==========================================
    # --- ROBUST QUARTERLY FINANCIALS MERGE ---
    # ==========================================
    # Yahoo Finance ke multiple tables merge karenge taaki September 2025 jaisa gap cover ho sake
    q_income = stock.quarterly_income_stmt
    if q_income is None or q_income.empty:
        q_income = stock.quarterly_financials

    # Cashflow table
    q_cf = stock.quarterly_cashflow

    # Combine all unique quarter end dates from both income & cashflow statements
    available_dates = set()
    if q_income is not None and not q_income.empty:
        available_dates.update(q_income.columns)
    if q_cf is not None and not q_cf.empty:
        available_dates.update(q_cf.columns)

    # Sort dates latest to oldest
    sorted_dates = sorted(list(available_dates), key=lambda d: pd.to_datetime(d), reverse=True)

    quarterly_records = []
    for col_date in sorted_dates:
        date_str = pd.to_datetime(col_date).strftime("%Y-%m-%d")

        # 1. Total Revenue
        rev = 0.0
        if q_income is not None and not q_income.empty and col_date in q_income.columns:
            for rev_label in ["Total Revenue", "Operating Revenue", "Revenue"]:
                if rev_label in q_income.index:
                    val = q_income.loc[rev_label, col_date]
                    rev = val if pd.notnull(val) else 0.0
                    break

        # 2. Operating Income
        op_inc = 0.0
        if q_income is not None and not q_income.empty and col_date in q_income.columns:
            for op_label in ["Operating Income", "Operating Profit", "EBIT"]:
                if op_label in q_income.index:
                    val = q_income.loc[op_label, col_date]
                    op_inc = val if pd.notnull(val) else 0.0
                    break

        # 3. Net Income
        net_inc = 0.0
        if q_income is not None and not q_income.empty and col_date in q_income.columns:
            for net_label in ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]:
                if net_label in q_income.index:
                    val = q_income.loc[net_label, col_date]
                    net_inc = val if pd.notnull(val) else 0.0
                    break

        # 4. Operating Cash Flow
        ocf_q = 0.0
        if q_cf is not None and not q_cf.empty and col_date in q_cf.columns:
            for cf_label in ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]:
                if cf_label in q_cf.index:
                    val = q_cf.loc[cf_label, col_date]
                    ocf_q = val if pd.notnull(val) else 0.0
                    break

        quarterly_records.append({
            "Quarter Ended": date_str,
            f"Revenue ({unit_label})": round(float(rev) / scale_factor, 2),
            f"Operating Income ({unit_label})": round(float(op_inc) / scale_factor, 2),
            f"Net Income ({unit_label})": round(float(net_inc) / scale_factor, 2),
            f"Operating CF ({unit_label})": round(float(ocf_q) / scale_factor, 2),
        })

    quarterly_df = pd.DataFrame(quarterly_records) if quarterly_records else pd.DataFrame()

    return {
        "ticker": ticker_symbol.upper(),
        "name": company_name,
        "currency": currency,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "fcf": fcf,
        "eps": eps_trailing,
        "bvps": book_value_per_share,
        "estimated_growth": growth_rate,
        "quarterly_df": quarterly_df,
        "is_inr": is_inr,
        "scale_factor": scale_factor,
        "unit_label": unit_label,
    }
import os
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup


def fetch_from_screener(ticker_symbol: str):
    """
    Directly scrapes Screener.in quarterly results table for Indian stocks.
    Returns cleaned DataFrame in ₹ Crores.
    """
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
            # Standalone fallback if consolidated doesn't exist
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

        # Parse Screener HTML table
        headers_row = [th.text.strip() for th in table.find("thead").find_all("th") if th.text.strip()]
        data_rows = []
        for tr in table.find("tbody").find_all("tr"):
            row = [td.text.strip() for td in tr.find_all("td")]
            if row:
                data_rows.append(row)

        df_raw = pd.DataFrame(data_rows, columns=headers_row)
        df_raw.set_index(df_raw.columns[0], inplace=True)

        # Transpose so Quarters become rows
        df_transposed = df_raw.T
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={"index": "Quarter Ended"}, inplace=True)

        # Standardize column mappings from Screener
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
                    df_transposed[screener_col]
                    .astype(str)
                    .str.replace(",", "")
                    .str.replace("%", "")
                )
                df_transposed[display_col] = pd.to_numeric(df_transposed[screener_col], errors="coerce").fillna(0.0)
                cols_to_keep.append(display_col)

        # Return latest 8 quarters
        return df_transposed[cols_to_keep].tail(8).iloc[::-1].reset_index(drop=True)

    except Exception:
        return None


def fetch_stock_data(ticker_symbol: str):
    """Fetch financial data needed for DCF and quarterly analysis with Screener fallback."""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not current_price:
        raise ValueError(f"Could not retrieve price for ticker: {ticker_symbol}")

    company_name = info.get("shortName", ticker_symbol)
    currency = info.get("currency", "USD")
    shares_outstanding = info.get("sharesOutstanding")

    eps_trailing = info.get("trailingEps", 0.0) or 0.0
    book_value_per_share = info.get("bookValue", 0.0) or 0.0

    # Annual FCF Calculation
    cf = stock.cashflow
    fcf = 0.0
    if cf is not None and not cf.empty:
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0]
        elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            ocf = cf.loc["Operating Cash Flow"].iloc[0]
            capex = cf.loc["Capital Expenditure"].iloc[0]
            fcf = ocf + capex

    growth_rate = info.get("earningsGrowth", 0.08) if info.get("earningsGrowth") is not None else 0.08
    growth_rate = max(0.03, min(growth_rate, 0.25))

    is_inr = currency.upper() == "INR"
    scale_factor = 10_000_000.0 if is_inr else 1_000_000.0
    unit_label = "₹ Cr" if is_inr else f"{currency} M"

    # --- Step 1: Check Screener for accurate Indian quarterly numbers ---
    quarterly_df = pd.DataFrame()
    data_source = "Yahoo Finance"

    if is_inr:
        print("[*] Fetching verified quarterly data from Screener.in...")
        screener_df = fetch_from_screener(ticker_symbol)
        if screener_df is not None and not screener_df.empty:
            quarterly_df = screener_df
            data_source = "Screener.in (Verified)"

    # --- Step 2: Fallback to yfinance if Screener unavailable ---
    if quarterly_df.empty:
        q_income = stock.quarterly_income_stmt
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

            rev = (
                q_income.loc["Total Revenue", col_date]
                if q_income is not None and "Total Revenue" in q_income.index and col_date in q_income.columns
                else 0.0
            )
            op_inc = (
                q_income.loc["Operating Income", col_date]
                if q_income is not None and "Operating Income" in q_income.index and col_date in q_income.columns
                else 0.0
            )
            net_inc = (
                q_income.loc["Net Income", col_date]
                if q_income is not None and "Net Income" in q_income.index and col_date in q_income.columns
                else 0.0
            )

            # Check Cash Flow or use Operating EBITDA Proxy if reported as 0
            ocf_q = 0.0
            if q_cf is not None and not q_cf.empty and col_date in q_cf.columns:
                if "Operating Cash Flow" in q_cf.index:
                    ocf_q = q_cf.loc["Operating Cash Flow", col_date]

            quarterly_records.append({
                "Quarter Ended": date_str,
                f"Revenue ({unit_label})": round(float(rev) / scale_factor, 2),
                f"Operating Income ({unit_label})": round(float(op_inc) / scale_factor, 2),
                f"Net Income ({unit_label})": round(float(net_inc) / scale_factor, 2),
                f"Operating CF ({unit_label})": round(float(ocf_q) / scale_factor, 2),
            })

        quarterly_df = pd.DataFrame(quarterly_records)

    return {
        "ticker": ticker_symbol.upper(),
        "name": company_name,
        "currency": currency,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "fcf": fcf,
        "eps": eps_trailing,
        "bvps": book_value_per_share,
        "estimated_growth": growth_rate,
        "quarterly_df": quarterly_df,
        "data_source": data_source,
        "is_inr": is_inr,
        "scale_factor": scale_factor,
        "unit_label": unit_label,
    }


if __name__ == "__main__":
    print("=" * 70)her
    print("  STOCK VALUATION & QUARTERLY ANALYSIS TOOL")
    print("  (Type ticker and press Enter. Type 'exit' or 'q' to quit)")
    print("=" * 70)

    while True:
        ticker_input = input("\nEnter Stock Ticker (e.g., RELIANCE, TCS, INFY.NS, AAPL) [or 'exit']: ").strip()

        # Check for exit commands
        if ticker_input.lower() in ["exit", "quit", "q"]:
            print("\nExiting valuation program. Goodbye!")
            break

        # If user presses Enter without typing anything
        if not ticker_input:
            print("[!] Empty input. Please enter a valid stock ticker or 'exit'.")
            continue

        # Run valuation report
        generate_valuation_report(ticker_input)

        # Prompt for next stock
        print("\n" + "-" * 70)
        print("Analysis complete. Press Enter to analyze the next stock...")
