
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="台股投資決策系統 V3.0",
    page_icon="📈",
    layout="wide"
)
@st.cache_data(ttl=900)
def get_stock(code):
    code = code.strip().upper()

    if code.endswith((".TW", ".TWO")):
        symbols = [code]
    else:
        symbols = [code + ".TW", code + ".TWO"]

    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                period="3y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty:
                df = df.dropna(subset=["Close"])

                if not df.empty:
                    return symbol, df

        except Exception:
            continue

    return None, pd.DataFrame()


def calculate_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    for n in [5, 10, 20, 60, 120, 240]:
        df[f"MA{n}"] = close.rolling(n).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(
        span=9,
        adjust=False
    ).mean()
    df["MACD"] = df["DIF"] - df["DEA"]

    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    rsv = (
        (close - lowest)
        / (highest - lowest).replace(0, np.nan)
        * 100
    )

    df["K"] = rsv.rolling(3).mean()
    df["D"] = df["K"].rolling(3).mean()

    df["BB_MID"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    df["BB_UPPER"] = df["BB_MID"] + 2 * std20
    df["BB_LOWER"] = df["BB_MID"] - 2 * std20

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    df["VOL5"] = volume.rolling(5).mean()
    df["VOL20"] = volume.rolling(20).mean()

    df["RETURN"] = close.pct_change()

    df["VOLATILITY"] = (
        df["RETURN"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    peak = close.cummax()
    df["DRAWDOWN"] = (close / peak - 1) * 100

    return df


def value(row, key):
    try:
        x = float(row[key])
        return x if np.isfinite(x) else None
    except Exception:
        return None


def display(x, suffix="", digits=2):
    if x is None or not np.isfinite(x):
        return "資料不足"

    return f"{x:,.{digits}f}{suffix}"


def analyze(df):
    row = df.iloc[-1]

    close = value(row, "Close")
    ma20 = value(row, "MA20")
    ma60 = value(row, "MA60")
    ma240 = value(row, "MA240")
    rsi = value(row, "RSI")
    k = value(row, "K")
    d = value(row, "D")
    macd = value(row, "MACD")
    atr = value(row, "ATR")

    positive = []
    negative = []

    if ma20 is not None:
        if close > ma20:
            positive.append("股價站上月線")
        else:
            negative.append("股價跌破月線")

    if ma60 is not None:
        if close > ma60:
            positive.append("股價站上季線")
        else:
            negative.append("股價跌破季線")

    if ma240 is not None:
        if close > ma240:
            positive.append("股價站上年線")
        else:
            negative.append("股價跌破年線")

    if rsi is not None:
        if 50 <= rsi <= 70:
            positive.append("RSI動能偏多")
        elif rsi >= 75:
            negative.append("RSI處於過熱區")
        elif rsi < 35:
            negative.append("RSI動能偏弱")

    if k is not None and d is not None:
        if k > d:
            positive.append("KD偏多")
        else:
            negative.append("KD偏弱")

    if macd is not None:
        if macd > 0:
            positive.append("MACD柱狀體為正")
        else:
            negative.append("MACD柱狀體為負")

    score = len(positive)
    risk = len(negative)

    if risk >= 5:
        signal = "🔴 技術風險偏高"
    elif risk >= 3:
        signal = "🟡 多空訊號混合"
    elif score >= 4:
        signal = "🟢 技術結構偏多"
    else:
        signal = "⚪ 中性觀察"

    return {
        "signal": signal,
        "score": score,
        "risk": risk,
        "positive": positive,
        "negative": negative,
        "atr": atr
    }


def create_chart(df, symbol):
    data = df.tail(180)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.22, 0.20]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="K線"
        ),
        row=1,
        col=1
    )

    for ma in [5, 20, 60, 120, 240]:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f"MA{ma}"],
                name=f"MA{ma}",
                mode="lines"
            ),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="成交量"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["DIF"],
            name="DIF"
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["DEA"],
            name="DEA"
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD"],
            name="MACD柱"
        ),
        row=3,
        col=1
    )

    fig.update_layout(
        title=f"{symbol} K線與技術分析",
        height=750,
        xaxis_rangeslider_visible=False
    )

    return fig

code = st.text_input(
    "請輸入股票代號",
    placeholder="例如：2330、0050、2317",
    key="stock_search_main"
)
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

start = st.button(
    "🔍 開始分析",
    type="primary",
    use_container_width=True
)

if start:
    st.session_state.analysis_started = True

if st.session_state.analysis_started:
    if not code.strip():
        st.warning("請先輸入股票代號。")
    else:
        with st.spinner("正在下載股票資料並分析..."):
            symbol, raw = get_stock(code)

        if raw.empty:
            st.error("查無資料，請檢查股票代號或網路連線。")
        else:
            df = calculate_indicators(raw)
            result = analyze(df)
            latest = df.iloc[-1]

            close = value(latest, "Close")
            previous = (
                float(df["Close"].iloc[-2])
                if len(df) >= 2
                else None
            )

            change = (
                (close / previous - 1) * 100
                if previous and close is not None
                else None
            )

            st.success(f"{symbol} 分析完成")

            st.caption(
                "資料日期："
                + df.index[-1].strftime("%Y-%m-%d")
                + "｜非即時報價"
            )

            a, b, c, d = st.columns(4)

            a.metric("收盤價", display(close))
            b.metric("日漲跌幅", display(change, "%"))
            c.metric("技術訊號", result["signal"])
            d.metric("ATR", display(result["atr"]))

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "📊 K線圖",
                    "📋 技術指標",
                    "🛡️ 風險分析",
                    "💰 持股試算"
                ]
            )

            with tab1:
                st.plotly_chart(
                    create_chart(df, symbol),
                    use_container_width=True
                )

            with tab2:
                indicators = [
                    ("收盤價", "Close"),
                    ("MA5", "MA5"),
                    ("MA10", "MA10"),
                    ("MA20 月線", "MA20"),
                    ("MA60 季線", "MA60"),
                    ("MA120 半年線", "MA120"),
                    ("MA240 年線", "MA240"),
                    ("RSI", "RSI"),
                    ("K值", "K"),
                    ("D值", "D"),
                    ("MACD DIF", "DIF"),
                    ("MACD DEA", "DEA"),
                    ("MACD柱", "MACD"),
                    ("布林上軌", "BB_UPPER"),
                    ("布林中軌", "BB_MID"),
                    ("布林下軌", "BB_LOWER"),
                    ("ATR", "ATR"),
                    ("20日年化波動率", "VOLATILITY"),
                    ("成交量", "Volume"),
                    ("5日均量", "VOL5"),
                    ("20日均量", "VOL20")
                ]

                table = pd.DataFrame(
                    [
                        {
                            "指標": name,
                            "數值": display(
                                value(latest, key),
                                "%" if key == "VOLATILITY" else ""
                            )
                        }
                        for name, key in indicators
                    ]
                )

                st.dataframe(
                    table,
                    hide_index=True,
                    use_container_width=True
                )

                st.download_button(
                    "下載技術指標 CSV",
                    table.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{symbol}_技術分析.csv",
                    mime="text/csv"
                )

            with tab3:
                st.subheader(result["signal"])

                st.write("技術偏多訊號數：", result["score"])
                st.write("技術風險訊號數：", result["risk"])

                st.markdown("#### 偏多理由")
                if result["positive"]:
                    for item in result["positive"]:
                        st.write("✅", item)
                else:
                    st.write("暫無明顯偏多訊號")

                st.markdown("#### 風險理由")
                if result["negative"]:
                    for item in result["negative"]:
                        st.write("⚠️", item)
                else:
                    st.write("暫無主要技術風險訊號")

                drawdown = value(latest, "DRAWDOWN")
                maximum = float(df["DRAWDOWN"].min())

                st.metric(
                    "目前相對歷史高點回撤",
                    display(drawdown, "%")
                )

                st.metric(
                    "資料期間最大回撤",
                    display(maximum, "%")
                )

                st.info(
                    "技術訊號數量不代表未來上漲機率，"
                    "也不能單獨作為買賣依據。"
                )

            with tab4:
                cost = st.number_input(
                    "平均買進成本（每股）",
                    min_value=0.0,
                    value=0.0,
                    step=0.5
                )

                lots = st.number_input(
                    "持有張數（1張＝1000股）",
                    min_value=0.0,
                    value=0.0,
                    step=1.0
                )

                shares = lots * 1000
                invested = cost * shares
                market_value = close * shares
                profit = market_value - invested

                rate = (
                    profit / invested * 100
                    if invested > 0
                    else None
                )

                x, y, z = st.columns(3)

                x.metric(
                    "投入成本",
                    f"NT$ {invested:,.0f}"
                )

                y.metric(
                    "目前市值",
                    f"NT$ {market_value:,.0f}"
                )

                z.metric(
                    "未實現損益",
                    f"NT$ {profit:+,.0f}",
                    delta=display(rate, "%")
                )

                if cost > 0:
                    st.write(
                        "成本下方8%參考價：",
                        display(cost * 0.92)
                    )

                    st.write(
                        "成本上方10%參考價：",
                        display(cost * 1.10)
                    )

                    st.write(
                        "成本上方20%參考價：",
                        display(cost * 1.20)
                    )

                st.caption(
                    "持股試算未扣除手續費、交易稅，"
                    "亦未計入實際領取股利。"
                )

st.divider()

st.caption(
    "資料來源：Yahoo Finance。"
    "本系統僅供研究與學習，不構成投資建議。"
)
