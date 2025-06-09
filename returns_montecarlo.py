import streamlit as st
import pandas as pd
import numpy as np


rng = np.random.default_rng()


def simulation(win_rate, avg_win, avg_loss, commission, num_trades):
    df = pd.DataFrame(
        rng.choice([1, -1], num_trades, True, [win_rate / 100, 1 - win_rate / 100]),
        columns=["outcome"],
    )
    df["amt"] = np.where(
        df["outcome"] > 0, avg_win - commission, -avg_loss - commission
    )
    df["equity"] = df["amt"].cumsum()
    df.drop(columns=["amt"], inplace=True)
    df["streaks"] = (
        df.groupby((df["outcome"] != df["outcome"].shift()).cumsum()).cumcount() + 1
    ) * df["outcome"]
    win_streak, loss_streak = df["streaks"].max(), df["streaks"].min()
    df.drop(columns=["streaks"], inplace=True)
    win_rate = 100 * sum(df["outcome"] > 0) / num_trades
    return df, win_streak, -loss_streak, win_rate


def simulate_trading(
    win_rate, avg_win, avg_loss, commission, trades_in_trial, num_trials
):
    df_min = pd.DataFrame()
    df_max = pd.DataFrame()
    final_equity_avg = 0
    max_final_equity = -(2**100)
    min_final_equity = 2**100
    peak_equity = -(2**100)
    peak_equity_avg = 0
    low_equity = 2**100
    low_equity_avg = 0
    peak_win_streak = -trades_in_trial
    low_win_streak = trades_in_trial
    win_streak_avg = 0
    peak_loss_streak = -trades_in_trial
    low_loss_streak = trades_in_trial
    loss_streak_avg = 0

    progress_bar = st.progress(0, text="Starting analysis. Please wait...")

    for trial in range(num_trials):
        # Simulate a single trial
        df, win_streak, loss_streak, sim_win_rate = simulation(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            commission=commission,
            num_trades=trades_in_trial,
        )
        final_equity = df["equity"].iloc[-1]
        if df_max.empty or final_equity > df_max["equity"].iloc[-1]:
            df_max = df
        if df_min.empty or final_equity < df_min["equity"].iloc[-1]:
            df_min = df
        final_equity_avg += final_equity
        max_final_equity = max(max_final_equity, final_equity)
        min_final_equity = min(min_final_equity, final_equity)

        max_equity = df["equity"].max()
        min_equity = df["equity"].min()
        peak_equity_avg += max_equity
        low_equity_avg += min_equity

        win_streak_avg += win_streak
        loss_streak_avg += loss_streak
        peak_win_streak = max(peak_win_streak, win_streak)
        peak_loss_streak = max(peak_loss_streak, loss_streak)
        low_win_streak = min(low_win_streak, win_streak)
        low_loss_streak = min(low_loss_streak, loss_streak)
        peak_equity = max(peak_equity, max_equity)
        low_equity = min(low_equity, min_equity)
        progress_bar.progress(
            100 * trial // num_trials, text="Analysis in progress. Please wait."
        )

    final_equity_avg = final_equity_avg / num_trials
    peak_equity_avg = peak_equity_avg / num_trials
    low_equity_avg = low_equity_avg / num_trials
    win_streak_avg = win_streak_avg / num_trials
    loss_streak_avg = loss_streak_avg / num_trials
    stats_df = pd.DataFrame(
        {
            "Max": [
                f"{peak_equity:,.2f}",
                "",
                f"{max_final_equity:,.2f}",
                f"{peak_win_streak}",
                f"{peak_loss_streak}",
            ],
            "Average": [
                f"{peak_equity_avg:,.2f}",
                f"{low_equity_avg:,.2f}",
                f"{final_equity_avg:,.2f}",
                f"{win_streak_avg:.1f}",
                f"{loss_streak_avg:.1f}",
            ],
            "Min": [
                "",
                f"{low_equity:,.2f}",
                f"{min_final_equity:,.2f}",
                f"{low_win_streak}",
                f"{low_loss_streak}",
            ],
        },
        index=[
            "Peak Equity",
            "Low Equity",
            "Final Equity",
            "Simulation Win Streak",
            "Simulation Loss Streak",
        ],
    )
    df["high_equity"] = df_max.equity
    df["low_equity"] = df_min.equity
    df["mid_equity"] = (df["high_equity"] + df["low_equity"]) / 2
    progress_bar.empty()
    return {
        "simulation": df,
        "win_rate": sim_win_rate,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
    }, stats_df


st.set_page_config(page_title="Trading System Monte Carlo Analysis", layout="wide")
st.title(":material/analytics: Trading System Analysis with Monte Carlo Simulations")

c1, c2 = st.columns([2, 8], vertical_alignment="top", gap="medium")
with c1:
    st.subheader(":blue[:material/function: Parameters]", divider=True)

    win_rate = st.slider(
        "Win rate (%):", min_value=0.0, max_value=100.0, value=50.0, step=0.05
    )
    st.warning(f"Loss probability (%): {100.0 - win_rate:.001f}")

    avg_win = st.number_input(
        "Average win ($):", min_value=0.0, value=1000.0, step=0.01
    )
    avg_loss = st.number_input(
        "Average loss ($):", min_value=0.0, value=500.0, step=0.01
    )
    commission = st.number_input("Commission ($):", min_value=0.0, value=0.0, step=0.01)
    st.success(
        f"Expectancy ($): {(avg_win - commission) * win_rate / 100 - (avg_loss + commission) * (100 - win_rate) / 100:.02f}"
    )

    num_simulations = st.slider(
        "Number of simulations:", min_value=100, max_value=3000, value=1000, step=1
    )
    num_trades = st.slider(
        "Number of trades in simulation:",
        min_value=1,
        max_value=1000,
        value=365,
        step=1,
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        btn_refres = st.button(":material/refresh: Single sim", type="secondary")
    with sc2:
        btn_simulate = st.button(":material/play_circle: Run", type="primary")

    if st.session_state.get("results") is None:
        st.session_state.results = {}
        st.session_state.stats = pd.DataFrame()

with c2:
    if btn_simulate:
        st.session_state.results, st.session_state.stats = simulate_trading(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            commission=commission,
            trades_in_trial=num_trades,
            num_trials=num_simulations,
        )
    else:
        c_sim, win_streak, loss_streak, win_rate = simulation(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            commission=commission,
            num_trades=num_trades,
        )
        st.session_state.results["win_streak"] = win_streak
        st.session_state.results["loss_streak"] = loss_streak
        st.session_state.results["win_rate"] = win_rate
        if "simulation" not in st.session_state.results:
            st.session_state.results["simulation"] = c_sim
        else:
            st.session_state.results["simulation"]["equity"] = c_sim["equity"]
            st.session_state.results["simulation"]["outcome"] = c_sim["outcome"]

    if len(st.session_state.results) > 0:
        results = st.session_state.results
        df = results["simulation"]

        headcol = st.columns(3)
        headcol[0].metric(
            label="Final P&L:", value=f"$ {df['equity'].iloc[-1]:,.2f}", border=True
        )
        headcol[1].metric(
            label="Peak Equity:", value=f"$ {df['equity'].max():,.2f}", border=True
        )
        headcol[2].metric(
            label="Lowest Equity:", value=f"$ {df['equity'].min():,.2f}", border=True
        )

        df["Trade"] = np.arange(1, len(df) + 1)
        if "Last" in df.columns:
            df.drop(columns=["Last"], inplace=True)
        df.rename(columns={"equity": "Last"}, inplace=True)
        if "High P&L" not in df.columns:
            if "high_equity" not in df.columns:
                df[["high_equity", "low_equity", "mid_equity"]] = None
            df.rename(
                columns={
                    "high_equity": "High P&L",
                    "low_equity": "Low P&L",
                    "mid_equity": "Mid P&L",
                },
                inplace=True,
            )
        df["Outcome"] = np.where(df["outcome"] > 0, "Win", "Loss")
        df.drop(columns=["outcome"], inplace=True)

        just_last_sim_on_chart = df["High P&L"].isna().all()
        st.line_chart(
            df,
            x="Trade",
            y="Last"
            if just_last_sim_on_chart
            else ["Last", "High P&L", "Low P&L", "Mid P&L"],
            y_label="P & L",
            use_container_width=True,
        )

        if len(st.session_state.stats) > 0:
            tabcol = st.columns([2, 6, 2])
            tabcol[1].table(st.session_state.stats)

        datacols = st.columns(4)
        datacols[0].metric(
            label="Actual Wins:", value=f"{results['win_rate']:,.2f} %", border=True
        )
        datacols[1].metric(
            label="Actual Losses:",
            value=f"{100 - results['win_rate']:,.2f} %",
            border=True,
        )
        datacols[2].metric(
            label="Max Win Streak:",
            value=f"{results['win_streak'] if results['win_streak'] > 0 else 'N/A'}",
            border=True,
        )
        datacols[3].metric(
            label="Max Loss Streak:",
            value=f"{results['loss_streak'] if results['loss_streak'] > 0 else 'N/A'}",
            border=True,
        )

        column_config = {
            "Last": st.column_config.NumberColumn(format="dollar"),
            "High P&L": st.column_config.NumberColumn(format="dollar"),
            "Low P&L": st.column_config.NumberColumn(format="dollar"),
            "Mid P&L": st.column_config.NumberColumn(format="dollar"),
        }

        def color_outcome(val):
            return "color: green" if val == "Win" else "color: red"

        st.dataframe(
            df.style.map(color_outcome, subset=["Outcome"]),
            hide_index=True,
            column_order=["Trade", "Outcome", "Last", "High P&L", "Low P&L", "Mid P&L"],
            column_config=column_config,
        )
