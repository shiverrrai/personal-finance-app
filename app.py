import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

# ======================================================
# Simulation Core
# ======================================================

def simulate_one_path(
    years,
    retirement_year,
    income_today,
    income_growth,
    expenses_today,
    inflation,
    liquid_assets,
    real_estate_props,
    expense_events,
):
    """
    Simulates one stochastic life path.
    Returns (success, history)
    """

    # --- State ---
    liquid = {k: v["value"] for k, v in liquid_assets.items()}

    properties = []
    for prop in real_estate_props:
        properties.append({
            "value": prop["value"],
            "mu": prop["mu"],
            "sigma": prop["sigma"],
            "mortgage_balance": prop["mortgage_balance"],
            "annual_payment": prop["annual_payment"],
            "years_remaining": prop["years_remaining"]
        })

    history = {
        "liquid": [],
        "home_equity": [],
        "total": []
    }

    for t in range(years):

        # --------------------
        # Income
        # --------------------
        if t < retirement_year:
            income = income_today * ((1 + income_growth) ** t)
        else:
            income = 0.0

        # --------------------
        # Base living expenses
        # --------------------
        base_expenses = expenses_today * ((1 + inflation) ** t)

        # --------------------
        # Mortgage expenses
        # --------------------
        mortgage_expense = 0.0
        for prop in properties:
            if prop["years_remaining"] > t and prop["mortgage_balance"] > 0:
                payment = min(prop["annual_payment"], prop["mortgage_balance"])
                mortgage_expense += payment
                prop["mortgage_balance"] -= payment

        # --------------------
        # Expense events (kids, tuition, etc.)
        # --------------------
        event_expenses = 0.0
        for event in expense_events:
            if event["start_year"] <= t <= event["end_year"]:
                event_expenses += event["annual_cost_today"] * ((1 + inflation) ** t)

        # --------------------
        # Total expenses & cashflow
        # --------------------
        total_expenses = base_expenses + mortgage_expense + event_expenses
        net_cashflow = income - total_expenses

        # --------------------
        # Apply cashflow to liquid assets only
        # --------------------
        liquid_total = sum(liquid.values())
        if liquid_total <= 0:
            return False, history

        for k in liquid:
            liquid[k] += net_cashflow * (liquid[k] / liquid_total)

        # --------------------
        # Apply returns
        # --------------------
        for k, params in liquid_assets.items():
            r = np.random.normal(params["mu"], params["sigma"])
            liquid[k] *= (1 + r)

        for prop in properties:
            r_home = np.random.normal(prop["mu"], prop["sigma"])
            prop["value"] *= (1 + r_home)

        # --------------------
        # Net worth accounting
        # --------------------
        liquid_nw = sum(liquid.values())
        home_equity = sum(
            max(0, p["value"] - p["mortgage_balance"]) for p in properties
        )
        total_nw = liquid_nw + home_equity

        history["liquid"].append(liquid_nw)
        history["home_equity"].append(home_equity)
        history["total"].append(total_nw)

        if total_nw <= 0:
            return False, history

    return True, history


def monte_carlo_probability(retirement_year, n_sims, **kwargs):
    successes = 0
    histories = []

    for _ in range(n_sims):
        ok, hist = simulate_one_path(retirement_year=retirement_year, **kwargs)
        if ok:
            successes += 1
        histories.append(hist)

    return successes / n_sims, histories


def find_earliest_retirement(candidate_years, target_prob, n_sims, **kwargs):
    for r in candidate_years:
        prob, _ = monte_carlo_probability(
            retirement_year=r,
            n_sims=n_sims,
            **kwargs
        )
        if prob >= target_prob:
            return r, prob
    return None, None


# ======================================================
# Streamlit UI
# ======================================================

st.title("Financial Forcasting Simulator")
st.markdown("The following simulator runs a monte carlo analysis to "
            "determine the earliest retirement age given a set of inputs. "
            "The following simulation assumes mortgage payments and property "
            "taxes are lumped under the \'Basic Living Expenses\' section.")

# --------------------
# Demographics
# --------------------
st.sidebar.header("Demographics")
current_age = st.sidebar.number_input("Current Age", 25, 60, 35)
retire_min = st.sidebar.number_input("Minimum Retirement Age", 40, 70, 50)
max_age = st.sidebar.number_input("Max Planning Age", 80, 110, 95)

years = max_age - current_age
candidate_years = list(range(retire_min - current_age, years))

# --------------------
# Income & Expenses
# --------------------
st.sidebar.header("Income & Expenses")
income_today = st.sidebar.number_input("After-Tax Household Income ($)", 0, 500_000, 180_000)
income_growth = st.sidebar.slider("Real Income Growth (%)", 0.0, 3.0, 1.0) / 100
expenses_today = st.sidebar.number_input("Base Living Expenses ($)", 0, 300_000, 70_000)
inflation = st.sidebar.slider("Inflation (%)", 1.0, 4.0, 2.0) / 100

# --------------------
# Liquid Assets
# --------------------
st.sidebar.header("Liquid Assets")
equities = st.sidebar.number_input("Equities ($)", 0, 5_000_000, 400_000)
mu_equities = st.sidebar.slider("Equity Projected Return (%)", 0.0, 15.0, 6.0) / 100
savings = st.sidebar.number_input("Savings ($)", 0, 5_000_000, 150_000)
mu_savings = st.sidebar.slider("Savings Projected Return (%)", 0.0, 5.0, 2.0) / 100

liquid_assets = {
    "Equities": {"value": equities, "mu": mu_equities, "sigma": 0.18},
    "Savings": {"value": savings, "mu": mu_savings, "sigma": 0.06},
}

# --------------------
# Real Estate (Multiple)
# --------------------
st.sidebar.header("Real Estate (Illiquid)")
num_props = st.sidebar.number_input("Number of Properties", 0, 5, 1)

real_estate_props = []

for i in range(num_props):
    st.sidebar.subheader(f"Property {i+1}")
    value = st.sidebar.number_input(f"Market Value #{i+1}", 0, 10_000_000, 800_000, key=f"val{i}")
    mortgage = st.sidebar.number_input(f"Mortgage Balance #{i+1}", 0, 10_000_000, 500_000, key=f"mort{i}")
    payment = st.sidebar.number_input(f"Annual Mortgage Payment #{i+1}", 0, 300_000, 36_000, key=f"pay{i}")
    years_left = st.sidebar.number_input(f"Mortgage Years Remaining #{i+1}", 0, 40, 20, key=f"yrs{i}")

    real_estate_props.append({
        "value": value,
        "mu": 0.03,
        "sigma": 0.10,
        "mortgage_balance": mortgage,
        "annual_payment": payment,
        "years_remaining": years_left
    })

# --------------------
# Expense Events
# --------------------
st.sidebar.header("Expense Events (Kids, Tuition, Care, etc.)")
num_events = st.sidebar.number_input("Number of Expense Events", 0, 5, 1)

expense_events = []

for i in range(num_events):
    st.sidebar.subheader(f"Event {i+1}")
    start = st.sidebar.number_input(f"Start Year #{i+1}", 0, 60, 2, key=f"es{i}")
    end = st.sidebar.number_input(f"End Year #{i+1}", start, 60, start + 18, key=f"ee{i}")
    cost = st.sidebar.number_input(
        f"Annual Cost (Today $) #{i+1}",
        0,
        200_000,
        20_000,
        key=f"ec{i}"
    )

    expense_events.append({
        "start_year": start,
        "end_year": end,
        "annual_cost_today": cost
    })

# --------------------
# Simulation Settings
# --------------------
st.sidebar.header("Simulation Settings")
n_sims = st.sidebar.slider("Monte Carlo Runs", 1000, 20000, 5000, step=1000)
target_prob = st.sidebar.slider("Target Success Probability", 0.7, 0.99, 0.9)

# ======================================================
# Run
# ======================================================

if st.button("Run Simulation"):
    with st.spinner("Running Monte Carlo simulations..."):
        r_year, prob = find_earliest_retirement(
            candidate_years=candidate_years,
            target_prob=target_prob,
            n_sims=n_sims,
            years=years,
            income_today=income_today,
            income_growth=income_growth,
            expenses_today=expenses_today,
            inflation=inflation,
            liquid_assets=liquid_assets,
            real_estate_props=real_estate_props,
            expense_events=expense_events
        )

    if r_year is None:
        st.error("No feasible retirement year under current assumptions.")
    else:
        retire_age = current_age + r_year
        st.success(f"Earliest Retirement Age: {retire_age} (≥ {target_prob:.0%} success)")

        _, histories = monte_carlo_probability(
            retirement_year=r_year,
            n_sims=300,
            years=years,
            income_today=income_today,
            income_growth=income_growth,
            expenses_today=expenses_today,
            inflation=inflation,
            liquid_assets=liquid_assets,
            real_estate_props=real_estate_props,
            expense_events=expense_events
        )

        fig, ax = plt.subplots()
        for h in histories:
            ax.plot(h["liquid"], alpha=0.15, color="blue")
        ax.set_title("Liquid Net Worth Trajectories")
        ax.set_xlabel("Years From Now")
        ax.set_ylabel("Liquid Net Worth ($)")
        st.pyplot(fig)

    # Consolidate results for saving
    results_df = pd.DataFrame({
        "start_year": current_age,
        "end_year": max_age,
        "income": income_today,
        "expenses": expenses_today,
        "inflation": inflation,
        "target_prob": target_prob,
        "retirement_year": current_age + r_year
    }, index=[0])

    for key in liquid_assets.keys():
        results_df[f"{key}_value"] = liquid_assets[key]["value"]
        results_df[f"{key}_mu"] = liquid_assets[key]["mu"]
        results_df[f"{key}_sigma"] = liquid_assets[key]["sigma"]

    for i, prop in enumerate(real_estate_props):
        results_df[f"property_{i}_value"] = prop["value"]
        results_df[f"property_{i}_appreciation"] = prop["mu"]
        results_df[f"property_{i}_variance"] = prop["sigma"]
        results_df[f"property_{i}_mortgage_balance"] = prop["mortgage_balance"]
        results_df[
            f"property_{i}_mortgage_payment"] = prop["annual_payment"]
        results_df[f"property_{i}_years_remaining"] = prop["years_remaining"]

    for i, event in enumerate(expense_events):
        results_df[f"event_{i}_start_year"] = event["start_year"]
        results_df[f"event_{i}_end_year"] = event["end_year"]
        results_df[f"event_{i}_annual_cost"] = event["annual_cost_today"]

    csv = results_df.to_csv()

    st.download_button(
        label="Download simulation results (CSV)",
        data=csv,
        file_name="retirement_monte_carlo_results.csv",
        mime="text/csv"
    )