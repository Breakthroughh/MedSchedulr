import cvxpy as cp
import numpy as np
import pandas as pd
import random
import datetime
import math  # for ceiling on 25% cap

# --------------------------------------------------------------------------------
# Utility to compute full months difference between two dates
def months_since(start_date, end_date):
    """
    Return the number of whole months between start_date and end_date.
    """
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

# --------------------------------------------------------------------------------
# Specify the roster period via start and end dates
roster_start = datetime.date(2025, 8, 4)    # e.g., August roster begins 4 Aug 2025
roster_end   = datetime.date(2025, 8, 31)   # roster ends 31 Aug 2025

# Build the list of dates in the roster month
date_list = []
current = roster_start
while current <= roster_end:
    date_list.append(current)
    current += datetime.timedelta(days=1)

# --------------------------------------------------------------------------------
# Clinic days configuration (weekday numbers: 0=Mon,1=Tue,...,6=Sun).
# Manually specify for each unit:
clinic_days = {
    "Unit1": [0, 4],    # Mondays and Fridays
    "Unit2": [0, 2],    # Mondays and Wednesdays
    "Unit3": [3],       # Thursdays
    "Unit4": [1,3],     # Tuesdays and Thursdays
    "Unit5": [0,2,4],   # Mondays, Wednesdays, and Fridays
    "Unit6": [0, 2],    # Mondays and Wednesdays
    "Unit7": [1, 3],    # Tuesdays and Thursdays
    "Unit8": [4],       # Fridays
    "Unit9": [3,4],     # Thursdays and Fridays
    "Unit10": [1,2],    # Tuesdays and Wednesdays
    "Unit11": [4],      # Fridays
    "Unit12": [0,3],    # Mondays and Thursdays
    "Unit13": [0,1,3],  # Mondays, Tuesdays, and Thursdays
}

# Simulation setup: 13 units, each with 7 doctors.
num_units = 14
doctors_per_unit = 6

units = [f"Unit{i+1}" for i in range(num_units)]
categories = ["floater", "junior", "senior", "registrar"]

# Weekday and weekend posts as per spec
posts_weekday = [
    "ED1", "ED2", "ED3",
    "Ward3", "Ward4",
    "ED Cover A1", "ED Cover A2"
]
posts_weekend = [
    "ED1", "ED2", "ED3",
    "Ward4", "Ward5", "Ward6", "Ward7", "Ward9", "Ward10",
    "Standby Oncall"
]

# On-call posts for rest/spacing logic (include wards, ED, and standby)
oncall_posts = set(posts_weekday + posts_weekend)

# --------------------------------------------------------------------------------
# Generate doctors and their metadata
doctors = []
doctor_info = {}
for u in units:
    for i in range(doctors_per_unit):
        name = f"{u}_Doc{i+1}"
        cat = random.choices(categories, weights=[0.1, 0.4, 0.4, 0.1])[0]
        doctors.append(name)
        if cat != "floater":
            months_back = random.randint(0, 3)
            m = roster_start.month - months_back
            y = roster_start.year
            while m <= 0:
                m += 12
                y -= 1
            last_standby_date = datetime.date(y, m, 1)
        else:
            last_standby_date = None
        doctor_info[name] = {
            "unit": u,
            "category": cat,
            "last_standby": last_standby_date,
            "workload": {
                "weekday": random.randint(0, 6),
                "weekend": random.randint(0, 4),
                "ED":      random.randint(0, 5),
            }
        }

# Build posts_by_day mapping based on weekday/weekend of each date
posts_by_day = {}
for idx, date in enumerate(date_list):
    posts_by_day[idx] = posts_weekend if date.weekday() >= 5 else posts_weekday

# --------------------------------------------------------------------------------
# Sets for CVXPY
D = doctors
S = list(range(len(date_list)))

# Precompute unit->doctor list (used in per-unit/day soft cap)
unit_to_docs = {u: [d for d in D if doctor_info[d]['unit'] == u] for u in units}

# === Simulated availability === (65% chance available)
# (Compute once so Phase 1 and Phase 2 see the same availability)
def build_base_availability():
    # We'll create a dict for all (d,s,t) keys that exist in the model
    avail = {}
    for d in D:
        for s in S:
            for t in posts_by_day[s]:
                avail[(d, s, t)] = int(random.random() < 0.65)
    # Floater MOs should never be on-call: force their availability to 0
    for (d, s, t) in list(avail):
        if doctor_info[d]["category"] == "floater" and t in oncall_posts:
            avail[(d, s, t)] = 0
    return avail

availability = build_base_availability()

# --------------------------------------------------------------------------------
# Initialize weights used in objective / penalties (shared by both phases)
# Clinic day vs on-call conflict & penalties (#10):
#   - Soft heavy penalty for day-before (10)
#   - Soft very heavy penalty for same-day (50)
#   - Soft lighter penalty for day-after  (5)
lambda_before_clinic = 10
lambda_same_clinic   = 50
lambda_after_clinic  = 5

# === Objective weights ===
lambda_rest        = 3   # penalty for rest violations
lambda_gap         = 1   # reward for 3-day gaps
lambda_ED          = 6   # penalty for seniors & registrars on ED
lambda_standby     = 5   # reward for long wait since last standby
lambda_min_one     = 10  # penalty if a non-floater gets zero shifts
lambda_reg_weekend = 2   # light penalty for registrars on weekend on-calls
lambda_unit_over   = 25  # penalty for exceeding 25% of a unit on a non-clinic day (soft cap)
lambda_junior_ward = 6  # medium penalty for juniors assigned to ward posts

# === RELAXATION (Phase 2) Big-M weight ===
BIG_M = 10000.0  # very large penalty applied to slack variables when relaxing "hard" rules

# --------------------------------------------------------------------------------
# Helper that builds & solves the model.
# If RELAX=False → your current hard/soft split.
# If RELAX=True  → convert every "hard" rule into a soft rule with nonneg slack and Big-M penalty.
def build_and_solve(RELAX: bool):
    # === Decision variables ===
    x = {(d, s, t): cp.Variable(boolean=True)
         for d in D for s in S for t in posts_by_day[s]}

    # === Soft variables (shared) ===
    rest_violation = {(d, s): cp.Variable(boolean=True)
                      for d in D for s in S if s <= len(date_list) - 3}
    z_gap          = {(d, s): cp.Variable(boolean=True)
                      for d in D for s in S if s <= len(date_list) - 3}
    min_one_slack  = {d: cp.Variable(boolean=True)
                      for d in D if doctor_info[d]["category"] != "floater"}

    # Initialize penalty terms (will collect everything here)
    penalty_terms = []

    # Clinic penalties (same for both phases, these are soft costs not constraints)
    for d in D:
        unit = doctor_info[d]["unit"]
        days_for_unit = clinic_days.get(unit, [])
        for s, date in enumerate(date_list):
            if date.weekday() in days_for_unit:
                for delta in (-1, 0, 1):
                    idx = s + delta
                    if 0 <= idx < len(date_list):
                        for t in posts_by_day[idx]:
                            if t in oncall_posts:
                                if delta == -1:
                                    penalty_terms.append(lambda_before_clinic * x[d, idx, t])
                                elif delta == 0:
                                    penalty_terms.append(lambda_same_clinic   * x[d, idx, t])
                                else:  # +1
                                    penalty_terms.append(lambda_after_clinic  * x[d, idx, t])

    # Soft penalty for registrars doing any on-call on a weekend
    for (d, s, t), var in x.items():
        if (doctor_info[d]["category"] == "registrar"
            and date_list[s].weekday() >= 5
            and t in oncall_posts):
            penalty_terms.append(lambda_reg_weekend * var)

    # NEW: Medium penalty for juniors on ward posts (weekday or weekend).
    # Ward posts are named "Ward..." in both weekday and weekend lists.
    for (d, s, t), var in x.items():
        if doctor_info[d]["category"] == "junior" and t.startswith("Ward"):
            penalty_terms.append(lambda_junior_ward * var)

    # === Constraints ===
    constraints = []

    # --- Soft cap: at most ~25% of a unit assigned to on-call per day (skip that unit's clinic days) ---
    cap_per_unit = math.ceil(0.25 * doctors_per_unit)  # e.g., 7 -> 2
    for u in units:
        u_docs = unit_to_docs[u]
        clinic_weekdays = set(clinic_days.get(u, []))
        for s, date in enumerate(date_list):
            # Skip cap on this unit's clinic weekdays
            if date.weekday() in clinic_weekdays:
                continue
            # Count all on-call assignments for unit u on day s
            assigned_u_s = cp.sum([
                x[d, s, t]
                for d in u_docs
                for t in posts_by_day[s]
                if t in oncall_posts
            ])
            # Soft overage slack and penalty
            over_us = cp.Variable(nonneg=True)
            constraints.append(assigned_u_s <= cap_per_unit + over_us)
            penalty_terms.append(lambda_unit_over * over_us)

    # 1) each post filled exactly once per day
    for s in S:
        for t in posts_by_day[s]:
            if not RELAX:
                constraints.append(cp.sum([x[d, s, t] for d in D]) == 1)
            else:
                cov = cp.sum([x[d, s, t] for d in D])
                s_pos = cp.Variable(nonneg=True)  # overfill
                s_neg = cp.Variable(nonneg=True)  # underfill
                constraints += [cov <= 1 + s_pos, cov >= 1 - s_neg]
                penalty_terms.append(BIG_M * (s_pos + s_neg))

    # 2) respect availability
    for (d, s, t), var in x.items():
        if not RELAX:
            constraints.append(var <= availability[d, s, t])
        else:
            # allow violation: var <= avail + slack
            s_av = cp.Variable(nonneg=True)
            constraints.append(var <= availability[d, s, t] + s_av)
            penalty_terms.append(BIG_M * s_av)

    # 3) no double booking
    for d in D:
        for s in S:
            if not RELAX:
                constraints.append(cp.sum([x[d, s, t] for t in posts_by_day[s]]) <= 1)
            else:
                s_db = cp.Variable(nonneg=True)
                constraints.append(cp.sum([x[d, s, t] for t in posts_by_day[s]]) <= 1 + s_db)
                penalty_terms.append(BIG_M * s_db)

    # 4) registrar-only
    for (d, s, t), var in x.items():
        if "Registrar" in t and doctor_info[d]["category"] != "registrar":
            if not RELAX:
                constraints.append(var == 0)
            else:
                # relax to var <= slack
                s_reg = cp.Variable(nonneg=True)
                constraints.append(var <= s_reg)
                penalty_terms.append(BIG_M * s_reg)

    # 5) Soft: standby priority + restriction
    for (d, s, t), var in x.items():
        if t == "Standby Oncall":
            last_date = doctor_info[d]["last_standby"]
            months_ago = months_since(last_date, roster_start) if last_date else 99
            # Disallow if did last month (still hard/relaxed below)
            if months_ago < 1:
                if not RELAX:
                    constraints.append(var == 0)
                else:
                    s_last = cp.Variable(nonneg=True)
                    constraints.append(var <= s_last)  # only through slack
                    penalty_terms.append(BIG_M * s_last)
            # Reward those who waited longer (soft bonus)
            penalty_terms.append(-lambda_standby * months_ago * var)

    # 5b) at most one Standby Oncall per doctor per month
    for d in D:
        if not RELAX:
            constraints.append(
                cp.sum([x[d, s, "Standby Oncall"]
                        for s in S
                        if "Standby Oncall" in posts_by_day[s]]) <= 1
            )
        else:
            s_once = cp.Variable(nonneg=True)
            constraints.append(
                cp.sum([x[d, s, "Standby Oncall"]
                        for s in S
                        if "Standby Oncall" in posts_by_day[s]]) <= 1 + s_once
            )
            penalty_terms.append(BIG_M * s_once)

    # 6) Soft: rest violation (2-day)
    for d in D:
        for s in range(len(date_list) - 2):
            onc = [x[d, s+i, t]
                   for i in range(3)
                   for t in posts_by_day[s+i]
                   if t in oncall_posts]
            constraints.append(rest_violation[d, s] >= cp.sum(onc) - 1)

    # 7) Soft: reward 3-day rest gaps
    for d in D:
        for s in range(len(date_list) - 2):
            for i in range(3):
                for t in posts_by_day[s+i]:
                    if t in oncall_posts:
                        constraints.append(z_gap[d, s] <= 1 - x[d, s+i, t])

    # 8) Soft: ED penalty for seniors & registrars
    for (d, s, t), var in x.items():
        if t.startswith("ED") and doctor_info[d]["category"] in ["senior", "registrar"]:
            penalty_terms.append(lambda_ED * var)

    # 9) Soft: every non-floater should get ≥1 on-call
    for d, slack in min_one_slack.items():
        assigned = cp.sum([x[d, s, t]
                           for s in S for t in posts_by_day[s]
                           if t in oncall_posts])
        constraints.append(slack >= 1 - assigned)
        penalty_terms.append(lambda_min_one * slack)

    # === Fairness objective: workload deviation ===
    workload_expr = []
    avg_wl = np.mean([
        info["workload"]["weekday"] +
        info["workload"]["weekend"] +
        info["workload"]["ED"]
        for info in doctor_info.values()
        if info["category"] != "floater"
    ])
    for d in D:
        if doctor_info[d]["category"] == "floater":
            continue
        past = sum(doctor_info[d]["workload"].values())
        assigned = cp.sum([x[d, s, t]
                           for s in S for t in posts_by_day[s]
                           if t in oncall_posts])
        workload_expr.append(cp.abs(past + assigned - avg_wl))

    # === Full objective ===
    objective = cp.Minimize(
        sum(workload_expr)
        + lambda_rest   * sum(rest_violation.values())
        - lambda_gap    * sum(z_gap.values())
        + sum(penalty_terms)
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=600)

    return prob, x

# --------------------------------------------------------------------------------
# === Phase 1: try the original (hard) model ===
prob1, x1 = build_and_solve(RELAX=False)
print("Phase 1 status:", prob1.status)

ok_statuses = {"optimal", "optimal_inaccurate", "user_limit"}  # accept if time-limited but feasible
if prob1.status in ok_statuses:
    chosen_x = x1
    print("Using Phase 1 solution.")
else:
    print("Phase 1 not successful (status:", prob1.status, ") → Trying relaxed Phase 2...")
    # === Phase 2: relaxed model with Big-M slack penalties on hard constraints ===
    prob2, x2 = build_and_solve(RELAX=True)
    print("Phase 2 status:", prob2.status)
    chosen_x = x2  # even if time-limited, we keep the best found here

# === Collect on-call results using the chosen solution ===
raw_results = []
for (d, s, t), var in chosen_x.items():
    if var.value is not None and var.value > 0.5:
        raw_results.append((d, date_list[s], t))

# --------------------------------------------------------------------------------
# Build final results: expand 2-day standby and then add all clinic entries
results = []

# 1) Expand standby: Saturday → Saturday + Sunday
for d, date_assigned, post in raw_results:
    if post == "Standby Oncall" and date_assigned.weekday() == 5:
        results.append((d, date_assigned, post))
        results.append((d, date_assigned + datetime.timedelta(days=1), post))
    elif post == "Standby Oncall" and date_assigned.weekday() == 6:
        # skip Sunday, handled by Saturday
        continue
    else:
        results.append((d, date_assigned, post))

# 2) Add clinic: every doctor in unit on each clinic day
for u, weekdays in clinic_days.items():
    unit_docs = [d for d in D if doctor_info[d]["unit"] == u]
    for s, date in enumerate(date_list):
        if date.weekday() in weekdays:
            for d in unit_docs:
                results.append((d, date, "clinic"))

# === Save schedule ===
df_schedule = pd.DataFrame(results, columns=["Doctor", "Date", "Post"])
df_schedule.to_csv("primeVersion2_schedule.csv", index=False)
print("Schedule saved to primeVersion2_schedule.csv")

# === Export availability for checking ===
avail_records = []
for (d, s, t), avail in availability.items():
    avail_records.append({
        "Doctor": d,
        "Date": date_list[s],
        "Post": t,
        "Available": avail
    })
pd.DataFrame(avail_records).to_csv("primeVersion2_availability.csv", index=False)
print("Availability saved to primeVersion2_availability.csv")

# === Export doctor metadata for checking ===
info_records = []
for d, info in doctor_info.items():
    info_records.append({
        "Doctor": d,
        "Unit": info["unit"],
        "Category": info["category"],
        "LastStandby": info["last_standby"],
        "Workload_weekday": info["workload"]["weekday"],
        "Workload_weekend": info["workload"]["weekend"],
        "Workload_ED": info["workload"]["ED"]
    })
pd.DataFrame(info_records).to_csv("primeVersion2_doctor_info.csv", index=False)
print("Doctor info saved to primeVersion2_doctor_info.csv")
