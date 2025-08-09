import cvxpy as cp
import numpy as np
import pandas as pd
from itertools import product
import random
import datetime



# We want to create a simulation to test if this rostering solver works.
# Simulation setup: 13 units, each with 7 doctors. 28 day schedule.  
# Weekday have 3 ED posts, 2 wards. Weekends have 3 ED, 6 wards, 1 registrar. 
num_units = 13
doctors_per_unit = 7
days = 28
units = [f"Unit{i+1}" for i in range(num_units)]
CATEGORIES = ["floater", "junior", "senior", "registrar"]
posts_weekday = ["ED1", "ED2", "ED3", "Ward3", "Ward4", "ED Cover A1", "ED Cover A2"]  #Scheduler doesn't do stroke oncall/reigstrar weekend on-call schedule
posts_weekend = ["ED1", "ED2", "ED3", "Ward4", "Ward5", "Ward6", "Ward7", "Ward9", "Ward10", "Standby Oncall"]
oncall_posts = set(p for p in posts_weekday + posts_weekend if "ED" in p or "Ward" in p)


# print(oncall_posts)
# {'ED3', 'Ward9', 'ED1', 'Ward4', 'Ward8Registar', 'Ward10', 'Ward6', 'Ward3', 'ED2', 'Ward7', 'Ward5'}



# Suppose your roster month starts on:
roster_start = datetime.date(2025, 8, 1)   # e.g. August 2025


# Generate doctors. 
# Each doctor has a unit, a "category" for seniority, how many months ago they did their last 
doctors = []
doctor_info = {}
for u in units:
    for i in range(doctors_per_unit):
        name = f"{u}_Doc{i+1}"
        cat = random.choices(CATEGORIES, weights=[0.1, 0.4, 0.4, 0.1])[0]
        doctors.append(name)


        # In practice: this is stored, but here we are simulating.  
        # pick a random month within the last 3 months:
        months_back = random.randint(0, 3)
        # compute year/month of that standby:
        m = roster_start.month - months_back
        y = roster_start.year
        # wrap around if m <= 0
        while m <= 0:
            m += 12
            y -= 1
        last_standby_date = datetime.date(y, m, 1)


        doctor_info[name] = {
            "unit": u,
            "category": cat,
            "last_standby": last_standby_date,
            "workload": {
                "weekday": random.randint(0, 6),
                "weekend": random.randint(0, 4),
                "ED":      random.randint(0, 5),
            }
            # TODO Should be datetime.date(YYYY, MM, 1)
            # If you store they last did standby X months ago, then need to update every month. 
        }


# Posts per day
posts_by_day = {}
for d in range(days):
    if d % 7 in [5, 6]:  # weekend
        posts_by_day[d] = posts_weekend
    else:
        posts_by_day[d] = posts_weekday

# Sets
D = doctors
S = list(range(days))
T = list(set(p for v in posts_by_day.values() for p in v))

# Decision variables
x = {(d, s, t): cp.Variable(boolean=True) for d in D for s in S for t in posts_by_day[s]}

# Availability (simulated)
availability = {(d, s, t): int(random.random() < 0.9) for d, s, t in x}

# Objective components
penalty_ed = 2
lambda_rest = 3
lambda_gap = 1
lambda_ed = 2

# Soft variables
rest_violation = {(d, s): cp.Variable(boolean=True) for d in D for s in S if s <= days - 3}
z_gap = {(d, s): cp.Variable(boolean=True) for d in D for s in S if s <= days - 3}

# Constraints
constraints = []

# Shift coverage
for s in S:
    for t in posts_by_day[s]:
        constraints.append(cp.sum([x[d, s, t] for d in D if (d, s, t) in x]) == 1)

# Availability
for (d, s, t), var in x.items():
    constraints.append(var <= availability[d, s, t])

# No double booking
for d in D:
    for s in S:
        constraints.append(cp.sum([x[d, s, t] for t in posts_by_day[s] if (d, s, t) in x]) <= 1)

# Soft: rest violation
for d in D:
    for s in range(days - 2):
        if all((d, s + i, t) in x for i in range(3) for t in posts_by_day[s + i]):
            oncalls = [x[d, s + i, t] for i in range(3) for t in posts_by_day[s + i] if t in oncall_posts and (d, s + i, t) in x]
            constraints.append(rest_violation[d, s] >= cp.sum(oncalls) - 1)

# Soft: reward gaps
for d in D:
    for s in range(days - 2):
        for i in range(3):
            for t in posts_by_day[s + i]:
                if t in oncall_posts and (d, s + i, t) in x:
                    constraints.append(z_gap[d, s] <= 1 - x[d, s + i, t])

# ED preference penalty
penalty_expr = []
for d, s, t in x:
    if t.startswith("ED") and doctor_info[d]["category"] == "senior":
        penalty_expr.append(lambda_ed * x[d, s, t])
    else:
        penalty_expr.append(0)

# Fairness objective approximation
workload_expr = []
avg_workload = np.mean([
    doctor_info[d]["workload"]["weekday"] + doctor_info[d]["workload"]["weekend"] + doctor_info[d]["workload"]["ED"]
    for d in D if doctor_info[d]["category"] != "floater"
])
for d in D:
    if doctor_info[d]["category"] == "floater":
        continue
    assigned = cp.sum([x[d, s, t] for s in S for t in posts_by_day[s] if (d, s, t) in x and t in oncall_posts])
    target = doctor_info[d]["workload"]["weekday"] + doctor_info[d]["workload"]["weekend"] + doctor_info[d]["workload"]["ED"]
    workload_expr.append(cp.abs(target + assigned - avg_workload))

# Final objective
objective = cp.Minimize(
    cp.sum(workload_expr) +
    lambda_rest * cp.sum(rest_violation.values()) -
    lambda_gap * cp.sum(z_gap.values()) +
    cp.sum(penalty_expr)
)

# Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CBC)

# Output results
results = []
for (d, s, t), var in x.items():
    if var.value is not None and var.value > 0.9:
        results.append((d, s, t))
df = pd.DataFrame(results, columns=["Doctor", "Day", "Post"])
tools.display_dataframe_to_user("Scheduled Shifts", df)
