import cvxpy as cp
import numpy as np
import pandas as pd

# Version 2: Multi-Post Scheduling

# Configuration
num_doctors = 5
num_days = 7
shifts_per_day = 2
posts = ["wards", "ER"]
num_posts = len(posts)
num_shifts = num_days * shifts_per_day

# Improved: 80% chance of being available
np.random.seed(42)
availability = (np.random.rand(num_doctors, num_shifts) < 0.8).astype(int)

# Decision variable: x[d, s, p] = 1 if doctor d is assigned to post p in shift s
x = cp.Variable((num_doctors, num_shifts, num_posts), boolean=True)

constraints = []

# Constraint 1: Each shift-post must be filled by one doctor
for s in range(num_shifts):
    for p in range(num_posts):
        constraints.append(cp.sum(x[:, s, p]) == 1)

# Constraint 2: Doctor must be available
for d in range(num_doctors):
    for s in range(num_shifts):
        for p in range(num_posts):
            constraints.append(x[d, s, p] <= availability[d, s])

# Constraint 3: No double-posting in the same shift
for d in range(num_doctors):
    for s in range(num_shifts):
        constraints.append(cp.sum(x[d, s, :]) <= 1)

# === Optional constraint: max 2 consecutive shifts (commented out for now) ===
for d in range(num_doctors):
    for s in range(num_shifts - 2):
        constraints.append(cp.sum(x[d, s:s+3, :]) <= 2)

# Objective: minimize deviation from average workload
shift_hours = 8
total_hours = cp.sum(x, axis=(1, 2)) * shift_hours
avg_hours = cp.sum(total_hours) / num_doctors
deviation = cp.abs(total_hours - avg_hours)
objective = cp.Minimize(cp.sum(deviation))

# Solve with verbosity
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.CBC, verbose=True, maximumSeconds=30)

# Check solution
print("Solver status:", problem.status)
print("Solver result value:", result)

if x.value is None:
    print("❌ No feasible solution found or solver failed.")
    exit()

# Format output
schedule = np.round(x.value).astype(int)
columns = [f"Day {i//2 + 1} - {'8am-4pm' if i % 2 == 0 else '4pm-8am'}" for i in range(num_shifts)]
index = [f"Doctor {i+1}" for i in range(num_doctors)]
multi_columns = pd.MultiIndex.from_product([columns, posts], names=["Shift", "Post"])
schedule_2d = schedule.reshape((num_doctors, num_shifts * num_posts))
schedule_df = pd.DataFrame(schedule_2d, index=index, columns=multi_columns)

# Export availability and schedule
availability_df = pd.DataFrame(availability, index=index, columns=columns)
schedule_df.to_csv("shift-schedule-ver2.csv")
availability_df.to_csv("availability-ver2.csv")

print("✅ Schedule and availability exported.")
