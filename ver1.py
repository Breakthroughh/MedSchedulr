import cvxpy as cp
import numpy as np
import pandas as pd

# Configuration
num_doctors = 5
num_days = 7
shifts_per_day = 2  # 8am–4pm and 4pm–8am next day
num_shifts = num_days * shifts_per_day

# Simulated availability matrix (1 = available, 0 = unavailable)
np.random.seed(42)
availability = np.random.randint(0, 2, size=(num_doctors, num_shifts))

# Decision variable: x[d, s] = 1 if doctor d is assigned shift s
x = cp.Variable((num_doctors, num_shifts), boolean=True)

constraints = []

# 1. Only assign if available
constraints.append(x <= availability)

# 2. Every shift must be covered by exactly 1 doctor
for s in range(num_shifts):
    constraints.append(cp.sum(x[:, s]) == 1)

# 3. No doctor can work more than 3 consecutive shifts
for d in range(num_doctors):
    for s in range(num_shifts - 2):
        constraints.append(cp.sum(x[d, s:s+3]) <= 2)


# 4. Fairness: minimize deviation from average workload using L1 norm
shift_hours = 8
total_hours = cp.sum(x, axis=1) * shift_hours
avg_hours = cp.sum(total_hours) / num_doctors
deviation = cp.abs(total_hours - avg_hours)
objective = cp.Minimize(cp.sum(deviation))

# Solve the problem using CBC
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC)

# Round and format result
schedule = np.round(x.value).astype(int)
columns = [f"Day {i//2 + 1} - {'8am-4pm' if i % 2 == 0 else '4pm-8am'}" for i in range(num_shifts)]
index = [f"Doctor {i+1}" for i in range(num_doctors)]

schedule_df = pd.DataFrame(schedule, index=index, columns=columns)
availability_df = pd.DataFrame(availability, index=index, columns=columns)

# Output to terminal
print("\nGenerated Shift Schedule:")
print(schedule_df)

print("\nDoctor Availability (1 = available):")
print(availability_df)

# Export to CSV
schedule_df.to_csv("shift_schedule.csv")
availability_df.to_csv("availability.csv")

print("\n✅ CSV files saved:")
print("- shift_schedule.csv")
print("- availability.csv")
