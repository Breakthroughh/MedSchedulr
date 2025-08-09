# Hospital On-Call Scheduler — README (Concise)

## What this does
Generates a monthly on-call roster given:
- period dates
- units & doctors (with category + history)
- clinic days per unit
- posts to cover each day (weekday/weekend)
- availability per doctor/post/day

If the strict model is infeasible, it automatically runs a relaxed pass (adds slacks with huge penalties) to return the **best possible** approximation of a schedule (plus signals where rules were bent maybe?).

---

## Key rules (summary)
- **Coverage**: every listed post is filled each day.
- **Availability**: no assignment if unavailable.
- **No double-booking**: 1 doctor ≤ 1 post/day.
- **Clinic days**: *every* doctor in that unit gets a `clinic` entry on clinic days. Soft penalties for on-call near clinic:
  - day before = 10, same day = 50, day after = 5.
- **Standby (weekends)**:
  - “Standby Oncall” is a **2-day** block (Sat+Sun in output).
  - **≤ 1** standby weekend per doctor per month (hard).
  - Ban if they did standby last month (hard). Longer wait ⇒ bonus.
- **Preferences & fairness**:
  - Seniors/registrars on ED (incl. ED Cover) penalised (λ=5).
  - Registrars on any weekend on-call: light penalty.
  - Juniors on Ward posts: medium penalty.
  - Balance workload vs rolling history (weekday/weekend/ED).
  - **Unit load soft cap**: on non-clinic days, penalise if > ~25% of a unit is on on-call.

> Naming matters: ED posts must start with `ED` (e.g., `ED1`, `ED Cover A1`); wards must start with `Ward`; weekend standby must be exactly `Standby Oncall`.

---

## Inputs you must supply
Replace all random test data with real data from your webapp.

1) **Roster period**
   - `roster_start` (date), `roster_end` (date, inclusive)

2) **Units**
   - `units: string[]` (e.g., `["Unit1","Unit2",...]`)
   - `doctors_per_unit` (per-unit counts recommended)

3) **Doctors** (master table; persisted)
   - `DoctorId`, `DisplayName`, `Unit`, `Category` (`floater|junior|senior|registrar`), `Active`
   - `LastStandby` (date or null)
   - Rolling workload history: `Workload_weekday`, `Workload_weekend`, `Workload_ED`
   - *(Update these after each published month.)*

4) **Clinic days** (by unit; weekdays 0=Mon..6=Sun)
   - `clinic_days: { [unit]: number[] }`

5) **Posts config**
   - `posts_weekday: string[]`
   - `posts_weekend: string[]`
   - `oncall_posts = set(posts_weekday ∪ posts_weekend)`

6) **Availability matrix** (most important operational input)
   - Long format rows for every `(DoctorId, Date, Post)` in the period:
   - `Available ∈ {0,1}`

7) **Solver weights/config** (expose in admin UI)
   - `lambda_rest, lambda_gap, lambda_ed=5, lambda_standby, lambda_min_one, lambda_reg_weekend, lambda_unit_over, lambda_junior_ward`
   - `clinic_penalties = { before:10, same:50, after:5 }`
   - `BIG_M = 10000` (relaxed pass), solver timeout, seed (optional)

---

## Outputs (current)
- `primeVersion2_schedule.csv` — **final schedule** (includes all clinic rows)  
  Columns: `Doctor,Date,Post`
- `primeVersion2_availability.csv` — availability used  
  Columns: `Doctor,Date,Post,Available`
- `primeVersion2_doctor_info.csv` — doctor attributes used

**Recommended additions (for UI/QA):**
- `summary.json` — phase used (`hard|relaxed`), counts of slacks/overages, solver status/time, per-unit/day counts, per-doctor totals.
- `violations.csv` — list of relaxed slacks (coverage, availability, double-book, registrar-only, standby-limit).

---

## Suggested API (so the webapp can call the solver)

```python
def schedule_roster(
    roster_start: date,
    roster_end: date,
    units: list[str],
    clinic_days: dict[str, list[int]],
    posts_weekday: list[str],
    posts_weekend: list[str],
    doctors_df: pd.DataFrame,        # one row per doctor
    availability_df: pd.DataFrame,   # long format, (DoctorId, Date, Post, Available)
    weights: dict,                   # all lambda_* and clinic penalties
    relax_if_infeasible: bool = True,
) -> dict:
    """
    Returns:
      {
        "phase": "hard" | "relaxed",
        "schedule": pd.DataFrame,           # Doctor, Date, Post
        "diagnostics": {
          "violations": pd.DataFrame | None,
          "per_unit_day": pd.DataFrame,
          "per_doctor_counts": pd.DataFrame,
          "solver_status": str,
          "runtime_sec": float
        }
      }
    """```


### Minimal data schemas (examples)

doctors.csv
```DoctorId,DisplayName,Unit,Category,Active,LastStandby,Workload_weekday,Workload_weekend,Workload_ED
U1_D1,Dr A,Unit1,junior,1,2025-05-01,12,6,10
U1_D2,Dr B,Unit1,registrar,1,,8,4,5```

clinic_days.json
```{ "Unit1": [0,4], "Unit2": [0,2], "Unit3": [3] }```

posts_config.json
```{
  "posts_weekday": ["ED1","ED2","ED3","Ward3","Ward4","ED Cover A1","ED Cover A2"],
  "posts_weekend": ["ED1","ED2","ED3","Ward4","Ward5","Ward6","Ward7","Ward9","Ward10","Standby Oncall"]
}```

availability.csv
```DoctorId,Date,Post,Available
U1_D1,2025-08-04,ED1,1
U1_D1,2025-08-04,Ward3,0```

config.json
```{
  "lambda_rest": 3,
  "lambda_gap": 1,
  "lambda_ed": 5,
  "lambda_standby": 5,
  "lambda_min_one": 10,
  "lambda_reg_weekend": 2,
  "lambda_unit_over": 25,
  "lambda_junior_ward": 6,
  "clinic_penalties": { "before": 10, "same": 50, "after": 5 },
  "big_m": 10000,
  "solver": { "name": "CBC", "maximumSeconds": 600 }
}```


### Likely workflow for admins:
1. Pick roster dates.
2. Confirm/edit posts (& weights?).
3. Set clinic weekdays per unit.
4. Maintain doctors master data (unit, category, histories, last standby).
5. Upload/edit availability grid.
6. Run solver → view schedule, make edits + diagnostics.
7. Export/publish schedule.
8. Close month → update workload histories & last standby.