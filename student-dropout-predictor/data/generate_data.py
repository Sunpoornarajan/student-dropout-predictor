import pandas as pd
import random

rows = []

for _ in range(1500):
    attendance = random.randint(10, 100)
    internal_marks = random.randint(10, 100)
    previous_gpa = round(random.uniform(1.0, 10.0), 1)
    assignments = random.randint(0, 10)
    behavior_score = random.randint(0, 10)

    # Dropout logic (realistic rules)
    if attendance < 50 or previous_gpa < 5 or assignments < 3:
        dropout = 1
    else:
        dropout = 0

    rows.append([
        attendance,
        internal_marks,
        previous_gpa,
        assignments,
        behavior_score,
        dropout
    ])

df = pd.DataFrame(rows, columns=[
    "attendance",
    "internal_marks",
    "previous_gpa",
    "assignments",
    "behavior_score",
    "dropout"
])

df.to_csv("data/student_data.csv", index=False)

print("✅ 1500 student records generated successfully!")
