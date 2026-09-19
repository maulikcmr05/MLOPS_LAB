import pandas as pd
data = pd.read_csv("Salary_Data.csv")
data["Years of Experience"] += 1
data.to_csv("Salary_Data.csv", index=False)
print("Dataset Version 2 created")