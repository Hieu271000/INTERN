import pandas as pd

df = pd.read_csv('dataset/PSM/test_label.csv')
df_head = df.head(int(len(df) * 0.1))
df_head.to_csv('dataset/PSM/test_label_10percent.csv', index=False)
