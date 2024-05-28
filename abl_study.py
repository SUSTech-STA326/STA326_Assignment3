from experiment import experiment
import pandas as pd


factor_nums = [8]
layer_nums = [0, 1, 2, 3, 4]

df_table3 = pd.DataFrame(columns=["factor_num", "layer_num", "HR@10"])
for factor_num in factor_nums:
    for layer_num in layer_nums:
        layers = [factor_num * (2**i) for i in range(layer_num+1)][::-1]
        HR, _ = experiment(method="MLP", embed_size=factor_num * (2**layer_num) // 2, layers=layers, epochs=10)
        new_row = pd.DataFrame({"factor_num": [factor_num], "layer_num": [layer_num], "HR@10": [HR]})
        df_table3 = pd.concat([df_table3, new_row], ignore_index=True)
        

print("layer_num=: ", layer_num)
df_table3.to_csv("table3.csv", index=False)
