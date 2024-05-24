import os
import re
import csv

def extract_metrics(filename):
    # Example: gmf_factor8neg4-implict_Epoch0_HR0.4478_NDCG0.2501.model
    pattern = r"_Epoch(\d+)_HR([\d.]+)_NDCG([\d.]+)\.model"
    match = re.search(pattern, filename)
    if match:
        epoch = int(match.group(1))
        hr = float(match.group(2))
        ndcg = float(match.group(3))
        return epoch, hr, ndcg
    return None

def write_csv(prefix, files, output_file):
    metrics_list = []
    for file in files:
        metrics = extract_metrics(file)
        if metrics:
            metrics_list.append(metrics)
    
    # Sort the metrics by epoch
    metrics_list.sort(key=lambda x: x[0])

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'HR', 'NDCG'])  # write the header
        for metrics in metrics_list:
            csvwriter.writerow(metrics)

def main():
    folder_path = 'checkpoints'
    files = os.listdir(folder_path)

    # Define the prefixes and output filenames
    tasks = [
        ("gmf", "result/gmf.csv"),
        ("mlp0", "result/mlp0.csv"),
        ("mlp1", "result/mlp1.csv"),
        ("mlp2", "result/mlp2.csv"),
        ("mlp3", "result/mlp3.csv"),
        ("mlp4", "result/mlp4.csv"),
        ("neumf3", "result/neumf.csv")
    ]

    for prefix, output_file in tasks:
        # Filter files that start with the given prefix
        selected_files = [f for f in files if f.startswith(prefix)]
        # Write the metrics to the corresponding CSV file
        write_csv(prefix, selected_files, output_file)

if __name__ == "__main__":
    main()
