import re
import csv
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Parse outlier-removal stats from clean-job logs.")
    parser.add_argument(
        "--log-file",
        default=os.path.join(PROJECT_ROOT, "logs", "clean_log.log"),
        help="Path to clean job log file",
    )
    parser.add_argument(
        "--csv-out",
        default=os.path.join(PROJECT_ROOT, "data", "cleaned_volumes", "outlier_stats.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


args = parse_args()
log_file = args.log_file
csv_out = args.csv_out

filename_pattern = re.compile(r'Processing (MOM_.*?)_pred\.npy ->')
erasing_pattern = re.compile(r'Erasing (\d+) totally disconnected floating outlier voxels \(([\d\.]+)% of predicted mass\)\.')

data = []

if not os.path.exists(log_file):
    print("Log file not found!")
    exit(1)

with open(log_file, 'r') as f:
    current_file = None
    for line in f:
        m1 = filename_pattern.search(line)
        if m1:
            current_file = m1.group(1)
            continue
            
        m2 = erasing_pattern.search(line)
        if m2 and current_file:
            voxels = int(m2.group(1))
            percentage = float(m2.group(2))
            data.append({"volume": current_file, "voxels_removed": voxels, "percentage_removed": percentage})
            current_file = None

if not data:
    print("No data found!")
    exit(1)

percentages = [d["percentage_removed"] for d in data]

def calc_mean(lst):
    return sum(lst) / len(lst)

def calc_var(lst):
    u = calc_mean(lst)
    return sum([(x - u)**2 for x in lst]) / len(lst)

normal_percentages = [p for p in percentages if p < 50.0]

print("=== OVERALL STATISTICS (Including Failed Volumes) ===")
print(f"Total Volumes Parsed: {len(percentages)}")
print(f"Mean Removal Ratio: {calc_mean(percentages):.4f}%")
print(f"Variance: {calc_var(percentages):.4f}")
print(f"Min Removal Ratio: {min(percentages):.4f}%")
print(f"Max Removal Ratio: {max(percentages):.4f}%\n")

print("=== NORMAL STATISTICS (Excluding >50% Outliers) ===")
print(f"Total Normal Volumes: {len(normal_percentages)}")
print(f"Mean Removal Ratio: {calc_mean(normal_percentages):.4f}%")
print(f"Variance: {calc_var(normal_percentages):.4f}")
print(f"Min Removal Ratio: {min(normal_percentages):.4f}%")
print(f"Max Removal Ratio: {max(normal_percentages):.4f}%")

with open(csv_out, 'w', newline='') as csvfile:
    fieldnames = ['volume', 'voxels_removed', 'percentage_removed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)
        
print(f"\nCSV successfully written to: {csv_out}")
