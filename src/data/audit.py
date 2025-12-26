import os
import yaml
import glob
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_audit_report(config_path="configs/data_config.yaml", output_dir="docs/data_audit"):
    config = load_config(config_path)
    local_dir = config['dataset']['local_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "audit_report.md")
    
    print(f"Auditing data in {local_dir}...")
    
    # Count files
    camera_files = glob.glob(os.path.join(local_dir, "camera", "**", "*.zip"), recursive=True)
    label_files = glob.glob(os.path.join(local_dir, "labels", "**", "*.zip"), recursive=True)
    
    total_size = 0
    for f in camera_files + label_files:
        total_size += os.path.getsize(f)
    
    total_size_gb = total_size / (1024**3)
    
    with open(report_path, "w") as f:
        f.write("# Data Audit Report\n\n")
        f.write(f"**Date**: {os.popen('date').read().strip()}\n")
        f.write(f"**Data Directory**: `{local_dir}`\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Total Size**: {total_size_gb:.2f} GB\n")
        f.write(f"- **Camera Files**: {len(camera_files)}\n")
        f.write(f"- **Label Files**: {len(label_files)}\n\n")
        
        f.write("## File Details\n")
        f.write("### Camera Files (Sample)\n")
        f.write("| Filename | Size (MB) |\n")
        f.write("| :--- | :---: |\n")
        for cf in camera_files[:10]:
            f.write(f"| {os.path.basename(cf)} | {os.path.getsize(cf) / 1024**2:.2f} |\n")
        if len(camera_files) > 10:
            f.write(f"| ... and {len(camera_files) - 10} more | - |\n")
            
        f.write("\n### Label Files (Sample)\n")
        f.write("| Filename | Size (MB) |\n")
        f.write("| :--- | :---: |\n")
        for lf in label_files[:10]:
            f.write(f"| {os.path.basename(lf)} | {os.path.getsize(lf) / 1024**2:.2f} |\n")
        if len(label_files) > 10:
            f.write(f"| ... and {len(label_files) - 10} more | - |\n")

    print(f"Audit report generated at {report_path}")
    print(f"Total Size: {total_size_gb:.2f} GB")

if __name__ == "__main__":
    generate_audit_report()
