import subprocess
import os
import sys

def run_experiment(script_name):
    print(f"Running {script_name}...")
    
    # Chạy script và stream output trực tiếp ra terminal
    process = subprocess.Popen(
        ["python", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    # Đọc và hiển thị output real-time
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()
        
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
        if error:
            sys.stderr.write(error)
            sys.stderr.flush()
            
        # Kiểm tra xem process đã kết thúc chưa
        if output == '' and error == '' and process.poll() is not None:
            break
    
    print(f"Finished {script_name}\n")
    
if __name__ == "__main__":
    os.chdir("src")  # Chuyển vào thư mục src trước khi chạy các script
    
    scripts = ["train_cnn.py", "train_gru.py", "train_lstm.py", "train_xgboost.py"]
    
    for script in scripts:
        run_experiment(script)
    
    print("All experiments completed!")
