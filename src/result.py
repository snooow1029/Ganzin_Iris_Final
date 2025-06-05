import subprocess
from pathlib import Path

# 多個 input list 路徑
input_lists = [
    #Path(r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\input_list\list_CASIA-Iris-Lamp.txt"),
    #Path(r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\input_list\list_CASIA-Iris-Thousand.txt"),
    #Path(r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\input_list\list_Ganzin-J7EF-Gaze.txt"),
    Path(r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\input_list\list_Ganzin-J7EF-Gaze_bonus.txt")
]

# output 存放資料夾
output_list_dir = Path(r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\output_list")
from datetime import datetime

# 產生時間戳記，格式為 YYYYMMDD_HHMM
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
result_tag = f"googlenet_gaze_{timestamp}"  # 加上時間戳記的 tag

for input_path in input_lists:
    # 在原始檔名後加上 _result_{tag}.txt
    output_name = f"{input_path.stem[5:]}_{result_tag}.txt"
    output_path = output_list_dir / output_name

    print(f"\n==> 正在處理：{input_path.name}")
    print(f"    準備建立 output 檔案：{output_path.name}")

    try:
        # 建立空的 output 檔案（若已存在則覆蓋為空）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")  # 建立為空檔

        # 執行 run.py
        subprocess.run([
            "python", "src/run.py",
            "--input", str(input_path),
            "--output", str(output_path)
        ], check=True)

        # 檢查是否 run.py 寫入成功
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ output 已成功建立並有內容，執行 eval.py")
            subprocess.run([
                "python", "src/eval.py",
                "--input", str(output_path)
            ], check=True)
        else:
            print(f"⚠️ output 檔存在但內容為空，跳過 eval")

    except subprocess.CalledProcessError as e:
        print(f"❌ 發生錯誤：{e}. 跳過此資料集。")

print("\n🎉 所有 input 處理完畢！")
