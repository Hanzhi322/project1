# project1.py

DATA_PATH = "Employee_Complete_Dataset.csv"  
NUMERIC_COLUMN = "Employee_age" 
GROUP_COL = None                   
MAX_ROWS = None                     


# Step 5

import sys
import os

def step5_with_pandas(path, value_col, max_rows=None):
    import pandas as pd  

    read_kwargs = {}
    if max_rows is not None:
        read_kwargs["nrows"] = int(max_rows)

    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception as e:
        print(f"[Error] 读取 CSV 失败：{e}")
        sys.exit(1)

    if value_col not in df.columns:
        print(f"[Error] 列名 '{value_col}' 不在 CSV 表头：{list(df.columns)}")
        sys.exit(1)

    s = pd.to_numeric(df[value_col], errors="coerce").dropna()

    # 计算
    mean_pd = s.mean()
    median_pd = s.median()
    modes = s.mode()                  
    mode_pd_value = modes.iloc[0] if len(modes) > 0 else None

    result = {
        "df": df,
        "series": s,
        "mean": float(mean_pd),
        "median": float(median_pd),
        "mode": None if mode_pd_value is None else float(mode_pd_value),
    }
    return result


## Step 6 — The Hard Way

import csv
from math import isfinite

def iter_numeric_from_csv_stdlib(path, numeric_col, max_rows=None):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return values

       
        header = [h.strip() for h in header]
        if numeric_col not in header:
            raise ValueError(f"列名 '{numeric_col}' 不在表头：{header}")

        idx = header.index(numeric_col)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if idx < len(row):
                cell = row[idx].strip()
                try:
                    v = float(cell)
                    if v == v and isfinite(v):
                        values.append(v)
                except:
                    
                    pass
    return values


def mean_py(nums):
    return sum(nums) / len(nums) if nums else None

def median_py(nums):
    if not nums:
        return None
    arr = sorted(nums)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    else:
        return (arr[mid - 1] + arr[mid]) / 2

def mode_py(nums):
    if not nums:
        return None
    counts = {}
    for x in nums:
        counts[x] = counts.get(x, 0) + 1
    
    return max(counts, key=counts.get)


def step6_hard_way(path, value_col, max_rows=None):
    values = iter_numeric_from_csv_stdlib(path, value_col, max_rows)
    mean_v = mean_py(values)
    median_v = median_py(values)
    mode_v = mode_py(values)
    return {
        "values": values,
        "mean": mean_v,
        "median": median_v,
        "mode": mode_v,
    }



# Step 7:Visualization


def ascii_bar_chart(labels, numbers, title=None, max_symbols=40, symbol="█"):
   
    pairs = []
    for l, n in zip(labels, numbers):
        try:
            n = float(n)
            if n == n and isfinite(n):
                pairs.append((str(l), n))
        except:
            pass
    if not pairs:
        print("(no data to plot)")
        return

    vals = [n for _, n in pairs]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1.0

    if title:
        print(title)
        print("-" * len(title))

    for lab, val in pairs:
        k = int(round((val - vmin) / span * max_symbols))
        bar = symbol * k
        print(f"{lab:>14}: {bar}  ({val:.6g})")


def html_bar_chart(labels, numbers, out_path="viz.html", title="Visualization"):
  
    rows = []
    safe = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
   
    nums = []
    for n in numbers:
        try:
            n = float(n)
            if n == n and isfinite(n):
                nums.append(n)
            else:
                nums.append(None)
        except:
            nums.append(None)

    valid_vals = [x for x in nums if x is not None]
    if not valid_vals:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("<p>No data to plot.</p>")
        return out_path

    vmin, vmax = min(valid_vals), max(valid_vals)
    span = (vmax - vmin) or 1.0

    for lab, val in zip(labels, nums):
        if val is None:
            width_pct = 0
            val_txt = "NaN"
        else:
            width_pct = int(round((val - vmin) / span * 100))
            val_txt = f"{val:.6g}"
        rows.append(f"""
        <div class="row">
          <div class="lab">{safe(lab)}</div>
          <div class="bar">
            <div class="fill" style="width:{width_pct}%"></div>
          </div>
          <div class="num">{safe(val_txt)}</div>
        </div>
        """)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{safe(title)}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 20px; margin: 0 0 16px 0; }}
.row {{ display:flex; align-items:center; gap:10px; margin:6px 0; }}
.lab {{ width:160px; text-align:right; color:#333; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar {{ flex:1; background:#f1f3f5; height:16px; border-radius:8px; overflow:hidden; }}
.fill {{ height:100%; background:#4c6ef5; }}  /* 颜色允许，仍属标准库生成的静态HTML */
.num {{ width:100px; text-align:left; color:#555; font-variant-numeric: tabular-nums; }}
.footer {{ margin-top:12px; color:#777; font-size:12px; }}
</style>
</head>
<body>
<h1>{safe(title)}</h1>
{''.join(rows)}
<div class="footer">Generated by standard-library-only HTML writer.</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path



def grouped_series_for_plotting(df, group_col, value_col):
   
    if group_col not in df.columns:
        return None, None
    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    agg = tmp.dropna().groupby(group_col)[value_col].mean().sort_index()
    labels = [str(x) for x in agg.index.tolist()]
    numbers = agg.values.tolist()
    return labels, numbers




def main():
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 找不到数据文件：{DATA_PATH}")
        sys.exit(1)

    print("========== Step 5: pandas ==========")
    p5 = step5_with_pandas(DATA_PATH, NUMERIC_COLUMN, MAX_ROWS)
    print(f"Rows loaded (raw): {len(p5['df'])}")
    print(f"Numeric values used: {len(p5['series'])}")
    print(f"PANDAS — mean   : {p5['mean']}")
    print(f"PANDAS — median : {p5['median']}")
    print(f"PANDAS — mode   : {p5['mode']}")

    print("\n========== Step 6: The hard way (纯标准库) ==========")
    p6 = step6_hard_way(DATA_PATH, NUMERIC_COLUMN, MAX_ROWS)
    print(f"Loaded numeric values (pure Python): {len(p6['values'])}")
    print(f"PURE PY — mean   : {p6['mean']}")
    print(f"PURE PY — median : {p6['median']}")
    print(f"PURE PY — mode   : {p6['mode']}")

   
    print("\n========== Summary ==========")
    summary = f"""
Summary — column = {NUMERIC_COLUMN}
PANDAS:
  mean   = {p5['mean']}
  median = {p5['median']}
  mode   = {p5['mode']}

PURE PYTHON:
  mean   = {p6['mean']}
  median = {p6['median']}
  mode   = {p6['mode']}
""".strip()
    print(summary)

 
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    
    print("\n========== Step 7: Visualization (标准库绘制) ==========")

    
    N = 25
    labels_A = [f"row{i+1}" for i in range(min(N, len(p6['values'])))]
    numbers_A = p6["values"][:N]
    ascii_bar_chart(labels_A, numbers_A, title=f"{NUMERIC_COLUMN} (first {len(numbers_A)} rows)", max_symbols=40, symbol="*")

    
        labels_B, numbers_B = grouped_series_for_plotting(p5["df"], GROUP_COL, NUMERIC_COLUMN)
        if labels_B and numbers_B:
            print()
            ascii_bar_chart(labels_B, numbers_B, title=f"{NUMERIC_COLUMN} by {GROUP_COL} (mean)", max_symbols=40, symbol="█")
            out_html = html_bar_chart(labels_B, numbers_B, out_path="viz.html", title=f"{NUMERIC_COLUMN} by {GROUP_COL} (mean)")
            print(f"\n[Saved] HTML 可视化：{out_html}（用浏览器打开查看）")
        else:
            print(f"(未生成分组图：GROUP_COL={GROUP_COL} 不存在于数据或没有有效数值)")

if __name__ == "__main__":
    main()
