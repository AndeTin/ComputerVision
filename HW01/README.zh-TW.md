# HW01 — 少樣本輸入管線與自訂擴增

> [English](README.md)

產出 **M0**（僅真實影像）與 **M1**（M0 加上自訂擴增）。
本作業全程未使用任何生成模型。

## 執行方式

所有指令都在 **repository 根目錄**（`ComputerVision/`，即本 README 的上一層）執行。
不需要任何絕對路徑 —— 程式碼中的每個路徑都由檔案自身位置推導而得，因此
repository 複製或移動到任何位置皆可運作。

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python HW01/run_experiments.py                # Q1 + Q2 + Q3，含所有圖表
python HW01/run_experiments.py --stage q1     # 只跑單一階段
python HW01/run_experiments.py --no-plots     # 略過圖表
python HW01/run_experiments.py --no-samples   # 略過耗時最久的蒙太奇圖
```

**Windows（PowerShell）**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python HW01\run_experiments.py
python HW01\run_experiments.py --stage q1
python HW01\run_experiments.py --no-plots
python HW01\run_experiments.py --no-samples
```

Windows 注意事項：

- 若 PowerShell 不允許啟用 venv，可允許目前 session 執行本機腳本：
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- `.\.venv\Scripts\Activate.ps1` 是 PowerShell 的路徑；
  `.venv\Scripts\activate.bat` 則是 `cmd.exe` 用的。
- 資料集會依 repository 位置自動找到，無需額外設定。若你將影像放在其他位置，
  可明確指定：`$env:HW01_DATASET_DIR = "D:\data\dataset"`
- 全程未使用任何僅限 POSIX 的工具，因此兩個平台的結果一致。

在 CPU 上從頭執行約需 23 分鐘。cache 命中時約 1 分鐘：backbone 產生的
512 維特徵會依「所有可能影響它的參數」作為 key，快取於 `HW01/.cache/`。
刪除該資料夾即可強制重新計算。

## 環境

相依版本固定於 `requirements.txt`（直接相依，位於 repository 根目錄）與
`requirements-lock.txt`（完整 venv）。僅安裝 CPU 版本即足夠，可省下約
4.5 GB 的 CUDA wheel：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

同一道指令在 PowerShell 中亦適用。ResNet-18 的 ImageNet 權重會在首次執行時
下載（約 45 MB），並快取於 `~/.cache/torch`（Windows 為 `%USERPROFILE%\.cache\torch`）。

### 設定

| 變數 | 預設值 | 用途 |
|------|--------|------|
| `HW01_DATASET_DIR` | `<repo root>/dataset` | 各類別資料夾所在位置 |

## 目錄結構

```
<repo root>/
  requirements.txt
  requirements-lock.txt
  dataset/               課程資料，每個類別一個資料夾（cl01..cl20）
  HW01/
    hw01/                函式庫 —— 不含任何階段邏輯
      data.py            資料切分、EXIF 安全載入、幾何策略、路徑、資料集統計
      models.py          凍結的 ResNet-18 特徵擷取器 + scikit-learn LogisticRegression 探針
      metrics.py         單次執行指標、跨 seed 的 mean ± sd、混淆矩陣、區間重疊判定
      augmentation.py    擴增配方、針對性複製、記錄擴張
      experiment.py      單一路徑：config → 特徵 → 探針 → 指標
      reporting.py       組裝 summary.json 與可直接貼入報告的 markdown 表格
    stages/              每個作業問題各一個模組
      q1_input_pipeline.py         切分規則、seed 固定順序、EXIF 稽核、正規化
      q2_resolution_comparison.py  3 種策略 × 2 種插值 × 3 組 seed，幾何稽核
      q3_augmentation_design.py    M0 基準線、弱類別挑選、完整性檢查、M1 掃描
    run_experiments.py   進入點：階段分派、summary 產生、所有圖表
    results/
      summary.json       面向報告：重點數字、各項排名、markdown 表格
      full/              詳細封存：per-seed 資料，可重繪每一張圖
    plots/               7 張圖表
    .cache/              快取的特徵（已 gitignore）
```

函式庫本身完全不知道目前正在跑哪個階段，各階段也不包含任何繪圖程式碼。
只有 `run_experiments.py` 同時知道這兩件事。

每個路徑都由 `__file__` 計算而得，因此本專案與位置無關，在 Windows、
macOS 與 Linux 上的行為一致。

## 結果檔案配置

兩種檔案，對應兩種用途：

| 路徑 | 內容 | 用途 |
|------|------|------|
| `HW01/results/summary.json` | M0/M1 重點表格、Q2 排名、Q3 比例掃描、各類別 recall、可直接貼上的 markdown | 在報告中引用數字 |
| `HW01/results/full/*.json` | 所有欄位，含 per-seed 20×20 混淆矩陣 | 不需重新訓練即可重繪任何圖表 |

`full/` 刻意保持詳細 —— 它是復現用封存，而不是給人閱讀的。cache 命中時完整執行
約 3.5 分鐘，cache 未命中約 23 分鐘。

## 復現性契約

Seed 固定為 `SEEDS = (42, 420, 4200)`，並依以下已記錄的順序套用：

1. `set_global_seed(seed)` —— `random`、`numpy`、`torch`、`torch`（若有 CUDA）
2. 各類別切分排列：`random.Random(seed*1000 + class_index)`
3. DataLoader 洗牌：另外以 `torch.Generator` 設定 seed
4. 擴增亂數來源：`random.Random((seed*1000003 + index)*97 + replicate)`

每個報告出來的數字，都是上述三種取樣設定下的平均值 ± 標準差。比較時一律以
「區間是否重疊」來描述；全文不使用「顯著」一詞，因為三組 seed 無法支持該主張。

`PYTHONHASHSEED` **並未**在程式內部設定 —— CPython 只在 interpreter 啟動時讀取
該變數，因此在執行中設定等於無效。雜湊順序的決定性改以結構方式達成：所有
目錄與檔案清單在使用前都明確 `sorted()`，因此沒有任何結果依賴 set 或 dict 的
迭代順序。已實際以 `PYTHONHASHSEED=0`、`1`、`12345` 各跑一次並確認輸出位元組
完全相同。若需為外部工具固定該變數，請在啟動前匯出。

## 模型

`torchvision.models.resnet18(weights=IMAGENET1K_V1)`，所有參數皆設為
`requires_grad=False`，並將 `fc` 換成 `nn.Identity` 以取出 512 維的 pooled 特徵。
分類器為 `sklearn.linear_model.LogisticRegression`（`solver='lbfgs'`，
`C=1.0` 在整份研究中固定不變）。舊版的 `multi_class` 參數已於
scikit-learn 1.5 移除，本專案並未使用。

## 已知限制

- 資料集資料夾僅含訓練影像，因此每類保留 3 張作為驗證集，是從既有檔案中切出來的。
  每個類別只貢獻 3 張驗證影像，這也是各類別 recall 只能取 {0, ⅓, ⅔, 1} 這四個值、
  其標準差偏粗的原因。這是 seed 之間變異的主要來源，也因此信賴區間較寬。
- 292 張影像中僅 32 張的短邊小於 224 px，因此 Q2 的低解析度子群很小
  （每組 seed 約 n≈8）。該結果僅供判讀趨勢方向，不用來排名策略。
- `C=1.0` 未經調參。在每類僅 7–21 張影像的情況下，正則化強度會實質改變決策邊界，
  因此它是 M2 一個自然的著力點。
- Q2 的幾何稽核是由 transform 定義解析計算而得，而非掃描暗色像素。較早的像素掃描
  版本曾對 `direct` 回報非零的「padding」，但該策略其實完全不會填入任何 padding ——
  它把照片本身真正的黑色內容算進去了。解析版本另能區分各策略在取捨的兩種失效模式 ——
  虛構的零填充，被丟棄的原始內容。`resize_crop` 是唯一同時犯兩種錯的策略。
- 已在 Linux 上以 Python 3.12 驗證。程式碼本身與平台無關，且上方已提供 Windows
  說明，但報告中的數字並未在 Windows 上重新實測。
