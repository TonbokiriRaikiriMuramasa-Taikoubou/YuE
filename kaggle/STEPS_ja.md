# はじめての Kaggle 実行手順（YuE2 / T4 x2）

所要: 準備 5 分 + 初回実行 10〜20 分（60 秒の曲・スモークテスト）。ここでは「Kaggle を開く」ところから順に書きます。

---

## 0. 事前に用意するもの（5 分）

1. **Kaggle アカウント**（Google アカウントでログイン可）
2. **電話番号認証** — これが無いと GPU を選べません
   `kaggle.com` 右上のアイコン → **Settings** → **Phone Verification** で SMS 認証
3. **ノートブックファイル** `kaggle/YuE2_Kaggle.ipynb` を手元に置く。入手方法はどちらでも:
   - このチャットで開いているファイルをダウンロードする
   - GitHub の Web で次のファイルを開き **Download raw file**:
     `https://github.com/TonbokiriRaikiriMuramasa-Taikoubou/YuE/blob/arena/01a0f78a-yue/kaggle/YuE2_Kaggle.ipynb`
     （branch を `arena/01a0f78a-yue` に切り替えてから `kaggle/` → `YuE2_Kaggle.ipynb`）

---

## 1. Kaggle でノートを作る

1. `https://www.kaggle.com` にログイン
2. 右上の **Create** → **New Notebook**
3. ノートが開いたら、左上メニューの **File → Import Notebook → Upload** で
   先ほどの `YuE2_Kaggle.ipynb` を選ぶ（Import すると新しいタブでノートが開きます）
4. 元の空ノートは閉じて構いません

> うまく Import できないときは、Kaggle の **File → Import Notebook** の画面に
> ノートファイルをドラッグ&ドロップしても同じです。

---

## 2. ノートの設定（ここが一番大事）

右サイドバーの **Settings**（歯車アイコン・セッションオプション）を開きます。

| 設定項目 | 選ぶ値 | 理由 |
|---|---|---|
| **Accelerator** | **GPU T4 x2** | P100 は BF16 非対応で YuE2 が起動しません |
| **Internet** | **On** | `pip install` とモデル（約 7.8 GB）のダウンロードに必須 |
| Persistence | Files only（既定のままで OK） | |
| Environment | 既定のままで OK | |

- 右上のヘッダに「**T4 x2**」と表示されていれば OK
- **Share は Private のまま**にしておきます（Save Version のときに Public にしない）。
  生成した歌詞・音声を公開したくなければ必須です。

> ⚠️ **Import したノートは Internet が既定で Off です。**
> これを On にしないと `git clone` が
> `fatal: unable to access 'https://github.com/...': Could not resolve host: github.com`
> で失敗します（セル2の接続チェックでも NG と出ます）。
> On に切り替えた直後にまだ失敗するときは **Run → Restart session** してから、
> 上から順に実行し直してください。

---

## 3. セルを上から順に実行する

Jupyter の操作は **Shift + Enter** で 1 セル実行して次へ進みます。

| # | セル | やること | 目安 |
|---|---|---|---|
| 1 | 設定 | 歌詞・スタイルを書き換える（下の「初回おすすめ設定」参照） | 1 分 |
| 2 | GPU の確認 | 何もせず実行。`Tesla T4` と出れば OK。**P100 が出たら §2 に戻って T4 x2 に変更し、カーネルを再起動** | 5 秒 |
| 3 | 依存のインストール | 何もせず実行（git clone + pip install + 整合性チェック） | **3〜8 分** |
| 4 | ドライバの書き出し | 何もせず実行（`%%writefile`） | 1 秒 |
| 5 | import | 何もせず実行。`driver: /kaggle/working/yue2_kaggle.py` と出れば OK | 1 秒 |
| 6 | プリフライト | 何もせず実行。`-> unquantized pipeline can run here` が出れば合格 | 2 秒 |
| 7 | 実行オプション | 何もせず実行。`frames = ...` とモデルパスが表示される | 1 秒 |
| 8 | ベンチ → 生成 | **そのまま待つ**。ベンチ 1〜2 分 → ETA 表示 → `fits: True` なら自動で生成開始 | ベンチ + 生成時間 |
| 9 | 結果 | 音声が埋め込まれて再生できます。下部にファイル一覧 | 数秒 |

**初回おすすめ設定**（セル 1）— まず「通るかどうか」を最短で確認するためのものです。

```python
SECONDS = 60            # 60 秒の曲
COT = "melody"          # メロディのみ計画（full より軽い）
ODE_STEPS = 8           # ★スモークテスト用（本番は 32）
PLAN_MAX_TOKENS = 512   # ★譜面生成を短く切る（本番は None = 公式 4096）
RUN_BENCHMARK = True
FORCE_RUN = False
```

うまく音が出たら、次の順で本番設定に上げていきます。

```python
SECONDS = 120           # 2 分
COT = "full"            # メロディ+コード譜
ODE_STEPS = 32          # 公式デフォルト（品質優先）
PLAN_MAX_TOKENS = None  # 公式上限 4096
```

> **インストール方法（`INSTALL_MODE`）**: 既定の `"image"` は Kaggle 同梱の numpy / torch を保ったまま
> yue2 本体と依存だけを入れる安全な経路です。`"pinned"` は上流の完全ピン留め（`numpy==2.2.6`）ですが、
> numpy を下げるためイメージ同梱の scipy / scikit-learn と衝突しえます（→ §6 の早見表を参照）。

> `ODE_STEPS` と `PLAN_MAX_TOKENS` を下げるのは**品質を変える**操作です。
> 出来上がりを評価するときは 32 / None に戻し、同じ条件どうしで比べてください。

---

## 4. 長い曲を回す（Commit 実行）

対話実行（画面で開いたまま）は **20 分間操作しないと自動停止**します。長い曲は
**Save Version → Save & Run All (Commit)** を使います。

- ブラウザを閉じても OK、最長 12 時間まで走ります
- 途中で落ちても、同じ設定でもう一度実行すれば**段ごとに保存された続きから再開**します
- 進捗はそのノートの Output（Logs）で確認できます

---

## 5. 出力の取り出し

ノート右上の **Output** タブを開くと `/kaggle/working/outputs/<OUTPUT_ID>/` の中身が保存されています。

| ファイル | 中身 |
|---|---|
| `audio.flac` | 完成音源（48 kHz stereo）。これが欲しいもの |
| `score.abc` | 生成された譜面（メロディ+コード）。中身を読むと曲の構成が分かります |
| `semantic.npy` / `latent.npy` | 中間表現（再デコードや比較用） |
| `result.json` | 成果物マニフェスト（ハッシュ付き・改ざん検出用） |
| `kaggle_run.json` | 各段の所要秒数・実測ベンチ値（＝次回の見積もり材料） |

MP3 が欲しい場合は結果セルの後に 1 セル足すだけです。

```python
!ffmpeg -y -i {OUTDIR}/audio.flac -b:a 192k {OUTDIR}/audio.mp3
from IPython.display import FileLink; FileLink(f"{OUTDIR}/audio.mp3")
```

---

## 6. つまずいたとき早見表

| 症状 | 原因 / 対処 |
|---|---|
| Accelerator に GPU の選択肢が無い | 電話番号認証が未完了（§0-2） |
| `Could not resolve host: github.com` / `pip` が落ちる / モデルが落ちてこない | **Settings の Internet が Off**。Import したノートは既定で Off です。**On** に変更 → 直らない場合は **Run → Restart session** → 上から実行。セル2の接続チェックで OK を確認できます |
| `ImportError: cannot import name '_center' from 'numpy._core.umath'` （または `numpy` / `scipy` / `sklearn` の ImportError） | pip が **numpy を差し替えた**ため、イメージ同梱の scipy / scikit-learn と不整合になった。**Run → Restart session** して、`INSTALL_MODE="image"`（現在の既定）で上から実行し直す。セル3末尾の整合性チェックが同じ判定をしてくれます |
| 起動時に BF16 エラー | **P100 を引いた**。Settings で T4 x2 に変更 → カーネル再起動 |
| 途中で止まった（20 分放置） | 対話実行の制限。**Commit 実行**に切り替える |
| 同じ設定で再実行したら一瞬で終わった | 前回の続きから再開した正常動作。`already-complete` と出ます |
| GPU クォータが足りない | 右のアカウント表示で残り時間を確認（週 30 時間） |
| CUDA out of memory | `SECONDS` を下げる → `ODE_STEPS` を下げる → ノートを再起動 |
| `FlashAttention ...` のエラー | 自動で `torch-eager` に切り替わります（ログに 1 行出る）。遅いが動きます |
| 出力が残らない / 保存に失敗 | `/kaggle/working` は 20 GB まで。`PERSIST_MODEL=False`、古い出力を削除 |
| ETA が予算を超えて生成されない | `SECONDS` を下げるか、`FORCE_RUN=True` で強行 |

---

## 7. ETA（所要時間の見積もり）の読み方

実行セルは生成の前に、実モデルで 1〜2 分のベンチを取って次のように表示します。

```
--- estimated cost on this device ------------------------------------
  requested audio : 120s -> 3000 latent frames (25 fps), 1 flow-matching chunk(s)
  plan           : 9.0 min (assumes 2400 score tokens at ...; release cap 4096)
  semantic       : 4.0 min
  flow_matching  : 62.0 min (32 midpoint steps = 64 velocity evals)
  decode         : 12s
  total          : 1.25 h
  deadline        : 9.83 h usable of 600 min  -> fits: True
---------------------------------------------------------------------
```

- **flow_matching が支配項**です（32 ステップ = 64 回の速度評価 × 曲の長さ）。曲を半分にすれば半分、`ODE_STEPS` を 8 にすれば 1/4 になります
- T4 は BF16 がエミュレーションなので、RTX 4090 の 10〜30 倍程度かかると見込んでください
- `fits: False` のときは生成せずに止まります（`FORCE_RUN=True` で強行可）

---

## 8. 2 回目以降を楽にするコツ

- **モデルの再ダウンロード（約 7.8 GB）を避ける**
  1. セル 1 で `PERSIST_MODEL = True` にして実行
  2. セッション終了後、Output の `hf-cache` フォルダを **Private な Kaggle Dataset** として保存
  3. 次回はその Dataset を **Add Data** で添付するだけで、ドライバが `/kaggle/input` から自動検出して再利用します
- **歌詞やスタイルを変えたら `OUTPUT_ID` も変える**（同じ ID は前回の続きと見なされます）
- 途中まで作った曲は、`OUTDIR/latent.npy` が残っていれば**デコード以降だけやり直せます**（同じノートを再実行）

---

## 9. 用語ミニ辞典

| 用語 | 意味 |
|---|---|
| Notebook / Kernel | Kaggle の実行環境。セッション単位で起動し、終了すると `/kaggle/working` 以外は消える |
| Commit（Save Version） | ノートをバックグラウンド実行して結果を保存する機能。長いジョブはこれ |
| Accelerator | GPU / TPU の選択。今回は GPU T4 x2 |
| Cell / セル | ノートの 1 ブロック。Shift+Enter で実行 |
| ETA | 所要時間の推定。このノートが実測から計算 |
| Stage（段） | plan / semantic / flow matching / decode の 4 段。段ごとに保存され、再開できる |
