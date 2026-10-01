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

**各セルの最後に `✅ [n/9 ...] OK` か `❌ [n/9 ...] NG` の判定が出ます。**
セル1〜6 がすべて ✅ なら、生成（バナー 8/9）を始めて大丈夫です。

| バナー | セル | やること | 目安 |
|---|---|---|---|
| 1/9 | **曲のアイディア** | `STYLE` と `LYRICS` を書く／プリセットを試す（`use_preset("citypop_ja")`）。実行すると **日本語メモ** が出ます | 1 分 |
| 2/9 | **生成の設定** | 長さ `SECONDS`、品質 `ODE_STEPS`、出力先などを決める。`QUICK_PREVIEW = True` で速い試作 | 1 分 |
| 3/9 | GPU・インターネット | 何もせず実行。`Tesla T4` と `Internet OK` が出れば ✅。**P100 だったら §2 に戻って T4 x2 に変更し、再起動** | 10 秒 |
| 4/9 | 依存のインストール | 何もせず実行（git clone + pip install + 整合性チェック） | **3〜8 分** |
| 5/9 | ドライバの書き出し＋読み込み | 何もせず実行。`build 2026-10-01.3` のようにドライバの版が出ます | 2 秒 |
| 6/9 | プリフライト | 何もせず実行。GPU・VRAM・import 整合性・ネット接続をまとめて検査 | 5 秒 |
| — | 実行オプション | 何もせず実行。出力フォルダとモデルパスが表示される | 1 秒 |
| 7/9 | 診断（任意） | OOM などで困ったときだけ実行。GPU メモリの実測値が出ます | 20 秒 |
| 8/9 | ベンチ → 生成 | **そのまま待つ**。ベンチ 1〜2 分 → ETA 表示 → `fits: True` なら自動で生成開始 | ベンチ + 生成時間 |
| 9/9 | 結果 | 音声が埋め込まれて再生できます。下部にファイル一覧 | 数秒 |

**セル1の「日本語メモ」** では、`STYLE` を「言語 / ジャンル / ボーカル / 楽器 / 雰囲気 / BPM」に
分解して日本語で表示します（辞書にある単語だけ訳します）。既定の歌詞のときは日本語訳も出るので、
「どんな曲ができるのか」を確認してから実行できます。

**セル2の自動採番**: `OUTPUT_ID = "auto"` のままなら、歌詞・STYLE・長さ・SEED・ステップ数などから
出力フォルダ名が自動で決まります。**設定を変えると別フォルダ**になるので、前回の途中結果と混ざりません
（同じ設定で再実行したときだけ、続きから再開します）。

**初回おすすめ設定** — まず「通るかどうか」を最短で確認するためのものです。

- セル1: そのまま（既定の City Lights）でも、`use_preset("ballad_ja")` でも
- セル2: `QUICK_PREVIEW = True` にすると 60 秒 / 8 ステップ / 譜面 512 トークンで走ります

```python
QUICK_PREVIEW = True    # ★セル2: 速い試作モード（本番は False に戻す）
```

うまく音が出たら、本番設定に上げます。

```python
QUICK_PREVIEW = False   # セル2
SECONDS = 120           # セル2: 2 分
COT = "full"            # セル2: メロディ+コード譜
ODE_STEPS = 32          # セル2: 公式デフォルト（品質優先）
PLAN_MAX_TOKENS = None  # セル2: 公式上限 4096
```

> 日本語の曲を作るには、`STYLE` の先頭を `"Japanese"` にして、`LYRICS` に日本語の歌詞をそのまま書きます
> （例: `STYLE = "Japanese, warm piano pop, female voice, 88 BPM"`）。

> **インストール方法（`INSTALL_MODE`）**: 既定の `"image"` は Kaggle 同梱の numpy / torch を保ったまま
> yue2 本体と依存だけを入れる安全な経路です。`"pinned"` は上流の完全ピン留め（`numpy==2.2.6`）ですが、
> numpy を下げるためイメージ同梱の scipy / scikit-learn と衝突しえます（→ §6 の早見表を参照）。

> `ODE_STEPS` と `PLAN_MAX_TOKENS` を下げるのは**品質を変える**操作です。
> 出来上がりを評価するときは 32 / None に戻し、同じ条件どうしで比べてください。

---

## 3.5 アイディアを次々に試す

| やりたいこと | 操作 |
|---|---|
| とりあえず別ジャンルを聴く | セル1で `use_preset("citypop_ja")`（`ballad_ja` / `lofi_en` / `rock_en` も） |
| 自分の歌詞で試す | セル1の `STYLE` と `LYRICS` を書き換えるだけ（他のセルは触りません） |
| 日本語で歌わせる | `STYLE` の先頭を `"Japanese"` にして、`LYRICS` に日本語をそのまま書く |
| 短時間でたくさん聴き比べる | セル2の `QUICK_PREVIEW = True`（60秒 / `ODE_STEPS=8` / 譜面 512 トークン） |
| 当たりを引いたら本番 | `QUICK_PREVIEW = False` に戻して再実行（出力フォルダは自動で別になります） |
| 同じ曲の長い版が欲しい | 歌詞・`STYLE`・`SEED`・`ODE_STEPS` はそのまま、`SECONDS` だけ伸ばして再実行 |
| 別テイクが欲しい | `SEED` を変える（同じシードでも T4 では完全な再現は保証されません） |

推奨の進め方:

1. **当たりを探す**: `QUICK_PREVIEW = True`（60 秒・8 ステップ）。T4 でも数分〜十数分で 1 曲聴けます
2. **当たりを伸ばす**: `QUICK_PREVIEW = False` にして `ODE_STEPS = 32`、`SECONDS = 120〜180`
3. 生成時間は **flow matching が支配的**で、`曲の長さ × ODE_STEPS` にほぼ比例します
   （8 → 32 で約 4 倍）。セル9のベンチが最初に実測 ETA を出します

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
| `❌ [8/9 生成] NG` が出た | その下に原因候補と対処が日本語で出ます。分からなければ **出力の最後の20行**をコピーして相談 |
| カーネルが古いままかも？ | セル4が出す `build` の値を確認。ノートを取り込み直したのに値が古いときは、ドライバのセル（`%%writefile`）を再実行 |
| GPU クォータが足りない | 右のアカウント表示で残り時間を確認（週 30 時間） |
| CUDA out of memory | **順番が大事**（下の「OOM が出たら」を参照）。まず **Run → Restart session** |
| `FlashAttention only supports Ampere GPUs or newer.` | **T4 では正常な動作です。** ノートは最初から / あるいは自動で `torch-eager` に切り替えます（ログに 1 行）。遅いが正しく動きます |
| 出力が残らない / 保存に失敗 | `/kaggle/working` は 20 GB まで。`PERSIST_MODEL=False`、古い出力を削除 |
| ETA が予算を超えて生成されない | `SECONDS` を下げるか、`FORCE_RUN=True` で強行 |

---

## 6.4 T4 では「自動で安全な経路」に切り替わります（正常）

上流の高速経路（CUDA graph + FlashAttention）は **Ampere 以降（cc 8.0+）専用**です。
T4 (cc 7.5) では使えないため、このノートは **最初から `torch-eager` を選びます**。実行ログに

```
[kaggle] backend='torch-eager' from the start: cc 7.5 has no FlashAttention;
         CUDA graphs are unavailable. Slower than Ampere, but correct.
[kaggle] pipeline ready: backend=torch-eager (cc 7.5 has no FlashAttention; ...)
```

と出ますが、**これはエラーではありません**。もし何かの拍子に高速経路で失敗しても、

```
[kaggle] downgrading in place to torch-eager: ... (released X.XX GiB; the verified
         pipeline is reused, nothing is hashed or loaded again)
```

と出て自動で切り替わり、**モデルの再検証（22秒）や再読み込みは行いません**。

## 6.5 OOM（GPU メモリ不足）が出たら

**いちばん多い原因は「前回失敗した実行のメモリが残っている」ことです。**
Jupyter は最後のトレースバック（エラーの中身）を保持し続けるため、失敗した実行が掴んでいた
モデルのテンソルが GPU に残ったままになります。そのまま再実行すると、同じ場所で必ず落ちます。

1. **Run → Restart session**（最優先。これをせず再実行しても直りません）
2. セル1で `SECONDS` を短くする（例: 60〜120）。必要なら `ODE_STEPS` も下げる
3. 「診断」セルを実行して、数字を確認する

```
GPU メモリ:
  before             allocated= 0.00 GiB reserved= 0.00 GiB free=14.6/14.6 GiB   ← まっさら
パラメータ: 6.76 GiB, dtype: torch.bfloat16                                      ← 正常（bf16）
  after model load   allocated= 6.78 GiB ...
✅ [診断] OK — モデル 6.8 GiB / 残骸 0.0 GiB
```

- `before` が **1 GiB 以上**なら残骸あり → Restart session
- パラメータが **13.5 GiB 前後**（fp32）なら環境異常 → セル5の `runtime:` 行と一緒に相談
- 長時間の本番は **Save & Run All (Commit)** がおすすめです。毎回まっさらなカーネルで走るので
  この残骸問題が起きません

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
