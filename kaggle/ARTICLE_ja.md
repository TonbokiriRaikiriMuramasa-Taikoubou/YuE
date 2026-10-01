# 無料の GPU で「曲が生成できる」を確かめる — YuE2 を Kaggle で動かす

> この記事の主旨は、良い曲を自慢することではありません。
> **「歌詞とスタイルを与えるだけで、無料の GPU でも曲が生成できる」** という問いを、
> 誰でも自分の手で確かめられる形にすることです。生成された 1 曲より、
> 「曲が作れる」と知ったときの景色のほうが価値がある、という立場で書いています。

---

## 1. 何が起きるのか

オープンウェイトの楽曲生成モデル **[YuE2](https://github.com/multimodal-art-projection/YuE)**（`m-a-p/YuE2-3B`）は、
歌詞とスタイル（ジャンル・楽器・声・テンポ…）を受け取って、**48 kHz ステレオの歌もの**を出力します。
面白いのは、音をいきなり作るのではなく、途中で**譜面（ABC 記法）を書く**ことです。

```
歌詞 + スタイル
      │  plan（ABC 譜を生成: メロディ + コード進行）   ← ★ここが目で読める
      ▼
   score.abc
      │  semantic（AR モデルが音楽トークンを生成）
      ▼
   semantic.npy
      │  flow matching（NAR が音響潜在を生成、32 ステップ）
      ▼
   latent.npy
      │  decode（VAE → 48 kHz ステレオ）
      ▼
   audio.flac
```

この「間に譜面が挟まる」構造のおかげで、**生成物を白箱として覗けます**。
`score.abc` を開けば「どんなメロディとコードで作られたか」が分かり、
書き換えて再生成すれば別アレンジになります（YuE2 の売りはこの編集可能性です）。

## 2. 用意するもの

| 必要なもの | 内容 |
|---|---|
| Kaggle アカウント | Google アカウントでログイン可 |
| 電話番号認証 | これが無いと GPU もインターネットも選べません |
| 実行するノート | このリポジトリの [`kaggle/YuE2_Kaggle.ipynb`](YuE2_Kaggle.ipynb) |

Kaggle の無料アクセラレータは次の 2 択です。

| 選択肢 | 判定 |
|---|---|
| **GPU T4 x2**（16 GB × 2, cc 7.5） | ✅ 動きます。ただし **BF16 がエミュレーション**なので遅い |
| GPU P100（16 GB, cc 6.0） | ❌ 起動時に BF16 非対応で停止します（ノートが先に検出して案内します） |

セッションは最長 12 時間、20 分無操作で停止、GPU は週 30 時間、`/kaggle/working` は 20 GB まで。
モデルの重みは約 7.8 GB で、セッションごとにダウンロードし直します（後述のコツで節約可）。

## 3. 動かす（最短経路）

> **確認済み**: このノートは Kaggle の無料枠（T4 x2）で、`audio.flac`（48 kHz ステレオ）が
> 実際に出力されるところまで到達しています。「無料枠では無理だろう」という予想は外れました。

1. Kaggle → **Create → New Notebook → File → Import Notebook** で `YuE2_Kaggle.ipynb` を取り込む
2. 右の **Settings**: **Accelerator = GPU T4 x2**、**Internet = On**
3. **セル1（曲のアイディア）** に歌詞とスタイルを書く。プリセットも使えます:
   ```python
   use_preset("citypop_ja")   # 日本語 / 80s シティポップ / 女性ボーカル
   use_preset("ballad_ja")    # 日本語 / ピアノバラード
   use_preset("lofi_en")      # English / lofi hip hop
   use_preset("rock_en")      # English / ロック
   ```
4. **セル2（生成の設定）** で `QUICK_PREVIEW = True`（60 秒・8 ステップ）にする
5. 上から順に実行。初回は依存のインストールに **3〜8 分**かかります
6. 生成セルが、**まず実測ベンチを取って所要時間（ETA）を出し**、問題なければ自動で走ります

```
--- estimated cost on this device ------------------------------------
  requested audio : 120s -> 3000 latent frames (25 fps), 1 flow-matching chunk(s)
  semantic KV     : 0.34 GiB (157 prefix + 3064 tokens)
  plan           : 5.0 min (assumes 2400 score tokens at 8.0 tok/s; release cap 4096)
  semantic       : 4.2 min
  flow_matching  : 64.5 min (32 midpoint steps = 64 velocity evals)
  decode         : 12s
  total          : 73.9 min
  deadline        : 9.83 h usable of 600 min  -> fits: True
---------------------------------------------------------------------
```

ETA が予算を超えるときは、勝手には走りません（`FORCE_RUN = True` で強行可）。

## 4. どれくらいかかるのか

生成時間の支配項は **flow matching** で、おおよそ `曲の長さ × ODE_STEPS` に比例します。
公式・コミュニティ計測では RTX 4090 で 3.6 分の曲が約 71 秒。T4 は BF16 がネイティブでなく
メモリ帯域も 1/3 程度なので、同じ条件で 10〜30 倍程度を見込んでください。

だからおすすめは 2 段構えです。

1. **当たりを探す**: `QUICK_PREVIEW = True`（60 秒・`ODE_STEPS=8`）
2. **当たりを伸ばす**: `ODE_STEPS = 32`、`SECONDS = 120〜180`

<!-- ここに自分の実測値を貼る: 例「T4 で 60 秒/OODE=8 は X 分、120 秒/ODE=32 は Y 分」 -->

> 生成中の一時停止は 20 分無操作で起きます。長い曲は **Save & Run All (Commit)** を使うと
> ブラウザを閉じても最大 12 時間走ります。途中で落ちても、**段ごとに保存**されているので
> 同じ設定で再実行すれば続きから再開します（plan / semantic / flow matching / decode）。

## 5. 生成物のどこを見るか

`/kaggle/working/outputs/<自動採番>/` に残ります。

| ファイル | 見どころ |
|---|---|
| `audio.flac` | 完成音源（48 kHz ステレオ） |
| **`score.abc`** | **生成された譜面。メロディとコードが読める＝ここが白箱** |
| `result.json` | 成果物のハッシュ付きマニフェスト（改ざん検出） |
| `kaggle_run.json` | 段ごとの所要時間・ベンチ値 |

`score.abc` を書き換えて再実行すると別アレンジになります（ABC を `ABC = """..."""` として渡すと
譜面生成を丸ごと省略でき、そのぶん速くもなります）。ここから先は「聞く」から「編集する」に進めます。

## 6. 限界と注意

- **ライセンス**: YuE2 の重みは **CC BY-NC 4.0**。個人クリエイターの制作物の利用・収益化は
  許諾されていますが、**企業の商用利用は別途ライセンスが必要**です。
- **T4 は速くありません**。BF16 がエミュレーションで、数値は正しいが速度が出ません。
- **品質パラメータを下げると品質が変わります**。`ODE_STEPS` を下げる、`cot="off"` にする、
  譜面を途中で切る、といった操作は「速くする」だけでなく「別の曲にする」ことでもあります。
  評価するときは同じ条件どうしで比べてください。
- **再現性は完全ではありません**。同じシードでも GPU・ドライバ・ライブラリの差で結果は変わります。
- Kaggle のノートは既定で公開です。出力を公開したくなければ Private にしてください。

## 7. つまずきどころ（実際に踏んだもの）

| 症状 | 原因 / 対処 |
|---|---|
| `Could not resolve host: github.com` | **Settings の Internet が Off**（Import 直後は既定 Off）。On → 直らなければ Restart session |
| `ImportError: cannot import name '_center' from 'numpy._core.umath'` | pip が numpy を差し替えて scipy/scikit-learn と不整合。Restart session → 既定の `INSTALL_MODE="image"` で再実行 |
| `FlashAttention only supports Ampere GPUs or newer.` | **T4 では正常**。ノートは最初から `torch-eager` を選びます（遅いが正しい） |
| `CUDA out of memory`（モデル読み込み時） | 前回失敗した実行のテンソルが GPU に残っている。**まず Restart session**、そのうえで `SECONDS` を短く |
| 生成が途中で止まった | 20 分無操作の停止。Commit 実行へ切り替え |

いずれもノートの各セルが `✅` / `❌` と日本語の対処を表示します。

## 8. まとめ

- 無料枠（T4）でも、**歌詞とスタイルから曲が生成できる**。これは確かめられます
- ただし「速い」とは言えません。だから**短い試作で当たりを探し、伸ばす**のが実用的です
- いちばん面白いのは完成音源より **`score.abc`**（生成された譜面）です。
  読める・直せるという性質が、生成 AI を「ガチャ」ではなく「道具」にしています

生成された 1 曲を聴くだけでなく、**「自分で確かめられる」状態を持ち帰ってもらえたら**、
この記事の目的は達成です。

## 参考リンク

- YuE2（本体）: <https://github.com/multimodal-art-projection/YuE>
- モデル: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) / [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)
- Kaggle 用ノートと手順書: [`kaggle/YuE2_Kaggle.ipynb`](YuE2_Kaggle.ipynb) / [`kaggle/STEPS_ja.md`](STEPS_ja.md)
- 詳しい技術的な話・つまずきの一次情報: [`kaggle/README.md`](README.md)
