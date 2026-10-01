# YuE2 を Kaggle の無料 GPU で動かす

`m-a-p/YuE2-3B` + `YuE2-Vae` を、Kaggle の無料アクセラレータ（P100 16 GB / T4 x2 16 GB×2）で
動かすためのノートブックとドライバです。上流の `yue2` パッケージは**一切改造せず**、そのまま呼び出します。

## 結論（先に要点）

| 項目 | 判定 |
|---|---|
| Kaggle の無料枠 | P100 (16 GB, cc 6.0) または T4 x2 (16 GB×2, cc 7.5)。セッション上限 12 時間、20 分無操作で停止、GPU は週 30 時間、`/kaggle/working` は 20 GB |
| **P100** | **不可**。`torch.cuda.is_bf16_supported()` が False になり、上流 pipeline が `RuntimeError: The unquantized preset requires CUDA BF16 support` で起動を拒否します（プリフライトで事前に検出して案内します） |
| **T4** | **起動は可能**。Turing では PyTorch が BF16 をエミュレーションするため `is_bf16_supported()` は True を返します。数値は正しいが**速くない**（ネイティブ BF16 の数分の一〜数十分の一の速度） |
| メモリ | 16 GB で**収まる見込み**。重み bf16 6.76 GiB + VAE fp32 0.49 GiB + KV キャッシュ（0.109 MiB/トークン）+ 中間活性。24 GB 前提の既定値ではなく、VRAM から自動算出した設定を使います |
| 時間 | **ここが最大の制約**。公式・コミュニティ計測では RTX 4090 で 3.6 分の曲に約 71 秒。T4 は同条件の 10〜30 倍程度を見込むため、**数十分/曲**。正確な数字はノート内の実測ベンチが出します |
| FP8 量子化 | 使えません（`quantization="fp8"` は compute capability 8.9 以上が必要 = Ada/Hopper 以降） |

したがって「**無料枠で動くか？**」の答えは **「T4 を選べば動くが、遅い。まず 1〜2 分の曲で試し、実測 ETA を見てから本番」** です。
どうしても現実的な時間で回したいなら、24 GB 級の BF16 GPU（A10G / L4 / A100 など）や GGUF/MLX 系の
軽量ランタイム・公式デモの方が確実です。

## ファイル

| ファイル | 役割 |
|---|---|
| **[`STEPS_ja.md`](STEPS_ja.md)** | **Kaggle を開くところからの手順書（最初はこれ）** |
| `YuE2_Kaggle.ipynb` | Kaggle に取り込むノートブック。設定 → GPU 確認 → インストール → プリフライト → 実測ベンチ → 生成 → 試聴 |
| `yue2_kaggle.py` | ノートに埋め込まれるドライバ。CLI として単体でも使えます |
| `build_notebook.py` | `yue2_kaggle.py` をノートブックへ埋め込むビルダー（ドライバを編集したら `python kaggle/build_notebook.py`） |
| `test_yue2_kaggle_logic.py` | torch 不要のロジックテスト（`python kaggle/test_yue2_kaggle_logic.py`） |

## 使い方（ノートブック）

1. Kaggle → **Code → New Notebook → File → Import Notebook** で `YuE2_Kaggle.ipynb` を取り込む
2. 右上 **Settings**: **Accelerator = GPU T4 x2**、**Internet = On**
3. 「設定」セルの `STYLE` / `LYRICS` / `SECONDS` などを編集
4. **Save & Run All (Commit)** を推奨（最大 12 時間、ブラウザを閉じても継続）。対話実行は 20 分無操作で停止します
5. 出力は `/kaggle/working/outputs/<id>/`（`audio.flac` ほか）。Output タブからダウンロード

## 使い方（CLI / ローカル）

```bash
python kaggle/yue2_kaggle.py check                 # 環境診断のみ（モデルは読まない）
python kaggle/yue2_kaggle.py bench  --seconds 120  # 実測ベンチと ETA
python kaggle/yue2_kaggle.py run \
    --outdir /kaggle/working/outputs/song1 \
    --style "English, indie pop, warm lead vocal" \
    --lyrics-file lyrics.txt \
    --seconds 120 --cot full
# 同じ --outdir で run をやり直すと、終わった段はスキップして続きから再開します
```

主なオプション: `--cot full|melody|off`、`--abc-file score.abc`（譜面を自分で渡すと譜面生成を丸ごと省略）、
`--plan-max-tokens N`（ABC 段の上限。スモークテスト用）、`--ode-steps N`、`--budget-minutes`、
`--offload-ar/--no-offload-ar`、`--backend auto|torch|torch-eager|vllm`。

## 16 GB で動かすために何を変えたか

上流の既定は「BF16 ネイティブ GPU / 24 GB VRAM」です。このドライバは**次の 5 点だけ**を環境に合わせます
（モデルの数式・サンプリング既定・成果物プロトコルには触れません）。

1. `memory_budget_gib` を実測 VRAM から算出（例: 15.9 GiB → 15.4。上流は 24 前提で、内部で 2 GiB を予約）
2. `vae_core_frames=512`（デコーダのピーク削減。上流は予算 > 12 GiB なら 1024）
3. `offload_ar=True`（VRAM < 15.5 GiB のとき。flow matching 中に AR 側の重みを CPU へ退避）
4. semantic 段の `max_tokens` を要求長に合わせて制限（KV キャッシュは `prefix + max_tokens` 分だけ確保されるため、
   9000 固定をやめると VRAM と無駄な生成を同時に削減）
5. CUDA graph / flash attention が Turing で拒否された場合、`torch-eager` に**自動フォールバック**して続行

## 実行の仕組み（セッション切れに耐える）

```
plan（ABC 譜の生成 or 外部譜の受け取り）      -> plan_manifest.json / score.abc / prefix.npy
semantic（AR でコーデックトークン生成）        -> semantic.npy
flow matching（NAR、32 中点ステップ×2 評価）   -> latent_chunk_XXXX.npy -> latent.npy
decode（VAE, 48 kHz stereo）                  -> audio.flac / result.json
```

各段の成果物は出力ディレクトリに保存され、`run` は**終わった段をスキップ**します。12 時間上限や
カーネル再起動で落ちても、同じ設定で再実行すれば続きから進みます（`plan` は `SymbolicPlan.load` で復元）。
NAR は元のチャンク単位でファイルに保存するため、長い曲でもチャンク単位で再開できます。

`result.json` は上流の成果物マニフェスト（`verify_result` で検証可能）そのままで、ドライバ自身の
ログ `kaggle_run.json` はマニフェスト生成後に書かれます。つまり**成果物のハッシュ検証を壊しません**。

## 実測ベンチが測っているもの

| 段 | 測定方法 | ETA への反映 |
|---|---|---|
| plan | ABC 生成を 16 トークンに制限して実行、`output_tps` を使用 | 想定譜面長 ÷ TPS（譜面長は frames×0.8、上限 4096） |
| semantic | AR 生成を 16 トークンに制限、`output_tps` を使用 | 要求 frames ÷ TPS |
| flow matching | NAR を 1 ソルバステップで 128/256 フレーム実行し線形フィット | チャンク数×固定費 + frames×per-frame×2×ode_steps |
| decode | VAE で 64 フレーム復号 | frames × per-frame |

これらは**推定**です。値を「公式ベンチマークの再現」として扱わないでください。ETA が予算を超える場合は
実行せずに止まります（`FORCE_RUN=True` で強行可）。

## 制約・注意

- **ライセンス**: YuE2 の重みは CC BY-NC 4.0。個人クリエイターの生成物の利用・収益化は許諾されていますが、
  企業の商用利用は別途ライセンスが必要です（`MODEL_LICENSE` 参照）。Kaggle のノートは既定で公開なので、
  出力を公開したくなければ Private にしてください。
- **品質**: `ode_steps` を下げる、`cot="off"` にする、`plan_max_tokens` で譜面を切る、といった変更は
  品質そのものを変えます。比較は同一条件で行い、結果を公式スコアと混同しないでください。
- **T4 の BF16 エミュレーション**は数値の正しさを損ないませんが、速度はネイティブの数分の一以下です。
- モデル（約 7.8 GB）はセッションごとに再ダウンロードになります。`PERSIST_MODEL=True` で
  `/kaggle/working/hf-cache` に置き、そのフォルダを Private Dataset 化しておくと次回 `/kaggle/input`
  から自動検出して再利用できます。
- vLLM バックエンド（`--backend vllm`）は AR 段の高速化が期待できますが、Kaggle のイメージに
  `vllm==0.19.0` を入れる必要があり、Turing では非対応・低速な場合があります。既定では使いません。
- インストールはセル1の `INSTALL_MODE` で選びます。既定の **`"image"`** は Kaggle 同梱の
  **numpy / torch をそのまま維持**し、yue2 本体（`--no-deps`）と依存（`transformers==4.57.6` 等）だけを
  入れます。**`"pinned"`** は上流の完全ピン留め（`numpy==2.2.6`）ですが、numpy を下げるため
  **イメージ同梱の scipy / scikit-learn と不整合になりえます**（症状: `ImportError: cannot import name
  '_center' from 'numpy._core.umath'`）。壊れたら Restart session して `"image"` に戻してください。

## 検証状況

- ローカル（CPU / stub）で、**実物の `yue2` データクラスと `verify_result` 検証**に対して、
  4 段の実行・成果物生成・段単位の再開・チャンク単位の再開・ベンチ/ETA の配線を確認済み
  （`test_yue2_kaggle_logic.py` は依存なしで実行可能）。
- GPU 実機（Kaggle T4 / P100）での速度測定は、このノートの実測ベンチがそのまま行います。
  **T4 での実測値は未取得**なので、上の「10〜30 倍」は 4090 の計測値からの見積もりです。
