#!/usr/bin/env python3
"""Build ``YuE2_Kaggle.ipynb`` from this directory's sources.

The notebook embeds ``yue2_kaggle.py`` verbatim in a ``%%writefile`` cell, so the
repository and the notebook can never drift apart. Re-run this script after
editing the driver::

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "yue2_kaggle.py"
NOTEBOOK = HERE / "YuE2_Kaggle.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


HEADER = """
# YuE2 を Kaggle の無料 GPU で動かす（T4 x2 / 16 GB 向け調整済み）

[YuE2](https://github.com/multimodal-art-projection/YuE)（`m-a-p/YuE2-3B` + `YuE2-Vae`）は
公式には **BF16 対応 GPU / 24 GB VRAM** を前提にしています。Kaggle の無料枠は
**P100 (16 GB, cc 6.0)** または **T4 x2 (各 16 GB, cc 7.5)** で、セッションは最大 12 時間、
GPU は週 30 時間まで、`/kaggle/working` は 20 GB まで保存されます。

このノートブックは、その差を埋めるための最小限の調整だけを行います。

| 項目 | 対応 |
|---|---|
| 16 GB VRAM | `memory_budget_gib` を実測 VRAM から算出、`vae_core_frames=512`、VAD は常に 512 フレーム単位 |
| BF16 非ネイティブ (T4) | PyTorch が Turing では BF16 をエミュレートするため**実行は可能**。ただし遅い（実測で確認） |
| P100 (cc 6.0) | `torch.cuda.is_bf16_supported()` が False になり pipeline が起動を拒否 → **T4 x2 を選び直してください** |
| セッション 12 時間 / 20 分アイドル | 生成を **plan → semantic → flow matching → decode** の4段に分割し、各段をディスクに保存。時間切れでも**再開可能** |
| どれくらい時間がかかるか不明 | 実モデルで 1〜2 分の**実測ベンチ**を取り、曲ごとの所要時間(ETA)を先に出します |

**モデルの重みは CC BY-NC 4.0（非商用）です。**個人クリエイターの制作物の利用は許諾されていますが、
企業の商用利用には別途ライセンスが必要です（[MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)）。
カグルのノートは既定で公開になるため、出力を公開したくない場合は右上の Settings で Private にしてください。

## 使い方

1. この `.ipynb` を Kaggle の **Code → New Notebook → File → Import Notebook** で取り込みます。
2. 右上 **Settings** で **Accelerator = GPU T4 x2**、**Internet = On** にします。
3. **セル1（曲のアイディア）** に STYLE と歌詞を書きます（プリセットも使えます）。
   **セル2（生成の設定）** で長さ・品質を決めて、**Save & Run All (Commit)** または上から順に実行します。
   - アイディアを次々試すなら、セル2の `QUICK_PREVIEW = True`（60秒・8ステップ）が便利です。
   - 長時間かかる曲は **Commit** 実行（最大12時間、ブラウザを閉じても継続）を推奨。対話実行は20分無操作で停止します。
   - 途中で止まっても、同じ設定でもう一度実行すれば続きから再開します。
4. 出力は `/kaggle/working/outputs/<id>/`（`audio.flac` など）に保存され、ノートブックの Output からダウンロードできます。
"""

IDEA_CELL = '''
# ============ 1) 曲のアイディア（ここだけ編集すれば試せます） ============
# STYLE : 「言語, ジャンル, ボーカル, 楽器, 雰囲気, テンポ(BPM)」をカンマ区切りで並べます。
#         日本語で歌わせたいときは先頭を "Japanese" にして、LYRICS に日本語をそのまま書けます。
# LYRICS: [Verse] / [Chorus] などのタグで区切り、1行は7音節くらいが歌いやすい長さです。
#         曲の長さ・品質はセル2「生成の設定」で決まります。
import os, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # torch より先に必要

STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, "
    "rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)
LYRICS = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
[Chorus]
Let the day come into view
Every road begins with you"""

# --- アイディアのプリセット: use_preset('名前') を実行すると下の STYLE/LYRICS が入れ替わります ---
PRESETS = {
    "citypop_ja": {
        "style": "Japanese, 80s city pop, female voice, smooth bass, glossy synth, "
                 "tight drums, nostalgic night drive, 108 BPM",
        "lyrics": """[Verse]
交差点 青いネオン
濡れた路面に 揺れる影
[Chorus]
夜が明けるまで このまま
君の横顔 追いかけて""",
    },
    "ballad_ja": {
        "style": "Japanese, warm piano ballad, gentle female voice, strings, "
                 "slow brush drums, tender, 70 BPM",
        "lyrics": """[Verse]
窓辺に置いた 小さな花
名前も知らずに 水をやる
[Chorus]
ありがとうと 言えなかった
その言葉を 歌にする""",
    },
    "lofi_en": {
        "style": "English, lofi hip hop, mellow female voice, dusty rhodes, vinyl noise, "
                 "soft boom bap drums, rainy night, 78 BPM",
        "lyrics": """[Verse]
Rain on the window, slow cassette
Coffee going cold, no regret
[Chorus]
Stay a little, let it loop
Dusty keys and lazy groove""",
    },
    "rock_en": {
        "style": "English, energetic rock, powerful male voice, distorted guitars, "
                 "driving drums, anthemic chorus, 140 BPM",
        "lyrics": """[Verse]
Streetlight sparks on a broken sign
Engine running, crossing the line
[Chorus]
Turn it up, we are not going home
Tonight the highway is our own""",
    },
}

# =================== ここから下は表示用（編集しなくてOK） ===================
_DEFAULT_LYRICS = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
[Chorus]
Let the day come into view
Every road begins with you"""
_DEFAULT_LYRICS_JA = """[Verse]  ネオンが路地に消えていく／足音は雨の刻みを打つ
[Chorus] 夜明けを迎えに行こう／どの道も君から始まる"""

_JA_TERMS = {
    "english": "英語", "japanese": "日本語", "chinese": "中国語", "korean": "韓国語",
    "spanish": "スペイン語", "french": "フランス語", "german": "ドイツ語",
    "acoustic": "アコースティック", "electric": "エレクトリック", "indie": "インディー",
    "pop": "ポップス", "rock": "ロック", "jazz": "ジャズ", "folk": "フォーク",
    "ballad": "バラード", "edm": "EDM", "techno": "テクノ", "lofi": "ローファイ",
    "hiphop": "ヒップホップ", "city": "シティ", "soul": "ソウル", "funk": "ファンク",
    "disco": "ディスコ", "ambient": "アンビエント", "orchestral": "オーケストラ",
    "anime": "アニメ調", "piano": "ピアノ", "guitar": "ギター", "bass": "ベース",
    "drums": "ドラム", "strings": "ストリングス", "violin": "バイオリン",
    "saxophone": "サックス", "synth": "シンセ", "rhodes": "ローズ", "organ": "オルガン",
    "flute": "フルート", "female": "女性", "male": "男性", "voice": "ボーカル",
    "vocal": "ボーカル", "expressive": "表現力豊かな", "soft": "柔らかな",
    "powerful": "力強い", "whisper": "ささやくような", "duet": "デュエット",
    "choir": "合唱", "warm": "温かい", "bright": "明るい", "sad": "切ない",
    "nostalgic": "郷愁のある", "energetic": "エネルギッシュ", "calm": "静かな",
    "dreamy": "夢見心地", "melancholic": "憂いのある", "uplifting": "高揚感のある",
    "cozy": "居心地のよい", "lyrical": "歌詞が伝わる", "memorable": "覚えやすい",
    "melody": "メロディ", "unhurried": "急がない", "phrasing": "歌い回し",
    "rounded": "丸みのある", "light": "軽快な", "steady": "安定した", "groove": "グルーヴ",
    "atmospheric": "雰囲気のある", "epic": "壮大な", "gentle": "優しい", "dark": "暗めの",
    "happy": "楽しい", "bpm": "BPM", "and": "と", "with": "と",
}

def mark(label, passed=True, note=""):
    """各セルの結果を ✅ / ❌ の 1 行で表示します。"""
    print(("✅" if passed else "❌") + f" [{label}] " + ("OK" if passed else "NG")
          + (f" — {note}" if note else ""))
    return passed

def describe_style(style):
    """STYLE を日本語メモに変換（辞書にある単語だけ訳し、残りはそのまま）"""
    parts = []
    for raw in style.split(","):
        words = raw.strip().replace("/", " ").split()
        if not words:
            continue
        text = " ".join(_JA_TERMS.get(w.lower().strip("."), w) for w in words)
        for _ in range(3):   # 漢字・かな・カタカナの間の空白を詰める
            text = re.sub(r"([ぁ-んァ-ヶ一-龥]) ([ぁ-んァ-ヶ一-龥])",
                          lambda m: m.group(1) + m.group(2), text)
        parts.append(text.strip())
    return " / ".join(parts)

def show_song():
    """いまの STYLE / LYRICS の内容を日本語で要約して表示します。"""
    print("🎼 この設定で作られる曲（STYLE の日本語メモ）")
    print("   ", describe_style(STYLE))
    bpm = re.findall(r"(\d+)\s*bpm", STYLE, flags=re.I)
    if bpm:
        print("   テンポ:", " / ".join(f"{b} BPM" for b in bpm))
    lines = [l for l in LYRICS.splitlines() if l.strip() and not l.strip().startswith("[")]
    sections = re.findall(r"\[([^\]]+)\]", LYRICS)
    print(f"🎤 歌詞: {len(lines)} 行 / セクション {sections if sections else '(タグなし)'}")
    if LYRICS.strip() == _DEFAULT_LYRICS.strip():
        print("   日本語訳（既定の歌詞）:")
        for line in _DEFAULT_LYRICS_JA.splitlines():
            print("     " + line)
    else:
        print("   （歌詞は編集済み。意味の訳はここには出ません）")
    print("   ヒント: プリセットを試す → use_preset('citypop_ja') / 'ballad_ja' / 'lofi_en' / 'rock_en'")

def use_preset(name):
    """プリセットを STYLE / LYRICS に読み込んで、内容を表示します。"""
    global STYLE, LYRICS
    if name not in PRESETS:
        raise KeyError(f"プリセット名が違います: {sorted(PRESETS)}")
    STYLE, LYRICS = PRESETS[name]["style"], PRESETS[name]["lyrics"]
    print(f"▶ プリセット '{name}' を読み込みました")
    show_song()

show_song()
mark("1/9 アイディア", True, f"{len([l for l in LYRICS.splitlines() if l.strip()])} 行の歌詞")
'''

SETTINGS_CELL = '''
# ============ 2) 生成の設定（速さ・品質・出力先） ============
from pathlib import Path
import hashlib

COT = "full"            # full: メロディ+コード譜を自動生成 / melody: メロディのみ / off: 譜面なし
SECONDS = 120.0         # 生成する長さ（秒）。まずは 60〜120 がおすすめ（長いほど時間がかかります）
SEED = 831001           # 乱数シード。同じ設定＋同じシードで「同じ狙い」の曲になります
ODE_STEPS = 32          # 公式デフォルト。8〜16 にすると速いが品質はトレードオフ
PLAN_MAX_TOKENS = None  # ABC 譜の上限（公式 4096）。512〜1024 にすると試作が速くなります
ABC = None              # 手持ちの ABC 譜を使う場合は文字列で（例: open("score.abc").read()）

OUTPUT_ID = "auto"      # "auto": 設定内容から自動採番（設定を変えると別フォルダ＝前回の結果を混ぜない）
                        # 名前を付けたいときは "song-01" のように文字列で指定
BUDGET_MINUTES = 600    # この実行で使う時間の上限（12h セッションなら 660 くらいまで可）
SAFETY_MINUTES = 10     # セッション破棄に備えて残す余裕

RUN_BENCHMARK = True    # 実モデルで実測ベンチを取る（1〜2分）。ETA を出してから本番へ
FORCE_RUN = False       # ETA が予算を超えていても強行する

INSTALL_MODE = "image"  # "image"(既定): 同梱の numpy/torch を保つ（安全・推奨）
                        # "pinned": 公式ピン留め。numpy 2.2.6 を入れるため scipy/scikit-learn と
                        #           衝突しうる（壊れたら Restart session して "image" に戻す）
PERSIST_MODEL = False   # True にすると HF キャッシュを /kaggle/working に置く（後で Dataset 化しやすい）
REPO_URL = "https://github.com/multimodal-art-projection/YuE.git"  # 自分の fork に差し替えてもOK
REPO_REF = "main"       # fork の branch 名（例: "arena/01a0f78a-yue"）

# --- アイディアをたくさん試すときの「速いモード」（品質より回転数） ---
QUICK_PREVIEW = False   # True にすると 60秒 / 8ステップ / 譜面512トークン で走ります
if QUICK_PREVIEW:
    SECONDS, ODE_STEPS, PLAN_MAX_TOKENS = 60.0, 8, 512
    print("QUICK_PREVIEW=True: 60秒 / ODE_STEPS=8 / PLAN_MAX_TOKENS=512（確認用・品質は低め）")

# =================== ここから下は計算用（編集しなくてOK） ===================
WORKDIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()
REPO_DIR = WORKDIR / "YuE"
if PERSIST_MODEL:
    import os
    os.environ["HF_HOME"] = str(WORKDIR / "hf-cache")

if OUTPUT_ID in (None, "auto"):
    _key = "|".join([STYLE, LYRICS, COT, str(SECONDS), str(SEED), str(ODE_STEPS),
                     str(PLAN_MAX_TOKENS), str(ABC)])
    OUTPUT_ID = "song-" + hashlib.sha256(_key.encode("utf-8")).hexdigest()[:8]
    print(f"OUTPUT_ID を自動採番: {OUTPUT_ID}（設定を変えると別フォルダになります）")
OUTDIR = WORKDIR / "outputs" / OUTPUT_ID

frames = int(-(-SECONDS // 0.04))          # 1 latent frame = 1920/48000 秒 = 40ms
print(f"長さ {SECONDS:.0f} 秒 = 約 {frames} フレーム / 譜面 cot={COT} / 出力先 {OUTDIR}")
print(f"速度設定: ODE_STEPS={ODE_STEPS}" + (f", 譜面上限={PLAN_MAX_TOKENS}" if PLAN_MAX_TOKENS else ""))
if not (REPO_DIR / "pyproject.toml").is_file():
    print("リポジトリ未取得: 次のセルで git clone します")
mark("2/9 設定", True, f"{OUTPUT_ID}: {SECONDS:.0f} 秒, ODE_STEPS={ODE_STEPS}")
'''

GPU_CELL = '''
# --- まず torch 抜きで確認: GPU の種類とインターネット接続 ---
import subprocess, socket

raw = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(raw if raw else "(nvidia-smi の出力なし = GPU が見えていません)")
print("python", sys.version.split()[0])

def reachable(host, timeout=5):
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError as exc:
        print(f"  {host}: NG ({exc})")
        return False

INTERNET = all([reachable("github.com"), reachable("pypi.org"), reachable("huggingface.co")])
names = [line.split(",")[0].strip() for line in raw.splitlines() if line.strip()]
is_p100 = any("P100" in name for name in names)
if not INTERNET:
    print()
    print(">>> インターネットに接続できていません。右の Settings で Internet を [On] にしてください。")
    print(">>> On にしても直らないときは Run -> Restart session してから、上から順に実行します。")
if is_p100:
    print()
    print(">>> P100 は BF16 非対応で YuE2 は動きません。Settings で Accelerator を GPU T4 x2 に変更し、")
    print(">>> Run -> Restart session してから、もう一度このセルを実行してください。")
mark("3/9 GPU・インターネット", bool(names) and INTERNET and not is_p100,
     f"{names[0] if names else 'GPU なし'} / Internet {'OK' if INTERNET else 'NG'}")
'''


INSTALL_CELL = '''
# --- リポジトリの取得と依存のインストール（初回 3〜8 分程度） ---
import subprocess, sys, os, importlib, importlib.metadata as md

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

def major_minor(text):
    try:
        parts = str(text).split("+")[0].split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)

print("image before:", {n: version(n) for n in
                        ("torch", "numpy", "scipy", "scikit-learn", "transformers")})

try:
    if not (REPO_DIR / "pyproject.toml").is_file():
        if not INTERNET:
            raise RuntimeError("Internet が Off です（バナー 3/9 の診断を参照）")
        if REPO_DIR.exists() and not any(REPO_DIR.iterdir()):
            REPO_DIR.rmdir()          # 失敗した clone の空ディレクトリが残っていたら片付ける
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)],
                       check=True)

    if INSTALL_MODE == "pinned":
        # 公式レシピそのまま。ただし numpy==2.2.6 を入れるため、イメージ同梱の scipy /
        # scikit-learn と不整合になることがあります（numpy._core.umath の ImportError）。
        print("WARNING: pinned モードは numpy を差し替えます。壊れたら Restart session して")
        print("         INSTALL_MODE='image'（既定）でやり直してください。")
        pip(str(REPO_DIR))
    else:
        # 既定: イメージ同梱の numpy / torch は触らない（numpy を下げると scipy/sklearn が壊れる）
        pip("--no-deps", str(REPO_DIR))
        pip("transformers==4.57.6", "huggingface-hub==0.36.2", "safetensors==0.7.0",
            "tiktoken==0.12.0", "soundfile==0.13.1", "accelerate==1.13.0")
        torch_version = version("torch")
        if major_minor(torch_version or "0") < (2, 10):
            print(f"image torch={torch_version}: 上流は torch 2.10 前提なので入れ替えます")
            pip("torch==2.10.0")
        elif not str(torch_version).startswith("2.10"):
            print(f"note: image torch is {torch_version}; 上流のピンは 2.10.0 です（そのまま使います）")
except subprocess.CalledProcessError as exc:
    mark("4/9 インストール", False, "git / pip が失敗しました")
    print("コマンドが失敗しました:", exc.cmd)
    print()
    print("確認すること:")
    print("  1) Settings -> Internet が [On] か（Off だと 'Could not resolve host' で失敗します）")
    print("  2) Settings -> Accelerator が [GPU T4 x2] か（P100 では YuE2 は動きません）")
    print("  3) 一時的なネットワーク障害なら、1分ほど待ってこのセルを再実行")
    print("  4) /kaggle/working の空き容量（保存できるのは 20GB まで）")
    raise

# --- 整合性チェック（numpy を差し替えて scipy/sklearn が壊れていないか） ---
broken = []
for name in ("numpy", "scipy", "sklearn", "transformers", "torch", "soundfile", "tiktoken"):
    try:
        module = importlib.import_module(name)
        print(f"  OK  {name} {getattr(module, '__version__', '')}")
    except Exception as exc:
        broken.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  NG  {name}: {type(exc).__name__}: {exc}")
try:
    from transformers import GenerationMixin  # noqa: F401  ← yue2 が実際に通る import 経路
    print("  OK  transformers.GenerationMixin（yue2 が必要とする import 経路）")
except Exception as exc:
    broken.append(f"transformers.generation: {type(exc).__name__}: {exc}")
    print(f"  NG  transformers.GenerationMixin: {type(exc).__name__}: {exc}")

if broken:
    print()
    print(">>> Python 環境が不整合です（典型: numpy を差し替えて scipy / scikit-learn が壊れた）。")
    print(">>> 対処: Run -> Restart session してから、INSTALL_MODE='image' で上から実行し直します。")
    mark("4/9 インストール", False, "Python 環境が不整合（numpy 差し替えの影響）")
    raise RuntimeError("inconsistent environment: " + " | ".join(broken))

import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print(">>> GPU が見えていません。Settings -> Accelerator を GPU T4 x2 にして Restart session してください。")
mark("4/9 インストール", torch.cuda.is_available(), f"torch {torch.__version__}")
'''


DRIVER_CELL_PREFIX = """%%writefile /kaggle/working/yue2_kaggle.py
"""

IMPORT_CELL = '''
# --- ドライバの読み込み（%%writefile で書いたファイルを使う） ---
import importlib
sys.path.insert(0, "/kaggle/working")          # 上の %%writefile セルが書いたファイル
sys.path.append(str(REPO_DIR / "kaggle"))      # オフライン時はリポジトリ側を使う
import yue2_kaggle as yk
importlib.reload(yk)
written = Path("/kaggle/working/yue2_kaggle.py")
print("driver:", yk.__file__)
print("build :", yk.DRIVER_BUILD)
mark("5/9 ドライバ", written.is_file() and str(yk.__file__) == str(written), f"build {yk.DRIVER_BUILD}")
'''

PREFLIGHT_CELL = '''
# --- プリフライト: GPU・VRAM・ディスク・BF16・import 整合性・ネット接続 ---
report = yk.environment()
READY = yk.print_report(report)["ok"]
mark("6/9 プリフライト", READY, "実行可能" if READY else "上の ❌ の指示に従ってください")
if READY:
    settings = yk.pipeline_settings(report)
    print("recommended:", {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in settings.items()})
else:
    print()
    print(">>> この環境では実行できません。上の表示に従ってください。")
    print(">>> P100 だった場合は Settings -> Accelerator -> GPU T4 x2 -> Run -> Restart session。")
'''

OPTIONS_CELL = '''
# --- 実行オプションを組み立てる（設定セル＋環境から） ---

# HF トークン（モデルが gated の場合は Kaggle Secrets に HF_TOKEN を登録）
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

opts = yk.Options(
    outdir=OUTDIR, style=STYLE, lyrics=LYRICS, cot=COT, seed=SEED, seconds=SECONDS,
    ode_steps=ODE_STEPS, abc=ABC, token=HF_TOKEN, plan_max_tokens=PLAN_MAX_TOKENS,
    budget_minutes=BUDGET_MINUTES, safety_minutes=SAFETY_MINUTES,
)
model_ref, vae_ref = yk.resolve_models(opts, report)
print("曲の長さ:", opts.frames, "frames =", f"{opts.seconds:.0f}s",
      "| semantic max_tokens:", opts.semantic_sampling()["max_tokens"])
print("モデル:", model_ref)
print("VAE   :", vae_ref)
print("既存の出力:", sorted(p.name for p in OUTDIR.iterdir()) if OUTDIR.exists() else "(なし=新規)")
mark("準備: 実行オプション", True, f"{opts.frames} frames / cot={COT}")
'''

DIAGNOSTIC_CELL = '''
# --- 任意: GPU メモリ診断（OOM が出たとき、本番の前に実行すると原因が分かります） ---
# ここでは生成は行いません。モデルを 1 回読んで、GPU メモリの実測値を出すだけです。
import gc
import torch

def gpu_mem(tag):
    free, total = torch.cuda.mem_get_info(0)
    print(f"  {tag:<18} allocated={torch.cuda.memory_allocated(0)/2**30:5.2f} GiB"
          f" reserved={torch.cuda.memory_reserved(0)/2**30:5.2f} GiB"
          f" free={free/2**30:5.2f}/{total/2**30:5.2f} GiB")

print("GPU メモリ:")
gpu_mem("before")
before = torch.cuda.memory_allocated(0) / 2**30
gc.collect(); torch.cuda.empty_cache()
gpu_mem("after gc")
if before > 1.0:
    print()
    print(f">>> このカーネルは既に {before:.1f} GiB を保持しています。前回失敗した実行の残骸です。")
    print(">>> Run -> Restart session してから実行し直すと解放されます（放置すると必ず OOM になります）。")

model_dir, vae_dir = yk.resolve_models(opts, report)
print("モデル:", model_dir)
from yue2.modeling_yue2 import YuE2ForCausalLM
try:
    model = YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                            dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
except TypeError:                      # 旧 transformers は torch_dtype という名前
    model = YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
size = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
print(f"パラメータ: {size:.2f} GiB, dtype: {next(model.parameters()).dtype}（bf16 なら 6.8 GiB 前後が正常）")
model.to("cuda:0")
gpu_mem("after model load")
del model
gc.collect(); torch.cuda.empty_cache()
gpu_mem("after free")
mark("7/9 診断", True, f"モデル {size:.1f} GiB / 残骸 {before:.1f} GiB")
'''

RUN_CELL = '''
# --- ベンチ → ETA → 4段ステージ実行（途中で落ちても再実行で再開） ---
if not READY:
    raise RuntimeError("プリフライトに失敗しています。バナー 6/9 の出力を確認してください。")

try:
    with yk.open_pipeline(report, opts) as handle:
        rates = None
        if RUN_BENCHMARK:
            rates = yk.measure(handle)
            stages = yk.predict(rates, opts)
            yk.print_estimate(stages, opts)
            if not stages["fits_budget"] and not FORCE_RUN:
                print()
                print(">>> ETA が予算を超えています。SECONDS を下げる / ODE_STEPS を下げる /")
                print(">>> FORCE_RUN=True で強行、のいずれかを選んでください。")
                outcome = {"status": "skipped", "reason": "estimate exceeds budget"}
            else:
                outcome = yk.run(handle, opts, rates=rates)
        else:
            outcome = yk.run(handle, opts)
except Exception as exc:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    print()
    print("❌ [7/8 生成] NG —", text[:400])
    print("   考えられる原因と対処:")
    if "_center" in low or "umath" in low or "_nocopy" in low:
        print("   ・Python 環境が壊れています（pip が numpy を差し替えた影響）。")
        print("     Run -> Restart session → バナー 4/9 のインストールセルを INSTALL_MODE='image' で実行し直してください。")
    if "out of memory" in low or ("cuda" in low and "memory" in low):
        print("   ・GPU メモリ不足。まず Run -> Restart session（前回失敗した実行の残骸が GPU に残って")
        print("     いるため、そのまま再実行すると必ず同じ場所で落ちます）。")
        print("   ・そのうえでセル2の SECONDS を 60 などに短くし、ODE_STEPS を下げてください。")
        print("   ・原因を数字で確認したいときは、上の「診断」セルを実行してください。")
    if "resolve host" in low or "connection" in low or "internet" in low:
        print("   ・外部通信に失敗。Settings -> Internet が [On] か確認してください。")
    if "cuda" in low and "available" in low:
        print("   ・GPU が見えていません。Settings -> Accelerator -> GPU T4 x2 -> Restart session。")
    if "flash" in low or "ampere" in low or "cudnn" in low:
        print("   ・GPU が最新の attention 経路に対応していない可能性。ログに 'retrying with")
        print("     backend=torch-eager' が出ていれば自動で切り替わっています。")
    print("   ・分からないときは、この出力の最後の20行をそのままコピーして相談してください。")
    raise
else:
    status = outcome.get("status")
    if status in ("complete", "already-complete"):
        mark("8/9 生成", True, f"{outcome.get('outdir')}"
                               + ("（前回の続きが完了済みでした）" if status != "complete" else ""))
    elif status == "interrupted":
        mark("8/9 生成", False, "時間切れで安全に停止しました。そのまま再実行で続きから再開します")
    else:
        mark("8/9 生成", False, f"{status}: {outcome.get('reason', '')}")
'''

RESULT_CELL = '''
# --- 結果の確認と試聴 ---
import json
from IPython.display import Audio, display

audio_path = OUTDIR / "audio.flac"
if audio_path.is_file():
    result = json.loads((OUTDIR / "result.json").read_text(encoding="utf-8"))
    print(f"audio: {result['audio_seconds']:.1f}s @ {result['sample_rate']} Hz "
          f"| truncated: {result['truncated']}")
    print("files:", ", ".join(f"{p.name} ({p.stat().st_size/1e6:.1f} MB)"
                              for p in sorted(OUTDIR.iterdir())))
    display(Audio(filename=str(audio_path)))
    score = OUTDIR / "score.abc"
    print("譜面(ABC):", score if score.exists() else "(cot=off のため譜面なし)")
    run_state = OUTDIR / "kaggle_run.json"
    if run_state.is_file():
        print(json.dumps(json.loads(run_state.read_text(encoding="utf-8")).get("stages", {}),
                         indent=1, ensure_ascii=False))
    mark("9/9 結果", True, f"{result['audio_seconds']:.0f} 秒の音声")
else:
    print("まだ音声がありません。バナー 8/9 の生成セルを実行してください（再実行で続きから再開します）。")
    mark("9/9 結果", False, "audio.flac がまだありません")
'''

TIPS = """
## アイディアを効率よく試す

| やりたいこと | 操作 |
|---|---|
| とりあえず別ジャンルを聴く | セル1で `use_preset("citypop_ja")`（`ballad_ja` / `lofi_en` / `rock_en` も） |
| 自分の歌詞で試す | セル1の `STYLE` と `LYRICS` を書き換えるだけ（他のセルは触らない） |
| 日本語で歌わせる | `STYLE` の先頭を `"Japanese"` にして、`LYRICS` に日本語をそのまま |
| 短時間でたくさん聴き比べる | セル2の `QUICK_PREVIEW = True`（60秒 / `ODE_STEPS=8` / 譜面 512 トークン） |
| 当たりを引いたら本番 | `QUICK_PREVIEW = False` に戻して再実行（`OUTPUT_ID` は自動で別フォルダになります） |
| 同じ曲の長い版が欲しい | セル1〜2の歌詞・STYLE・`SEED` はそのまま、`SECONDS` だけ伸ばす |

- **`OUTPUT_ID` は既定で「設定内容のハッシュ」から自動採番**されます。設定を変えると別フォルダになるので、
  前回の途中結果と混ざりません（同じ設定で再実行したときだけ、続きから再開します）。
- フォルダ名を自分で決めたい場合は `OUTPUT_ID = "song-01"` のように文字列を入れてください。
- `SEED` を変えると別テイクになります。T4 では同じシードでも完全な再現は保証されません（GPU/ドライバ差）。

## ✅ / ❌ の見方

各セルの最後に 1 行の判定が出ます。**1〜6 がすべて ✅ なら、生成を始めて大丈夫**です。

```
✅ [1/8 設定] OK — song-01: 120 秒, cot=full
...
❌ [7/8 生成] NG — 原因の候補と対処が続けて表示されます
```

- 生成（7/8）は長いので、途中の経過は `[kaggle] ...` のログで確認します
- ❌ が出たら、そのセルの下に書かれている対処に従ってください

## セル1が出す「日本語メモ」

セル1を実行すると、`STYLE` の意味（言語・ジャンル・ボーカル・楽器・雰囲気・BPM）と、
歌詞の行数・セクション、既定の歌詞の日本語訳が表示されます。自分の歌詞に書き換えたときは
訳は出ません（構成だけ表示）。英単語の辞書に載っている語は自動で日本語になります。

## 詰まりやすい点とコツ

**時間の見積もり。** 公式リポジトリの資料やコミュニティ計測では、RTX 4090 で 3.6 分の曲に約 71 秒
とされています。T4 は BF16 がネイティブでなくメモリ帯域も 1/3 程度なので、**同条件で 10〜30 倍程度**かかる
ことを想定してください。正確な数字はこのノートの実測ベンチが出します。内訳の支配項は
flow matching（`frames × 28層 × 64 velocity evals`）で、`ODE_STEPS` と曲の長さにほぼ比例します。

**短く試す。** まず `SECONDS=60〜120`, `COT="melody"` + 手持ち ABC, `ODE_STEPS=8〜16` で
「鳴るかどうか」を確かめ、それから本番設定に上げるのが安全です。`cot="off"` は譜面生成を省ける代わりに
guidance 1.01（CFG 2分岐）になり、メモリも計算もほぼ倍になるので T4 では不利です。

**時間切れの扱い。** 実行セルは `BUDGET_MINUTES - SAFETY_MINUTES` を過ぎると pipeline の
`cancelled` フックで**安全に停止**します（例外で落ちず、status が `interrupted` になります）。
このときも完了済みの段はディスクに残っているので、再実行で続きから進みます。

**再開の仕組み。** 出力ディレクトリに `plan_manifest.json` / `semantic.npy` / `latent.npy` /
`latent_chunk_XXXX.npy` / `audio.flac` が段ごとに保存されます。同じ `OUTPUT_ID` で再実行すると、
終わった段はスキップして続きから進みます。`Plan` が保存されているので、歌詞を変えずに再開するのは安全です
（歌詞やスタイルを変えるときは `OUTPUT_ID` を変えてください）。

**セッションを跨ぐコスト。** モデル(約 7.8 GB)はセッションごとに Hugging Face から取り直します
（数分）。`PERSIST_MODEL=True` にすると `/kaggle/working/hf-cache` に置かれ、そのフォルダを
Kaggle Dataset（Private）にすれば次回 `/kaggle/input` から自動検出して再利用できます。

**メモリ。** このノートは `memory_budget_gib` を VRAM から自動決定し、`vae_core_frames=512`、
必要なら `offload_ar=True` を使います。それでも OOM になる場合は `SECONDS` を短くしてください
（semantic の KV キャッシュは要求長に比例して縮みます）。

**品質についての注意。** T4 の BF16 はエミュレーションで**数値は正しくても速くない**だけです。
一方 `ODE_STEPS` を下げる/譜面なし(`cot="off"`)にするといった変更は品質そのものを変えます。
比べるときは同じ条件どうしで比べ、結果を「公式ベンチマークの再現」と呼ばないでください。

**うまくいかないとき。**
- `FlashAttention only supports Ampere GPUs` などが出たら、このノートは自動で `torch-eager` に切り替えます
  （CUDA graph / flash を使わない経路。遅いが T4 でも動きます）。
- `ImportError: cannot import name '_center' from 'numpy._core.umath'` → numpy を差し替えた影響です。
  Run → Restart session して `INSTALL_MODE="image"`（既定）で上から実行し直してください。
- GPU が P100 だった → Settings で `GPU T4 x2` に変更。
- CUDA OOM → `SECONDS` を下げる、`ODE_STEPS` を下げる、ノートを再起動。
- 生成が途中で止まった → 出力ディレクトリを確認し、そのまま再実行（続きから再開）。

**代替手段。** どうしても T4 で時間が足りない場合、量子化版（GGUF / MLX）や
公式のオンラインデモ・API を使う手もあります。このリポジトリの公式要件（24 GB BF16）を満たす
GPU が使えるなら、そちらの方が確実です。

## 参考

- 公式リポジトリ: https://github.com/multimodal-art-projection/YuE
- モデル: https://huggingface.co/m-a-p/YuE2-3B / https://huggingface.co/m-a-p/YuE2-Vae
- ライセンス: 重みは CC BY-NC 4.0（個人の制作物の収益化は許諾、企業の商用は要相談）
"""


def build() -> dict:
    driver = DRIVER.read_text(encoding="utf-8")
    cells = [
        md(HEADER),
        md("## 1. 曲のアイディア（STYLE と歌詞）\n\nここを書き換えるだけで別の曲を試せます。"
           "プリセットも使えます: `use_preset(\"citypop_ja\")`。"),
        code(IDEA_CELL),
        md("## 2. 生成の設定（長さ・品質・出力先）"),
        code(SETTINGS_CELL),
        md("## 3. GPU の確認（インストール前）"),
        code(GPU_CELL),
        md("## 4. 依存のインストール"),
        code(INSTALL_CELL),
        md("## 5. 実行ドライバの書き出し"),
        code(DRIVER_CELL_PREFIX + driver),
        code(IMPORT_CELL),
        md("## 6. プリフライト"),
        code(PREFLIGHT_CELL),
        md("## 7. 実行オプション（自動）"),
        code(OPTIONS_CELL),
        md("## 8. うまく動かないときの診断（任意）"),
        code(DIAGNOSTIC_CELL),
        md("## 9. 実測ベンチ → 生成（再開可能）"),
        code(RUN_CELL),
        md("## 10. 結果"),
        code(RESULT_CELL),
        md(TIPS),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> int:
    notebook = build()
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # sanity: the embedded cell must reproduce the driver (modulo trailing newline)
    writer = next(cell for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(DRIVER_CELL_PREFIX))
    embedded = "".join(writer["source"])[len(DRIVER_CELL_PREFIX):]
    if embedded.rstrip("\n") != DRIVER.read_text(encoding="utf-8").rstrip("\n"):
        raise SystemExit("embedded driver differs from kaggle/yue2_kaggle.py")
    print(f"wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1024:.0f} KiB, "
          f"{len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
