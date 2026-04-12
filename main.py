"""
グローバル（英語）の BrainTech / BCI / 神経科学ニュースを Google News RSS から取得し、
Gemini で日本語要約を生成して Bluesky に投稿する。

環境変数（.env）:
  GEMINI_API_KEY, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  任意: GEMINI_MODEL（既定: gemini-2.5-flash → gemini-2.5-flash-lite）
  任意: DRY_RUN=1 で Bluesky への投稿をスキップし、生成テキストのみ表示

直近投稿した記事 URL を last_post.txt に保存する。
取得した記事 URL が（正規化後）それと一致する場合は投稿・Gemini を一切行わず終了する。
"""

from __future__ import annotations

import os
import re
import sys
from html import unescape
from pathlib import Path

import feedparser
from atproto import Client
from dotenv import load_dotenv
from google import genai

# 英語圏のグローバルニュース（記事本文の要約は Gemini で日本語に変換）
RSS_URL = (
    "https://news.google.com/rss/search?"
    "q=Brain-Computer+Interface+OR+BCI+OR+BrainTech+OR+Neuroscience"
    "&hl=en&gl=US&ceid=US:en"
)

# Bluesky の投稿は 300 文字まで（本文末尾にスペース＋記事 URL を含めた合計）
POST_MAX = 300
DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
LAST_POST_FILE = Path(__file__).resolve().parent / "last_post.txt"


def normalize_article_url(url: str) -> str:
    """比較・保存用に URL を揃える（末尾スラッシュや前後空白の差で二重投稿しない）。"""
    s = (url or "").strip()
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def strip_html(text: str) -> str:
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def load_env() -> None:
    load_dotenv()
    required = (
        "GEMINI_API_KEY",
        "BLUESKY_HANDLE",
        "BLUESKY_APP_PASSWORD",
    )
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"環境変数が未設定です: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def fetch_latest_entry() -> tuple[str, str, str]:
    parsed = feedparser.parse(RSS_URL)
    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
        print(f"RSS の解析に失敗しました: {getattr(parsed, 'bozo_exception', '')}", file=sys.stderr)
        sys.exit(1)
    entries = getattr(parsed, "entries", []) or []
    if not entries:
        print("RSS にエントリがありません。", file=sys.stderr)
        sys.exit(1)
    e = entries[0]
    title = strip_html(getattr(e, "title", "") or "")
    link = normalize_article_url((getattr(e, "link", "") or "").strip())
    raw = getattr(e, "summary", None) or getattr(e, "description", "") or ""
    content = strip_html(raw) or title
    if not title:
        print("タイトルを取得できませんでした。", file=sys.stderr)
        sys.exit(1)
    if not link:
        print("リンクを取得できませんでした。", file=sys.stderr)
        sys.exit(1)
    return title, content, link


def read_last_post_url() -> str | None:
    if not LAST_POST_FILE.is_file():
        return None
    try:
        s = LAST_POST_FILE.read_text(encoding="utf-8").strip()
        return normalize_article_url(s) if s else None
    except OSError:
        return None


def write_last_post_url(url: str) -> None:
    LAST_POST_FILE.write_text(normalize_article_url(url) + "\n", encoding="utf-8")


def body_char_budget(url: str) -> int:
    """本文のみに使える最大文字数（末尾にスペース＋URL を付与し合計 POST_MAX 以内にする）。"""
    sep = 1
    return max(0, POST_MAX - len(url) - sep)


def build_prompt(title: str, content: str, url: str) -> str:
    budget = body_char_budget(url)
    return f"""あなたはブレインテック（BCI・脳コンピュータインタフェース・神経科学など）に詳しい編集者です。
以下のニュースは英語ソースからの情報です。内容を踏まえ、Bluesky 向けの「投稿本文」だけを**日本語で**書いてください。

【記事タイトル（英語のまま引用可）】
{title}

【記事の概要・内容（英語など原文のまま）】
{content}

【投稿末尾に付けるURL（本文には含めない。スペース1つ＋このURLで300字以内に収める）】
{url}

【最優先のトーン・内容（すべて日本語で）】
・記事を理解するうえで必要な前提知識を、必ず日本語で補足すること（短い一文や括弧書きでよい）。
・専門用語は日本語で噛み砕き説明し、要点を絞ってわかりやすくまとめること。
・英語のまま出力してはいけない（固有名詞・製品名・論文名など必要最小限の英単語のみ可）。

【形式】
・出力は「日本語の投稿本文のみ」。引用符・箇条書き・「投稿:」などのラベルは付けないこと。
・上記URLは本文に含めない。
・本文の文字数上限: {budget} 文字（日本語は1文字で1と数える）。改行は使わないこと。
・最終投稿は「日本語本文＋スペース＋URL」の合計が厳密に {POST_MAX} 文字以下になるようにすること（URLの文字数は {len(url)} 文字）。"""


def _response_text(response: object) -> str:
    try:
        t = getattr(response, "text", None)
        return (t or "").strip() if t is not None else ""
    except (ValueError, AttributeError):
        return ""


def generate_post_text(title: str, content: str, url: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL")
    candidates = [model_name] if model_name else list(DEFAULT_MODELS)

    prompt = build_prompt(title, content, url)
    last_error: Exception | None = None
    cap = body_char_budget(url)
    for name in candidates:
        if not name:
            continue
        try:
            response = client.models.generate_content(model=name, contents=prompt)
            body = _response_text(response)
            body = re.sub(r"\s+", " ", body)
            if cap and len(body) > cap:
                body = body[:cap].rstrip()
            if body:
                return body
            last_error = RuntimeError("Gemini が空の応答を返しました")
        except Exception as ex:
            last_error = ex
            continue
    print(f"Gemini の生成に失敗しました: {last_error}", file=sys.stderr)
    sys.exit(1)


def compose_post(body: str, url: str) -> str:
    """本文 + スペース + URL。合計 POST_MAX 文字以内。"""
    body = re.sub(r"\s+", " ", body.strip())
    sep = " "
    suffix = f"{sep}{url}" if url else ""
    if not url:
        return body[:POST_MAX]
    if not body:
        return url[:POST_MAX] if len(url) <= POST_MAX else url[:POST_MAX]
    out = body + suffix
    if len(out) <= POST_MAX:
        return out
    room = POST_MAX - len(suffix)
    if room < 1:
        return url[:POST_MAX]
    return body[:room].rstrip() + suffix


def _dry_run() -> bool:
    v = (os.getenv("DRY_RUN") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def post_to_bluesky(text: str) -> None:
    if _dry_run():
        print("[DRY_RUN] Bluesky への投稿はスキップしました。", file=sys.stderr)
        return
    client = Client()
    client.login(
        os.environ["BLUESKY_HANDLE"],
        os.environ["BLUESKY_APP_PASSWORD"],
    )
    client.send_post(text)


def main() -> None:
    load_env()
    title, content, url = fetch_latest_entry()
    last_key = read_last_post_url()
    if last_key is not None and url == last_key:
        # 重複時は Gemini / Bluesky / last_post 更新を一切行わない（Early Return）
        print("No new news found. Skipping post.")
        raise SystemExit(0)
    # 「スペース + URL」単体で 300 文字を超える場合は投稿不可
    if len(url) > POST_MAX - 1:
        print(
            f"記事URLが長すぎます（{len(url)} 文字）。Bluesky の1投稿は {POST_MAX} 文字までです。",
            file=sys.stderr,
        )
        sys.exit(1)
    body = generate_post_text(title, content, url)
    post_text = compose_post(body, url)
    post_to_bluesky(post_text)
    if not _dry_run():
        write_last_post_url(url)
    print(post_text)


if __name__ == "__main__":
    main()
