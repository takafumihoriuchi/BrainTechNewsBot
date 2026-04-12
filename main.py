"""
BrainTech 関連ニュースを Google News RSS から取得し、Gemini で要約投稿文を生成、Bluesky に投稿する。
環境変数（.env）:
  GEMINI_API_KEY, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  任意: GEMINI_MODEL（既定: gemini-2.5-flash → gemini-2.5-flash-lite）
  任意: DRY_RUN=1 で Bluesky への投稿をスキップし、生成テキストのみ表示

直近投稿URLはリポジトリ直下の last_post.txt に保存する。
RSS の先頭記事 URL がそれと一致する場合は新しい記事がないとみなし、Bluesky には投稿しない。
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

RSS_URL = (
    "https://news.google.com/rss/search?"
    "q=BrainTech+OR+BCI+OR+Neuroscience&hl=ja&gl=JP&ceid=JP:ja"
)

# Bluesky の投稿は 300 文字まで（本文末尾にスペース＋記事 URL を含めた合計）
POST_MAX = 300
DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
LAST_POST_FILE = Path(__file__).resolve().parent / "last_post.txt"


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
    link = (getattr(e, "link", "") or "").strip()
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
        return s or None
    except OSError:
        return None


def write_last_post_url(url: str) -> None:
    LAST_POST_FILE.write_text(url.strip() + "\n", encoding="utf-8")


def should_skip_no_new_article(current_url: str) -> bool:
    """直近に投稿した記事と同じ URL なら、新規ニュースなしとしてスキップする。"""
    previous = read_last_post_url()
    return previous is not None and current_url == previous


def body_char_budget(url: str) -> int:
    """本文のみに使える最大文字数（末尾にスペース＋URL を付与し合計 POST_MAX 以内にする）。"""
    sep = 1
    return max(0, POST_MAX - len(url) - sep)


def build_prompt(title: str, content: str, url: str) -> str:
    budget = body_char_budget(url)
    return f"""あなたはブレインテック（BCI・神経科学など）に詳しい編集者です。次のニュースをもとに、Bluesky 向けの投稿本文だけを書いてください。

【記事タイトル】
{title}

【記事の概要・内容】
{content}

【投稿に付けるURL（本文の直後にスペース1つ空けて末尾に付与する）】
{url}

【最優先のトーン・内容】
・記事を理解するうえで必要な前提知識を、必ず補足すること（短い一文や括弧書きでよい）。
・専門用語は噛み砕き、要点を絞ってわかりやすくまとめること。

【形式】
・出力は「投稿本文のみ」。引用符・箇条書き・「投稿:」などの余計なラベルは付けないこと。
・上記URLは出力に含めない。本文のみを出力すること。
・本文の文字数上限: {budget} 文字（日本語は1文字で1と数える）。改行は使わないこと。
・最終投稿は「本文＋スペース＋URL」の合計が厳密に {POST_MAX} 文字以下になるようにすること（URLの文字数は {len(url)} 文字）。"""


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
    if should_skip_no_new_article(url):
        # 重複時は Bluesky / Gemini を呼ばず正常終了（GitHub Actions を赤くしない）
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
