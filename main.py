"""
グローバル（英語）の BrainTech / BCI / 神経科学ニュースを Google News RSS から取得し、
Gemini で日本語要約を生成し、Bluesky にリンクカード付きで投稿する。

環境変数（.env）:
  GEMINI_API_KEY, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  任意: GEMINI_MODEL（既定: gemini-2.5-flash → gemini-2.5-flash-lite）
  任意: DRY_RUN=1 で Bluesky への投稿をスキップし、生成テキストのみ表示

直近投稿した記事 URL（リダイレクト解決後）を last_post.txt に保存する。
取得 URL が一致する場合は投稿・Gemini を一切行わず終了する。
"""

from __future__ import annotations

import os
import re
import sys
from html import unescape
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import feedparser
import requests
from atproto import Client, models
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

# 英語圏のグローバルニュース（記事本文の要約は Gemini で日本語に変換）
RSS_URL = (
    "https://news.google.com/rss/search?"
    "q=Brain-Computer+Interface+OR+BCI+OR+BrainTech+OR+Neuroscience"
    "&hl=en&gl=US&ceid=US:en"
)

# Bluesky の投稿テキストは 300 文字まで（本文のみ。URL はリンクカード embed に載せる）
POST_MAX = 300
DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
LAST_POST_FILE = Path(__file__).resolve().parent / "last_post.txt"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BrainTechNewsBot/1.0; "
        "+https://github.com/takafumihoriuchi/BrainTechNewsBot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
}
_MAX_IMAGE_BYTES = 900_000
_EMBED_TITLE_MAX = 300
_EMBED_DESC_MAX = 1000


def normalize_article_url(url: str) -> str:
    """比較・保存用に URL を揃える（末尾スラッシュや前後空白の差で二重投稿しない）。"""
    s = (url or "").strip()
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def resolve_final_url(url: str, timeout: float = 25.0) -> str:
    """HTTP リダイレクトを追い、最終的な記事 URL を返す（Google News の超長 URL を短縮）。"""
    u = normalize_article_url(url)
    if not u:
        return u
    try:
        session = requests.Session()
        session.headers.update(_REQUEST_HEADERS)
        r = session.get(u, allow_redirects=True, timeout=timeout)
        return normalize_article_url(r.url) or u
    except requests.RequestException:
        return u


class LinkPreviewData(NamedTuple):
    title: str | None
    description: str | None
    image_bytes: bytes | None


def _meta_content(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        tag = soup.find("meta", property=key) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            s = str(tag["content"]).strip()
            if s:
                return s
    return None


def _fetch_image_bytes(image_url: str, timeout: float = 20.0) -> bytes | None:
    try:
        session = requests.Session()
        session.headers.update(_REQUEST_HEADERS)
        r = session.get(image_url, timeout=timeout, stream=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/"):
            return None
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=65536):
            buf.extend(chunk)
            if len(buf) > _MAX_IMAGE_BYTES:
                return None
        data = bytes(buf)
        if len(data) < 100:
            return None
        return data
    except (requests.RequestException, OSError):
        return None


def fetch_link_preview(article_url: str, timeout: float = 20.0) -> LinkPreviewData:
    """記事ページから OGP を取得し、サムネイル画像バイト列を可能なら返す。"""
    try:
        session = requests.Session()
        session.headers.update(_REQUEST_HEADERS)
        r = session.get(article_url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = _meta_content(soup, "og:title", "twitter:title")
        desc = _meta_content(soup, "og:description", "twitter:description", "description")
        raw_img = _meta_content(soup, "og:image", "twitter:image", "twitter:image:src")
        image_bytes: bytes | None = None
        if raw_img:
            img_url = urljoin(article_url, raw_img.strip())
            image_bytes = _fetch_image_bytes(img_url, timeout=timeout)
        return LinkPreviewData(title=title, description=desc, image_bytes=image_bytes)
    except (requests.RequestException, OSError):
        return LinkPreviewData(title=None, description=None, image_bytes=None)


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


def body_char_budget() -> int:
    """投稿テキスト（日本語本文のみ）の上限。URL はリンクカード側のため本文には含めない。"""
    return POST_MAX


def build_prompt(title: str, content: str, article_url: str) -> str:
    budget = body_char_budget()
    return f"""あなたはブレインテック（BCI・脳コンピュータインタフェース・神経科学など）に詳しい編集者です。
以下のニュースは英語ソースからの情報です。内容を踏まえ、Bluesky 向けの「投稿本文」だけを**日本語で**書いてください。

【記事タイトル（英語のまま引用可）】
{title}

【記事の概要・内容（英語など原文のまま）】
{content}

【参照用の記事URL（本文には書かない。別途リンクカードとして添付する）】
{article_url}

【最優先のトーン・内容（すべて日本語で）】
・記事を理解するうえで必要な前提知識を、必ず日本語で補足すること（短い一文や括弧書きでよい）。
・専門用語は日本語で噛み砕き説明し、要点を絞ってわかりやすくまとめること。
・英語のまま出力してはいけない（固有名詞・製品名・論文名など必要最小限の英単語のみ可）。

【形式】
・出力は「日本語の投稿本文のみ」。引用符・箇条書き・「投稿:」などのラベルは付けないこと。
・記事URLやハイパーリンクは本文に含めない（リンクはシステム側でカード表示する）。
・本文の文字数上限: {budget} 文字（日本語は1文字で1と数える）。改行は使わないこと。"""


def _response_text(response: object) -> str:
    try:
        t = getattr(response, "text", None)
        return (t or "").strip() if t is not None else ""
    except (ValueError, AttributeError):
        return ""


def generate_post_text(title: str, content: str, article_url: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL")
    candidates = [model_name] if model_name else list(DEFAULT_MODELS)

    prompt = build_prompt(title, content, article_url)
    last_error: Exception | None = None
    cap = body_char_budget()
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


def finalize_post_body(body: str) -> str:
    """日本語本文のみ。リンクカードの URL は含めない。POST_MAX 文字以内。"""
    body = re.sub(r"\s+", " ", body.strip())
    if len(body) <= POST_MAX:
        return body
    return body[:POST_MAX].rstrip()


def _dry_run() -> bool:
    v = (os.getenv("DRY_RUN") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def post_to_bluesky(
    body_text: str,
    article_url: str,
    *,
    rss_title: str,
    preview: LinkPreviewData,
) -> None:
    if _dry_run():
        print("[DRY_RUN] Bluesky への投稿はスキップしました。", file=sys.stderr)
        return
    client = Client()
    client.login(
        os.environ["BLUESKY_HANDLE"],
        os.environ["BLUESKY_APP_PASSWORD"],
    )

    card_title = (preview.title or rss_title or "Article").strip()
    card_title = card_title[:_EMBED_TITLE_MAX]
    card_desc = (preview.description or body_text or " ").strip()
    if not card_desc:
        card_desc = " "
    card_desc = card_desc[:_EMBED_DESC_MAX]

    thumb_ref = None
    if preview.image_bytes:
        try:
            thumb_ref = client.upload_blob(preview.image_bytes).blob
        except Exception:
            thumb_ref = None

    embed = models.AppBskyEmbedExternal.Main(
        external=models.AppBskyEmbedExternal.External(
            uri=article_url,
            title=card_title,
            description=card_desc,
            thumb=thumb_ref,
        )
    )

    client.send_post(body_text, embed=embed, langs=["ja"])


def main() -> None:
    load_env()
    title, content, rss_link = fetch_latest_entry()
    article_url = resolve_final_url(rss_link)

    last_key = read_last_post_url()
    if last_key is not None and article_url == last_key:
        print("No new news found. Skipping post.")
        raise SystemExit(0)

    body = generate_post_text(title, content, article_url)
    body_text = finalize_post_body(body)
    preview = fetch_link_preview(article_url)
    post_to_bluesky(body_text, article_url, rss_title=title, preview=preview)
    if not _dry_run():
        write_last_post_url(article_url)
    print(body_text)


if __name__ == "__main__":
    main()
