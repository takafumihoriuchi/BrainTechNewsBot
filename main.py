"""
専門メディアの RSS から BCI / 神経科学ニュースを取得し、Gemini で日本語要約を生成、
Bluesky にリンクカード付きで投稿する。

環境変数（.env）:
  GEMINI_API_KEY, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  任意: GEMINI_MODEL（既定: gemini-2.5-flash → gemini-2.5-flash-lite）
  任意: DRY_RUN=1 で Bluesky への投稿をスキップし、生成テキストのみ表示

投稿済み記事 URL を last_post.txt に 1 行 1 URL で蓄え（直近 100 件を上限）、
新しい順の一覧の先頭から、履歴に無い最初の記事（未投稿のうち最新）を投稿する。
"""

from __future__ import annotations

import os
import re
import sys
import time
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

RSS_FEEDS: tuple[str, ...] = (
    "https://www.sciencedaily.com/rss/computers_math/brain-computer_interfaces.xml",
    "https://www.nature.com/subjects/brain-machine-interface.rss",
    "https://neurosciencenews.com/neuroscience-topics/bci/feed/",
    "https://news.mit.edu/topic/neuroscience-rss.xml",
)

POST_MAX = 300
DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
LAST_POST_FILE = Path(__file__).resolve().parent / "last_post.txt"
POSTED_URL_HISTORY_MAX = 100


def _default_http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


_REQUEST_HEADERS = _default_http_headers()
_MAX_IMAGE_BYTES = 900_000
_EMBED_TITLE_MAX = 300
_EMBED_DESC_MAX = 1000


def normalize_article_url(url: str) -> str:
    s = (url or "").strip()
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


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


def _looks_like_placeholder_og_image(image_url: str) -> bool:
    """汎用ロゴ・favicon っぽい URL はサムネに使わない。"""
    il = image_url.lower()
    markers = (
        "/favicon",
        "/logo",
        "/logos/",
        "/branding/",
        "/apple-touch-icon",
        "favicon.ico",
        "sprite",
        "placeholder",
    )
    return any(m in il for m in markers)


def _fetch_image_bytes(
    image_url: str, timeout: float = 20.0, *, referer: str | None = None
) -> bytes | None:
    if _looks_like_placeholder_og_image(image_url):
        return None
    try:
        session = requests.Session()
        h = {
            "User-Agent": _REQUEST_HEADERS["User-Agent"],
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        }
        if referer:
            h["Referer"] = referer
        session.headers.update(h)
        r = session.get(image_url, timeout=timeout, stream=True, allow_redirects=True)
        r.raise_for_status()
        if _looks_like_placeholder_og_image(str(r.url)):
            return None
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
    """記事 URL に GET し、BeautifulSoup で OGP を抽出する。"""
    try:
        session = requests.Session()
        session.headers.update(_REQUEST_HEADERS)
        r = session.get(article_url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = _meta_content(soup, "og:title", "twitter:title")
        desc = _meta_content(soup, "og:description", "twitter:description", "description")
        raw_img = _meta_content(soup, "og:image", "twitter:image", "twitter:image:src")
        if raw_img and _looks_like_placeholder_og_image(raw_img):
            raw_img = _meta_content(soup, "twitter:image", "twitter:image:src")
        if raw_img and _looks_like_placeholder_og_image(raw_img):
            raw_img = None
        image_bytes: bytes | None = None
        if raw_img:
            img_url = urljoin(article_url, raw_img.strip())
            if not _looks_like_placeholder_og_image(img_url):
                image_bytes = _fetch_image_bytes(img_url, timeout=timeout, referer=article_url)
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


def _entry_published_ts(entry: object) -> float:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if t:
        try:
            return float(time.mktime(t))
        except (OverflowError, TypeError, ValueError):
            pass
    return 0.0


def _entry_to_article(e: object) -> tuple[str, str, str] | None:
    title = strip_html(getattr(e, "title", "") or "")
    link = normalize_article_url((getattr(e, "link", "") or "").strip())
    raw = getattr(e, "summary", None) or getattr(e, "description", "") or ""
    content = strip_html(raw) or title
    if not title or not link:
        return None
    return title, content, link


def fetch_all_entries() -> list[tuple[str, str, str]]:
    """複数 RSS からエントリを集め、公開日時の新しい順に並べたリストを返す（同一 URL は新しい方のみ）。"""
    scored: list[tuple[float, object]] = []
    for feed_url in RSS_FEEDS:
        parsed = feedparser.parse(feed_url)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
            print(
                f"RSS 警告 ({feed_url}): {getattr(parsed, 'bozo_exception', '')}",
                file=sys.stderr,
            )
        for e in getattr(parsed, "entries", []) or []:
            scored.append((_entry_published_ts(e), e))

    if not scored:
        print("いずれの RSS からもエントリを取得できませんでした。", file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda x: x[0], reverse=True)

    seen_urls: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for _, e in scored:
        row = _entry_to_article(e)
        if row is None:
            continue
        _, _, link = row
        if link in seen_urls:
            continue
        seen_urls.add(link)
        out.append(row)
    if not out:
        print("有効なエントリを 1 件も組み立てられませんでした。", file=sys.stderr)
        sys.exit(1)
    return out


def load_posted_urls() -> set[str]:
    """last_post.txt の全行を読み、正規化した URL の集合を返す。"""
    if not LAST_POST_FILE.is_file():
        return set()
    try:
        raw = LAST_POST_FILE.read_text(encoding="utf-8")
    except OSError:
        return set()
    out: set[str] = set()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        out.add(normalize_article_url(s))
    return out


def _trim_posted_url_file() -> None:
    """追記で増えすぎた履歴を末尾 POSTED_URL_HISTORY_MAX 行に切り詰める。"""
    try:
        raw = LAST_POST_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    lines = [normalize_article_url(l.strip()) for l in raw.splitlines() if l.strip()]
    if len(lines) <= POSTED_URL_HISTORY_MAX:
        return
    kept = lines[-POSTED_URL_HISTORY_MAX:]
    try:
        LAST_POST_FILE.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except OSError:
        pass


def append_posted_url(url: str) -> None:
    """新しい投稿 URL を 1 行追記し、履歴が長すぎる場合は古い行から削除する。"""
    normalized = normalize_article_url(url)
    try:
        with LAST_POST_FILE.open("a", encoding="utf-8") as f:
            f.write(normalized + "\n")
    except OSError:
        return
    _trim_posted_url_file()


def body_char_budget() -> int:
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
    posted_urls = load_posted_urls()
    entries = fetch_all_entries()

    chosen: tuple[str, str, str] | None = None
    for title, content, article_url in entries:
        article_url = normalize_article_url(article_url)
        if article_url not in posted_urls:
            chosen = (title, content, article_url)
            break

    if chosen is None:
        print("No new news found.")
        raise SystemExit(0)

    title, content, article_url = chosen

    body = generate_post_text(title, content, article_url)
    body_text = finalize_post_body(body)
    preview = fetch_link_preview(article_url)
    post_to_bluesky(body_text, article_url, rss_title=title, preview=preview)
    if not _dry_run():
        append_posted_url(article_url)
    print(body_text)


if __name__ == "__main__":
    main()
