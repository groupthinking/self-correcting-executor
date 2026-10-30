"""Source-faithful URL/video/PDF/Google Docs intake with real extraction.

HTTP origins and OAuth credentials belong to the embedding application. No
transcript/PDF placeholder is ever returned when a provider is unavailable.
"""

import asyncio
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
import re
from urllib.parse import urlsplit, parse_qs, quote

import requests

from .correction import invoke


def origin(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Source requires an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Source URL cannot contain credentials or fragment")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


class HttpSource:
    """Bounded real HTTP transport, restricted to host-approved origins.

    Configure origins in trusted application code, not from the source document.
    Redirects are rejected so a source cannot forward OAuth tokens elsewhere.
    """

    def __init__(self, allowed_origins, *, session=None, max_bytes=25_000_000):
        self.allowed_origins = {origin(item) for item in allowed_origins}
        self.session = session or requests.Session()
        self.max_bytes = max_bytes

    def get(self, url, headers=None):
        if origin(url) not in self.allowed_origins:
            raise ValueError("Source origin has not been authorized by the host")
        with self.session.get(
            url, headers=headers or {}, timeout=20, allow_redirects=False, stream=True
        ) as response:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Source HTTP {response.status_code}")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError("Source exceeds configured byte limit")
                chunks.append(chunk)
            return b"".join(chunks), dict(response.headers)


def content_type(headers):
    return next(
        (
            str(value).lower()
            for key, value in headers.items()
            if key.lower() == "content-type"
        ),
        "",
    )


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def classify_source(url):
    origin(url)
    parsed = urlsplit(url)
    host = parsed.hostname.lower()
    if host in {
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "vimeo.com",
        "www.vimeo.com",
        "dailymotion.com",
        "www.dailymotion.com",
    }:
        return "video"
    if parsed.path.lower().endswith(".pdf"):
        return "pdf"
    if host in {"docs.google.com", "drive.google.com"}:
        return "doc"
    return "url"


def youtube_id(url):
    parsed = urlsplit(url)
    host = parsed.hostname
    value = None
    if host == "youtu.be":
        value = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com"}:
        value = parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/embed/"):
            value = parsed.path.split("/")[2]
    return value if value and re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else None


class GoogleCaptions:
    """Downloads actual caption tracks using the caller's authorized OAuth token."""

    def __init__(self, transport, token):
        self.transport, self.token = transport, token

    async def __call__(self, url, language=None):
        video_id = youtube_id(url)
        if not video_id or not self.token:
            raise ValueError("YouTube caption access requires video ID and OAuth token")
        headers = {"Authorization": f"Bearer {self.token}"}
        data, _ = await asyncio.to_thread(
            self.transport.get,
            f"https://www.googleapis.com/youtube/v3/captions?videoId={video_id}&part=snippet",
            headers,
        )
        tracks = json.loads(data).get("items", [])
        selected = next(
            (t for t in tracks if not language or t["snippet"]["language"] == language),
            None,
        )
        if selected is None:
            raise RuntimeError("No authorized caption track for requested language")
        caption_id = selected["id"]
        raw, _ = await asyncio.to_thread(
            self.transport.get,
            f"https://www.googleapis.com/youtube/v3/captions/{quote(caption_id, safe='')}?tfmt=srt",
            headers,
        )
        return {
            "text": raw.decode("utf-8"),
            "metadata": {
                "service": "youtube-api",
                "videoId": video_id,
                "captionId": caption_id,
                "language": selected["snippet"]["language"],
            },
        }


class Ingestor:
    def __init__(self, transport, *, transcriber=None, google_token=None):
        self.transport = transport
        self.transcriber = transcriber
        self.google_token = google_token

    async def ingest(self, url, *, source_type=None, language=None):
        kind = source_type or classify_source(url)
        if kind not in {"url", "video", "pdf", "doc"}:
            raise ValueError("Unknown source type")
        metadata = {"source_url": url, "source_type": kind}
        if kind == "video":
            if self.transcriber is None:
                raise ValueError(
                    "Video ingestion requires a configured transcription adapter"
                )
            result = await invoke(self.transcriber, url, language)
            text = result["text"]
            metadata.update(result.get("metadata", {}))
        elif kind == "doc":
            parsed = urlsplit(url)
            if parsed.hostname not in {"docs.google.com", "drive.google.com"}:
                raise ValueError("Document source must be a Google Docs URL")
            match = re.search(r"/(?:document|file)/d/([A-Za-z0-9_-]+)", parsed.path)
            if not match:
                raise ValueError("Missing Google document ID")
            doc_id = match.group(1)
            headers = {}
            if self.google_token:
                target = f"https://www.googleapis.com/drive/v3/files/{doc_id}/export?mimeType=text%2Fplain"
                headers["Authorization"] = f"Bearer {self.google_token}"
            else:
                target = (
                    f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                )
            raw, response_headers = await asyncio.to_thread(
                self.transport.get, target, headers
            )
            if "text/html" in content_type(response_headers):
                raise ValueError("Google Docs returned HTML instead of exported text")
            text = raw.decode("utf-8")
            metadata.update(docId=doc_id, service="google-docs-export")
        else:
            raw, headers = await asyncio.to_thread(self.transport.get, url)
            metadata["download_sha256"] = sha256(raw).hexdigest()
            if kind == "pdf":
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(raw))
                if reader.is_encrypted:
                    raise ValueError(
                        "Encrypted PDF requires an explicit decryption workflow"
                    )
                if len(reader.pages) > 500:
                    raise ValueError("PDF exceeds page limit")
                parts = []
                for page in reader.pages:
                    stream = page.get_contents()
                    if stream and len(stream.get_data()) > 20_000_000:
                        raise ValueError("PDF page content exceeds extraction limit")
                    parts.append(page.extract_text())
                text = "\n".join(parts)
                metadata.update(pageCount=len(reader.pages), service="pypdf")
            else:
                text = raw.decode("utf-8")
                if "text/html" in content_type(headers):
                    parser = TextHTML()
                    parser.feed(text)
                    text = " ".join(" ".join(parser.text).split())
                metadata["contentType"] = content_type(headers)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Source extraction returned no text (OCR is not implied)")
        if len(text) > 500_000:
            raise ValueError("Extracted source exceeds text limit")
        return {
            "content": text,
            "sourceType": kind,
            "metadata": {
                **metadata,
                "length": len(text),
                "content_sha256": sha256(text.encode()).hexdigest(),
            },
        }


class OpenAIMediaTranscriber:
    """Real ASR for an explicitly approved direct audio/video URL.

    Web pages such as Vimeo require the host to supply a lawful media extractor;
    HTML is never submitted as audio. YouTube can use GoogleCaptions instead.
    """

    def __init__(self, transport, connector, *, model="whisper-1"):
        self.transport, self.connector, self.model = transport, connector, model

    async def __call__(self, url, language=None):
        import aiohttp

        if not self.connector.openai_key:
            raise ValueError("OPENAI_API_KEY is required for media transcription")
        raw, headers = await asyncio.to_thread(self.transport.get, url)
        mime = content_type(headers).split(";", 1)[0]
        if not mime.startswith(("audio/", "video/")):
            raise ValueError("Transcription requires a direct audio/video resource")
        if not raw or len(raw) > 25_000_000:
            raise ValueError("Transcription media must be between 1 and 25000000 bytes")
        form = aiohttp.FormData()
        filename = urlsplit(url).path.rsplit("/", 1)[-1] or "media.mp4"
        form.add_field("file", raw, filename=filename, content_type=mime)
        form.add_field("model", self.model)
        if language:
            form.add_field("language", language)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            async with session.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.connector.openai_key}"},
                data=form,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Transcription provider HTTP {response.status}")
                value = await response.json()
        if not isinstance(value.get("text"), str) or not value["text"].strip():
            raise ValueError("Transcription provider returned no text")
        return {
            "text": value["text"],
            "metadata": {
                "service": "openai-transcription",
                "model": self.model,
                "media_sha256": sha256(raw).hexdigest(),
                "source_url": url,
            },
        }
