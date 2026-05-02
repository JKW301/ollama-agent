import re
import ssl
import urllib.request
import urllib.error
from config import MAX_OUTPUT_CHARS

_CTX = ssl.create_default_context()


def web_fetch(url: str, max_chars: int = 6000) -> str:
    """Télécharge le contenu texte d'une URL (HTML converti en texte brut)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ollama-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=_CTX) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")

            if "html" in content_type.lower():
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S | re.I)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [tronqué — {len(text)} chars total]"
            return text
    except urllib.error.HTTPError as e:
        return f"[ERR] HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"[ERR] URL: {e.reason}"
    except Exception as e:
        return f"[ERR] {e}"


def http_request(url: str, method: str = "GET", headers: dict = None, body: str = None) -> str:
    """Effectue une requête HTTP (GET, POST, PUT, DELETE…) avec headers et body optionnels."""
    try:
        data = body.encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers or {})
        req.add_header("User-Agent", "ollama-agent/1.0")
        if data and not (headers or {}).get("Content-Type"):
            req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=15, context=_CTX) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", errors="replace")
            if len(text) > MAX_OUTPUT_CHARS:
                text = text[:MAX_OUTPUT_CHARS] + "\n... [tronqué]"
            return f"HTTP {status}\n{text}"
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        return f"[ERR] HTTP {e.code}: {e.reason}\n{err_body}"
    except Exception as e:
        return f"[ERR] {e}"
