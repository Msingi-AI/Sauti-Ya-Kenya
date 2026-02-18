import os
from typing import Iterable, Optional, Set


DEFAULT_TRUSTED_REMOTE_IDS: Set[str] = {
    "fishaudio/fish-speech-1.5",
    "FunAudioLLM/CosyVoice2-0.5B",
    "google/WaxalNLP",
}


def _parse_allowlist(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def is_remote_code_trusted(resource_id: str, env: Optional[dict] = None, allowlist: Optional[Iterable[str]] = None) -> bool:
    """Resolve whether remote code is allowed for a model/dataset ID.

    Precedence:
      1. SAUTI_TRUST_REMOTE_CODE=1/true enables globally.
      2. SAUTI_TRUST_REMOTE_CODE=0/false disables globally.
      3. SAUTI_REMOTE_CODE_ALLOWLIST and explicit allowlist are checked.
      4. Fall back to DEFAULT_TRUSTED_REMOTE_IDS.
    """
    env_map = env or os.environ
    raw = str(env_map.get("SAUTI_TRUST_REMOTE_CODE", "")).strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False

    combined = set(DEFAULT_TRUSTED_REMOTE_IDS)
    combined.update(_parse_allowlist(env_map.get("SAUTI_REMOTE_CODE_ALLOWLIST")))
    if allowlist:
        combined.update(allowlist)

    return resource_id in combined
