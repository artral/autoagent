import os
from pathlib import Path
from typing import Optional


class SpecLoader:
    """Loads the SDD spec from either inline text or an external file."""

    def __init__(
        self,
        spec_source: Optional[str] = None,
        *,
        default_path: Optional[str] = None,
    ) -> None:
        self.spec_source = spec_source
        self.default_path = default_path

    def load(self) -> str:
        candidate = (
            self.spec_source
            or self.default_path
            or os.getenv("SPEC_FILE")
            or os.getenv("SPEC_PATH")
        )

        if candidate is None or not str(candidate).strip():
            raise ValueError(
                "No spec provided. Pass a file path via --spec-file or inline text via --spec."
            )

        candidate = str(candidate).strip()

        if os.path.exists(candidate):
            return Path(candidate).read_text(encoding="utf-8")

        return candidate
