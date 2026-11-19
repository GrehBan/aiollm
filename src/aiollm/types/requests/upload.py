"""The shared request of a file upload."""

from __future__ import annotations

from typing import Annotated

from pydantic import MISSING, Field

from aiollm.types.files import InputFile
from aiollm.types.requests.base import BaseRequest


class FileUpload(BaseRequest):
    """A file to store with a provider, as ``client.files.upload`` takes it.

    Attributes:
        file: The content: an ``InputFile``, ``bytes``, a
            ``pathlib.Path`` or a binary stream. Its ``filename`` is the
            name the provider keeps; give ``bytes`` a name with
            ``BytesFile(data, "report.pdf")``.
        expires_in_seconds: Seconds after the upload when the provider
            deletes the file. Each provider has its own range (OpenAI
            3600 to 2 592 000, Anthropic 3600 to 7 776 000) and checks
            it before anything is sent; left out, the file is kept until
            it is deleted.
    """

    file: InputFile
    expires_in_seconds: Annotated[int, Field(gt=0)] | MISSING = MISSING
