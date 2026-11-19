"""Resources and use cases that adapt provider methods to shared types."""

from __future__ import annotations

from aiollm.resources.base import (
    BaseResource,
    BaseSessionAccess,
)
from aiollm.resources.files import (
    FilesResource,
)
from aiollm.resources.generation import (
    GenerationResource,
)
from aiollm.resources.protocols import (
    ICompositeUseCase,
    IFilesAdapter,
    IFilesClient,
    IFilesResource,
    IGenerationAdapter,
    IGenerationClient,
    IGenerationResource,
    IResource,
    ISessionAccess,
    IStreamUseCase,
    IUseCase,
)
from aiollm.resources.stream import (
    GenerationStream,
)
from aiollm.resources.usecases import (
    AdapterCompositeUseCase,
    AdapterStreamUseCase,
    AdapterUseCase,
    BaseCompositeUseCase,
    BaseStreamUseCase,
    BaseUseCase,
)

__all__ = [
    "AdapterCompositeUseCase",
    "AdapterStreamUseCase",
    "AdapterUseCase",
    "BaseCompositeUseCase",
    "BaseResource",
    "BaseSessionAccess",
    "BaseStreamUseCase",
    "BaseUseCase",
    "FilesResource",
    "GenerationResource",
    "GenerationStream",
    "ICompositeUseCase",
    "IFilesAdapter",
    "IFilesClient",
    "IFilesResource",
    "IGenerationAdapter",
    "IGenerationClient",
    "IGenerationResource",
    "IResource",
    "ISessionAccess",
    "IStreamUseCase",
    "IUseCase",
]
