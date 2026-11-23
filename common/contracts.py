"""Explicit, versioned contracts. No ML libraries are imported here."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DIMENSIONS = (
    "solvability",
    "linguistic_clarity",
    "contextual_coherence",
    "pedagogical_appropriateness",
    "curriculum_alignment",
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class Request(Contract):
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    curso: str = Field(min_length=1)
    codigo_oa: str = Field(min_length=1)
    texto_oa: str = Field(min_length=1)
    habilidad_evaluada: str = Field(min_length=1)
    condiciones_curriculares: str = Field(min_length=1)
    contexto: str = Field(min_length=1)
    solicitud: str = Field(min_length=1)


class BenchmarkV2Request(Request):
    id: str = Field(alias="id_solicitud", min_length=1)
    version: str = Field(pattern=r"^2\.\d+(?:-[a-z]+)?$")
    eje: Literal[
        "Números y operaciones",
        "Patrones y álgebra",
        "Geometría",
        "Medición",
        "Datos y probabilidades",
    ]
    variante: int = Field(ge=1, le=5)
    tipo_representacion: Literal["Texto", "Figura especificada", "Tabla o registro textual"]
    especificacion_representacion: str = Field(min_length=1)
    estado_revision: str
    observaciones_revision: str
    cobertura_banco: str

    @field_validator("curso")
    @classmethod
    def normalize_course(cls, value):
        match = re.fullmatch(r"([1-6])° básico", value)
        if match is None:
            raise ValueError("Curso del benchmark v2 no reconocido")
        return match[1]


class Problem(Contract):
    enunciado: str = Field(min_length=1)
    solucion: str = Field(min_length=1)


class GeneratedProblem(Contract):
    enunciado: str = Field(min_length=1)
    solucion: str | None = None
    descripcion_figura: str | None = None


class Example(Request, Problem):
    procedencia: str = Field(min_length=1)
    estado: Literal["aceptado"]


class Model(Contract):
    id: str | None = None
    path: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class Experiment(Contract):
    model: Model
    benchmark: str
    phase: Literal["benchmark", "development", "pilot", "benchmark_candidate"] = "benchmark"
    pilot_manifest: str | None = None
    request_limit: int | None = Field(default=None, ge=1)
    development: str = "data/development.jsonl"
    oa_registry: str = "data/oa.json"
    skill_review: str = "configs/skill-review.json"
    judge_review: str = "configs/judge-review.json"
    port: int = Field(default=8000, ge=1024, le=65535)
    ready_timeout_s: int = Field(default=600, gt=0)
    load_conditions: str = ""
    container_image: str = ""
    train: str = "data/train.jsonl"
    validation: str = "data/validation.jsonl"
    demonstrations: str = "data/demonstrations.json"
    skill: str = "skill/SKILL.md"
    output: str = "output"
    seed: int = 42
    gpu_memory_utilization: float = Field(default=0.85, gt=0, lt=1)
    max_model_len: int = Field(default=4096, gt=0)
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.6, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetitions: int = Field(default=3, ge=1)
    chat_template_kwargs: dict = Field(default_factory=dict)
    adapter: str | None = None
    adapter_sha256: str | None = None
    learning_rate: float = Field(default=0.0001, gt=0)

    @model_validator(mode="after")
    def check_limits(self):
        if self.max_tokens >= self.max_model_len:
            raise ValueError("max_tokens debe dejar espacio para la entrada")
        return self

    target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])


class Criterion(Contract):
    id: str = Field(pattern=r"^[a-z][a-z_]*$")
    descriptor: str = Field(min_length=1)


class Rubric(Contract):
    version: str
    status: Literal["draft", "approved"]
    criteria: list[Criterion] = Field(min_length=5, max_length=5)
    solution_descriptor: str
    request_descriptor: str
