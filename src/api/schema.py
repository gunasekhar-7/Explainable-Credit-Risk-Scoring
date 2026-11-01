"""Pydantic request/response schemas for the credit risk API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreditApplication(BaseModel):
    """Incoming applicant record. Fields mirror the model input schema.

    Every field is optional: an unprovided field is imputed to a neutral WoE
    contribution, and the response's ``data_quality`` block reports exactly how
    many were supplied. Extra fields are ignored. Note that only the fields
    listed by ``/model-info`` can change the score.
    """

    SK_ID_CURR: int | None = Field(default=None, description="Applicant ID")
    NAME_CONTRACT_TYPE: str | None = Field(default=None, description="Contract type")
    CODE_GENDER: str | None = Field(default=None, description="Gender")
    FLAG_OWN_CAR: int | None = Field(default=None, description="Owns a car (0/1)")
    FLAG_OWN_REALTY: int | None = Field(default=None, description="Owns realty (0/1)")
    CNT_CHILDREN: float | None = Field(default=None, description="Number of children")
    AMT_INCOME_TOTAL: float | None = Field(default=None, description="Total annual income")
    AMT_CREDIT: float | None = Field(default=None, description="Credit amount")
    AMT_ANNUITY: float | None = Field(default=None, description="Annuity amount")
    DAYS_BIRTH: float | None = Field(default=None, description="Birth date (days, negative)")
    DAYS_EMPLOYED: float | None = Field(default=None, description="Employment duration (days, negative)")
    NAME_INCOME_TYPE: str | None = Field(default=None, description="Income type category")
    NAME_EDUCATION_TYPE: str | None = Field(default=None, description="Education category")
    NAME_FAMILY_STATUS: str | None = Field(default=None, description="Family status category")
    NAME_HOUSING_TYPE: str | None = Field(default=None, description="Housing category")
    OCCUPATION_TYPE: str | None = Field(default=None, description="Occupation category")
    CNT_FAM_MEMBERS: float | None = Field(default=None, description="Family members")
    EXT_SOURCE_1: float | None = Field(default=None, description="External source 1")
    EXT_SOURCE_2: float | None = Field(default=None, description="External source 2")
    EXT_SOURCE_3: float | None = Field(default=None, description="External source 3")

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "summary": "Partial applicant",
                    "description": "Scores, but reports 15 of 19 fields as imputed.",
                    "value": {"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 200000},
                },
                {
                    "summary": "Applicant with the fields that move the score",
                    "description": (
                        "EXT_SOURCE_1, OCCUPATION_TYPE, NAME_INCOME_TYPE and the "
                        "CNT_CHILDREN/CNT_FAM_MEMBERS ratio are the champion's inputs."
                    ),
                    "value": {
                        "SK_ID_CURR": 100002,
                        "EXT_SOURCE_1": 0.53,
                        "OCCUPATION_TYPE": "Managers",
                        "NAME_INCOME_TYPE": "Working",
                        "CNT_CHILDREN": 1,
                        "CNT_FAM_MEMBERS": 3,
                        "AMT_INCOME_TOTAL": 240000,
                        "AMT_CREDIT": 600000,
                    },
                },
            ]
        },
    )

    @field_validator(
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
    )
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("must be non-negative")  # noqa: TRY003
        return v

    @field_validator("FLAG_OWN_CAR", "FLAG_OWN_REALTY")
    @classmethod
    def _binary_flag(cls, v: int | None) -> int | None:
        if v is not None and v not in (0, 1):
            raise ValueError("must be 0 or 1")  # noqa: TRY003
        return v


class Contributor(BaseModel):
    """One reason the applicant scored as they did.

    `bin_label` is the modelled bin the applicant fell into (a WoE interval for
    the scorecard, the raw or imputed input value for the tree model) so the
    reason can be shown to a person, not just to a data scientist.
    """

    feature: str
    direction: str  # "increases_risk" | "decreases_risk"
    magnitude: float
    bin_label: str | None = None
    shap_value: float | None = None


class DataQuality(BaseModel):
    """How much of the applicant record was actually supplied.

    Scores on partial records are legitimate, but a consumer must be able to see
    that the answer rests on imputed inputs before acting on it.
    """

    fields_total: int = Field(description="Model input fields expected")
    fields_provided: int = Field(description="Fields supplied with a value")
    fields_imputed: int = Field(description="Fields defaulted by the model")
    imputed_features: list[str] = Field(default=[], description="Which fields were defaulted")
    complete: bool = Field(description="True when every model input was supplied")


class PredictionResponse(BaseModel):
    SK_ID_CURR: int | None = None
    champion: str | None = None
    model_version: str | None = None
    default_probability: float = Field(ge=0.0, le=1.0)
    scorecard_points: float | None = None
    risk_band: str | None = None
    top_contributors: list[Contributor] = []
    data_quality: DataQuality | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    """What is scoring, how well it does, and which inputs it can actually use."""

    model_type: str
    model_version: str | None = None
    features: list[str] = []
    metrics: dict = {}
    challenger: str | None = None
    challenger_metrics: dict = {}
    influencing_fields: list[str] = []
    ignored_fields: list[str] = []
    auth_mode: str = "local-open"


class ServiceInfo(BaseModel):
    """Service index served at ``/``."""

    service: str
    version: str
    docs_url: str
    champion: str | None = None
    auth_mode: str
    auth_header: str
    rate_limits: dict = {}
    endpoints: dict[str, str] = {}
