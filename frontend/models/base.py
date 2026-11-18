"""Shared wire types; coordinates are longitude, latitude."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TripID = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{1,80}$")]
NonNegative = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Coordinate = tuple[Annotated[float, Field(ge=-180, le=180)], Annotated[float, Field(ge=-90, le=90)]]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
