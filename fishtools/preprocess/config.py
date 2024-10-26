from json import JSONEncoder
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, PlainSerializer


class NumpyEncoder(JSONEncoder):
    def default(self, o):  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class Fiducial(BaseModel):
    fwhm: float = Field(
        default=4.0,
        description="Full width at half maximum for fiducial spot detection. The higher this is, the more spots will be detected.",
    )
    threshold: float = Field(
        default=3.0,
        description="Threshold for fiducial spot detection in standard deviation above the median.",
    )
    priors: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Shifts to apply before alignment. Name must match round name.",
    )
    overrides: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Overrides for fiducial spot detection. Name must match round name.",
    )


class RegisterConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    fiducial: Fiducial
    downsample: int = Field(default=1, description="Downsample factor")
    reduce_bit_depth: int = Field(
        default=0,
        description="Reduce bit depth by n bits. 0 to disable. This is to assist in compression of output intended for visualization.",
    )
    crop: int = Field(
        default=30,
        description="Pixels to crop from each edge. This is to account for translation during alignment.",
    )
    slices: Annotated[list[tuple[int | None, int | None]] | slice, PlainSerializer(lambda x: str(x))] = Field(
        default=[slice(None)], description="Slice range to use for registration"
    )
    split_channels: bool = False
    chromatic_shifts: dict[str, Annotated[str, "path for 560to{channel}.txt"]]


class ChannelConfig(BaseModel):
    discards: dict[str, list[str]] = Field(
        description="In case of duplicated key(s), discard the ones from the file in value."
    )


class Config(BaseModel):
    dataPath: str
    registration: RegisterConfig
    channels: ChannelConfig | None = None
    basic: Literal["per_round", "all_round"] | None = Field(
        None,
        description="Perform BaSiC correction specific to each round or use the same template for all rounds.",
    )
