from pydantic import BaseModel, Field


class DeconvolutionConfig(BaseModel):
    psf: str = Field(description="Path to the PSF file.")


class RegisterConfig(BaseModel):
    class Fiducial(BaseModel):
        fwhm: float = Field(
            4.0,
            description="Full width at half maximum for fiducial spot detection. The higher this is, the more spots will be detected.",
        )
        threshold: float = Field(
            3.0, description="Threshold for fiducial spot detection in standard deviation above the median."
        )

    fiducial: Fiducial
    downsample: int = Field(1, description="Downsample factor")
    reduce_bit_depth: int = Field(
        0,
        description="Reduce bit depth by n bits. 0 to disable. This is to assist in compression of output intended for visualization.",
    )
    crop: int = Field(
        30, description="Pixels to crop from each edge. This is to account for translation during alignment."
    )
    max_proj: bool = True
    split_channels: bool = False


class Config(BaseModel):
    dataPath: str
    deconvolution: DeconvolutionConfig
    registration: RegisterConfig


class Experiment(BaseModel):
    name: str
    bit_mapping: dict[str, str]
