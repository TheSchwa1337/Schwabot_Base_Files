from pydantic import BaseModel, conint, confloat

class QuantizationSchema(BaseModel):
    """Pydantic schema used to validate the quantization profile section of YAML configs."""

    decay_power: float
    terms: conint(gt=0)
    dimension: conint(gt=0)
    epsilon_q: confloat(gt=0, lt=1)
    precision: float