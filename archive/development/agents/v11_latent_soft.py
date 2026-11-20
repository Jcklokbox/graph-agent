"""L2: v11_latent with the prediction marginalised over the HMM belief."""
from agents.v11_latent import Agent as _Base


class Agent(_Base):
    SOFT = True
