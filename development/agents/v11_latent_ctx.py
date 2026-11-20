"""L3: L2 with latent channels as context only (never targets of v11)."""
from agents.v11_latent import Agent as _Base


class Agent(_Base):
    SOFT = True
    CONTEXT_ONLY = True
