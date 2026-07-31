"""Sentinel agents.

Agents are intentionally not imported eagerly.  Several have optional runtime
dependencies; importing one agent must not initialize every provider client or
evaluation framework in the package.
"""
