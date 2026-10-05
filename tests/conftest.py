"""Shared fixtures. Everything runs on the offline reasoner with the cache off
so tests never hit the network.
"""

import dataclasses

import pytest

from deductive_mas.agents.orchestrator import DeductiveOrchestrator
from deductive_mas.config import Config, LLMConfig
from deductive_mas.knowledge.ontology import knowledge_graph
from deductive_mas.ui.ansi import Style


@pytest.fixture(scope="session")
def config() -> Config:
    return dataclasses.replace(Config(), llm=LLMConfig(backend="offline", cache_dir=None))


@pytest.fixture(scope="session")
def orchestrator(config: Config) -> DeductiveOrchestrator:
    return DeductiveOrchestrator(config)


@pytest.fixture(scope="session")
def graph():
    return knowledge_graph()


@pytest.fixture()
def plain_style() -> Style:
    style = Style("never")
    style.width = 92
    return style
