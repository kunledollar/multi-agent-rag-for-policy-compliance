"""Sentinel R6 architecture-level ablation configurations."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from .dispatcher import ExecutionDispatcher
from .models import ExecutionMode, ModeExecution


class AblationId(str, Enum):
    A0="A0"; A1="A1"; A2="A2"; A3="A3"; A4="A4"; A5="A5"; A6="A6"; A7="A7"


@dataclass(frozen=True)
class AblationConfiguration:
    configuration_id: AblationId
    configuration_name: str
    execution_mode: ExecutionMode
    enable_critic: bool = True
    enable_verification: bool = True
    enable_policy: bool = True
    enable_revision_loop: bool = True
    architecture: str = "r6"

    @property
    def disabled_components(self):
        flags={"critic":self.enable_critic,"verification":self.enable_verification,
               "policy":self.enable_policy,"revision_loop":self.enable_revision_loop}
        return tuple(k for k,v in flags.items() if not v)

    @property
    def enabled_components(self):
        return tuple(x for x in ("critic","verification","policy","revision_loop") if x not in self.disabled_components)


CONFIGURATIONS: Dict[AblationId,AblationConfiguration] = {
 A0:AblationConfiguration(A0,"full_sentinel_r6",ExecutionMode.FULL_SENTINEL),
 A1:AblationConfiguration(A1,"no_critic_agent",ExecutionMode.FULL_SENTINEL,enable_critic=False),
 A2:AblationConfiguration(A2,"no_verification_agent",ExecutionMode.FULL_SENTINEL,enable_verification=False),
 A3:AblationConfiguration(A3,"no_policy_agent",ExecutionMode.FULL_SENTINEL,enable_policy=False),
 A4:AblationConfiguration(A4,"no_revision_loop",ExecutionMode.FULL_SENTINEL,enable_revision_loop=False),
 A5:AblationConfiguration(A5,"sequential_sentinel_r5",ExecutionMode.FULL_SENTINEL,architecture="r5"),
 A6:AblationConfiguration(A6,"single_stage_rag",ExecutionMode.RAG_ONLY,False,False,False,False,"single_stage"),
 A7:AblationConfiguration(A7,"llm_only",ExecutionMode.LLM_ONLY,False,False,False,False,"llm_only"),
}


def get_configuration(value):
    try: return CONFIGURATIONS[value if isinstance(value,AblationId) else AblationId(str(value).upper())]
    except ValueError as exc: raise ValueError(f"Unknown configuration {value!r}; expected A0 through A7") from exc


class AblationDispatcher:
    def __init__(self, production=None, **_): self.production=production or ExecutionDispatcher()
    def execute(self, question, configuration, **kwargs) -> ModeExecution:
        # Dispatcher injection keeps experiment runners deterministic; production graph
        # consumes architecture flags when an ablation-capable callable is supplied.
        if configuration.configuration_id == AblationId.A5:
            from app.rag.sequential_graph import run_sentinel_graph as run_r5_graph
            output=ExecutionDispatcher(full=run_r5_graph).execute(
                question, ExecutionMode.FULL_SENTINEL, **kwargs)
        elif configuration.configuration_id in {AblationId.A1, AblationId.A2, AblationId.A3, AblationId.A4}:
            from app.rag.multi_agent_graph import run_sentinel_graph
            disabled = set(configuration.disabled_components) - {"revision_loop"}
            max_iterations = 0 if not configuration.enable_revision_loop else 2
            dispatcher = ExecutionDispatcher(full=lambda question, top_k=5: run_sentinel_graph(
                question, top_k=top_k, max_iterations=max_iterations, disabled_agents=disabled))
            output = dispatcher.execute(question, ExecutionMode.FULL_SENTINEL, **kwargs)
        else:
            output=self.production.execute(question, configuration.execution_mode, **kwargs)
        output.audit.update({"configuration_id":configuration.configuration_id.value,
          "configuration_name":configuration.configuration_name,"architecture":configuration.architecture,
          "enabled_components":list(configuration.enabled_components),"disabled_components":list(configuration.disabled_components)})
        return output
