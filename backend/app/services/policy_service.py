"""Application service for context policy management."""

from app.core.policy_orchestrator import PolicyOrchestrator
from app.models import ContextPolicy, PolicyExecution, PolicyState


class PolicyService:
    """Service facade for policy CRUD and execution observation."""

    def __init__(self, orchestrator: PolicyOrchestrator | None = None) -> None:
        self._orchestrator = orchestrator or PolicyOrchestrator()

    def list_policies(self) -> list[ContextPolicy]:
        """Return all policies including the default."""
        return self._orchestrator.list_policies()

    def get_policy(self, policy_id: str) -> ContextPolicy | None:
        """Fetch a single policy by id."""
        return self._orchestrator.get_policy(policy_id)

    def create_policy(self, policy: ContextPolicy) -> tuple[ContextPolicy | None, list[str]]:
        """Validate and store a new policy."""
        errors = self._orchestrator.upsert_policy(policy)
        if errors:
            return None, errors
        return self._orchestrator.get_policy(policy.policy_id), []

    def update_policy(
        self, policy_id: str, policy: ContextPolicy
    ) -> tuple[ContextPolicy | None, list[str]]:
        """Replace an existing policy."""
        policy.policy_id = policy_id
        errors = self._orchestrator.upsert_policy(policy)
        if errors:
            return None, errors
        return self._orchestrator.get_policy(policy_id), []

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy; returns False if it is the default."""
        return self._orchestrator.delete_policy(policy_id)

    def set_default(self, policy_id: str) -> ContextPolicy | None:
        """Mark a policy as the default."""
        policy = self._orchestrator.get_policy(policy_id)
        if policy is None:
            return None
        policy.default = True
        errors = self._orchestrator.upsert_policy(policy)
        if errors:
            return None
        return self._orchestrator.get_policy(policy_id)

    def get_execution(self, session_id: str) -> PolicyExecution | None:
        """Return the latest policy execution for a session."""
        return self._orchestrator.get_execution(session_id)

    def get_state(self, session_id: str) -> PolicyState | None:
        """Return the observed policy state for a session."""
        return self._orchestrator.get_state(session_id)

    @property
    def orchestrator(self) -> PolicyOrchestrator:
        """Expose the underlying orchestrator for ChatOrchestrator integration."""
        return self._orchestrator
