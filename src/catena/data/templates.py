from __future__ import annotations

DOMAIN_FIELDS: dict[str, dict[str, list[str]]] = {
    "api": {
        "auth_method": ["API_KEY", "OAUTH2", "SIGNED_JWT", "MTLS"],
        "region": ["EU", "US", "APAC"],
        "endpoint": ["v1/payments", "v2/payments", "v3/charges"],
        "retry_policy": ["fixed-3", "exponential-5", "no-retry"],
        "timeout_ms": ["1500", "3000", "5000"],
    },
    "access": {
        "role": ["viewer", "editor", "admin", "auditor"],
        "resource": ["billing", "reports", "secrets", "deployments"],
        "approval": ["none", "manager", "security"],
        "region": ["EU", "US", "APAC"],
        "session_ttl": ["15m", "1h", "8h"],
    },
    "workflow": {
        "owner": ["alice", "bob", "carol", "dave"],
        "channel": ["email", "slack", "pager"],
        "cadence": ["daily", "weekly", "monthly"],
        "priority": ["low", "normal", "urgent"],
        "approval": ["none", "team-lead", "director"],
    },
}

SCHEMA_FAMILIES: dict[str, list[str]] = {
    "api": ["payment-client", "analytics-client", "notification-client", "inventory-client"],
    "access": ["finance-console", "data-warehouse", "release-system", "support-portal"],
    "workflow": ["incident-review", "invoice-approval", "weekly-report", "customer-escalation"],
}

FILLER_SENTENCES = [
    "The operator recorded a routine status check and found no exception.",
    "A previous dry run completed without changing the canonical configuration.",
    "The monitoring note refers to an unrelated component and remains valid.",
    "A user requested a summary, but no policy field was modified.",
    "The audit log contains a successful action under the then-current rule.",
    "The task owner confirmed the existing plan before the later update arrived.",
]
