package agent_reflex

# Return no result for decisions that are not hard-policy concerns.
default decision := null

decision := {"value": "deny", "reason": "recursive root deletion is forbidden"} if {
    contains(sprintf("%v", [input.state]), "rm -rf /")
}

decision := {"value": "review", "reason": "force push requires review"} if {
    contains(sprintf("%v", [input.state]), "git push")
    contains(sprintf("%v", [input.state]), "--force")
}

decision := {"value": "deny", "reason": "credential exfiltration pattern"} if {
    contains(lower(sprintf("%v", [input.state])), ".env")
    contains(lower(sprintf("%v", [input.state])), "http")
}
