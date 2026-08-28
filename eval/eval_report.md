# RAG Evaluation Report

**Summary:** 10/10 passed

This report was generated locally from golden questions.

## Results

| ID | Status | Question | Expected source | Actual refusal | Grounding |
| --- | --- | --- | --- | --- | --- |
| q001 | PASS | Can contractors access production systems? | access_policy.md | False | PASS |
| q002 | PASS | Do privileged users need MFA? | mfa_policy.md | False | PASS |
| q003 | PASS | What should employees do after a security incident? | incident_response.md | False | PASS |
| q004 | PASS | Can employees share passwords in chat? | password_policy.md | False | PASS |
| q005 | PASS | Is VPN required when using airport Wi-Fi? | remote_work_policy.md | False | PASS |
| q006 | PASS | Can customer data be shared with public AI tools? | privacy_policy.md | False | PASS |
| q007 | PASS | Can employees paste API keys into public AI tools? | ai_usage_policy.md | False | PASS |
| q008 | PASS | What is the CEO's favorite restaurant? | - | True | PASS |
| q009 | PASS | What is the employee salary increase plan for next year? | - | True | PASS |
| q010 | PASS | Which secret project will be launched next quarter? | - | True | PASS |

## Notes

- This evaluation uses synthetic policy documents only.
- The default local hash embedding provider is deterministic but not semantically strong.
- Failing cases should be inspected with the RAG Debugger before changing the pipeline.
