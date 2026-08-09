# Security & Production Notes

ML Atlas is a portfolio/educational platform. Before production use:

- Do not load untrusted pickle files.
- Place authentication and authorization in front of the inference API.
- Enforce payload and schema limits.
- Add request logging without storing sensitive feature data.
- Use durable experiment/model-registry storage.
- Scan container images and dependencies.
- Store credentials in managed secrets, never in Git.
- Add rate limiting and network controls.
- Validate fairness, privacy, retention, and regulatory obligations for the specific use case.
