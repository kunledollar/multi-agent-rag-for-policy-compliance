# Governance evaluation

## Decision Trace Completeness

Decision Trace Completeness evaluates whether an answer exposes an explicit
link from its material claims to cited, retrieved evidence. It does **not**
award explainability credit merely because retrieval or another execution stage
ran.

The benchmark requirement `Link material claims to the evidence and source.`
has one exact, deterministic mapping: `claim_evidence_linkage`.

The Full Sentinel and RAG-only pipelines emit this canonical trace element only
for a non-empty answer with a citation that identifies a retrieved source or
chunk. Retrieval without a grounded citation does not emit it. LLM-only cannot
emit it because that mode does not retrieve evidence.

Generic stages such as `retrieval step`, `source selection`, `policy
interpretation`, and `risk assessment` describe execution, not
claim-to-evidence linkage, and therefore do not satisfy this requirement. The
metric remains required-item coverage; the canonical mapping changes neither
the formula nor its strictness.
