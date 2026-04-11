# Knowledge Graphs

## What is a knowledge graph?

A knowledge graph is a database that stores facts as a network of **nodes** and **edges**.

- A **node** represents a thing: a person, place, event, or concept.
- An **edge** represents a relationship between two nodes.

Every fact is stored as a triple: `(subject) → [relationship] → (object)`

For Rose's memoir:

```
(Sean Cosenza)    → [IS_GRANDSON_OF]  → (Rose)
(Rose)            → [ATTENDED]        → (Nazareth Academy)
(Nazareth Academy)→ [LOCATED_IN]      → (LaGrange, Illinois)
(Laura)           → [IS_DAUGHTER_OF]  → (Rose)
(Sean Cosenza)    → [IS_SON_OF]       → (Laura)
```

The power comes from traversal. To answer "who are Rose's grandchildren?", you walk the graph:

```
Rose → [HAS_CHILD] → Laura → [HAS_CHILD] → Sean
Rose → [HAS_CHILD] → Jean  → [HAS_CHILD] → Julia
                           → [HAS_CHILD] → Jennifer
```

No text search required. No hoping the word "grandchildren" appears near "Sean" in the same chunk.

---

## How this differs from a vector store

| | Vector store | Knowledge graph |
|---|---|---|
| Stores | Chunks of text + embeddings | Nodes and typed edges |
| Queries | "Find text semantically similar to this" | "Find all nodes connected to this node via this relationship" |
| Strength | Vague, conceptual questions | Precise relational questions |
| Weakness | Name/relationship lookup | Nuanced or open-ended questions |

They're complementary. Most production systems use both: the knowledge graph answers relational questions, the vector store answers everything else.

---

## The data model for this project

You'd define node types and edge (relationship) types upfront.

**Node types:**
- `Person` — Rose, Sean, Laura, Grandpa Michael, etc.
- `Place` — Cicero IL, Nazareth Academy, Griswold Lake, etc.
- `Event` — weddings, graduations, trips, births, deaths
- `Organization` — St. Mary's Church, Automatic Canteen Company, etc.

**Edge types:**
- `IS_CHILD_OF`, `IS_GRANDCHILD_OF`, `IS_SIBLING_OF`, `IS_COUSIN_OF`
- `MARRIED_TO`, `DIVORCED_FROM`
- `ATTENDED` (school, event)
- `WORKED_AT`
- `LIVED_AT`
- `BORN_IN`, `DIED_IN`
- `PARTICIPATED_IN`

A fragment of the graph for Rose's family:

```
(Rose) ──IS_CHILD_OF──► (Frank Wesolowski Sr.)
(Rose) ──IS_CHILD_OF──► (Mother Wesolowski)
(Rose) ──IS_SIBLING_OF──► (Frank Jr.)
(Rose) ──IS_SIBLING_OF──► (Bob)
(Rose) ──HAS_CHILD──► (Laura)
(Rose) ──HAS_CHILD──► (Jean)
(Laura) ──HAS_CHILD──► (Sean Cosenza)
(Laura) ──HAS_CHILD──► (Zac Cosenza)
(Jean) ──HAS_CHILD──► (Julia)
(Jean) ──HAS_CHILD──► (Jennifer)
(Sean Cosenza) ──IS_GRANDCHILD_OF──► (Rose)  ← derived edge, added for query convenience
```

---

## How you'd build it: extraction

You don't write these triples by hand. You run an LLM over the text to extract them.

**Step 1 — define your schema as a prompt:**

```
Read the following passage and extract all people, places, and relationships.
Return JSON in this format:
{
  "nodes": [{"id": "Sean Cosenza", "type": "Person"}],
  "edges": [{"from": "Sean Cosenza", "relationship": "IS_GRANDCHILD_OF", "to": "Rose"}]
}
```

**Step 2 — run it per document (or per section):**

For 3 files you could pass the entire file. At scale you'd chunk first, extract, then do a merge/deduplication pass (since "Rose" appears in every chunk and you don't want 400 separate Rose nodes).

**Step 3 — merge and resolve:**

The hardest part. "Grandma", "Babcia", "Grandma Florentyna", and "Florentyna Wesolowski" are all the same person. LLM extraction gives you raw strings — you need entity resolution to merge them into one node.

At small scale this is manageable. At large scale it's a research problem.

---

## Storage: how graphs are actually stored

**Option 1: Neo4j**

The most popular graph database. Uses a query language called **Cypher**.

Finding Rose's grandchildren in Cypher:
```cypher
MATCH (rose:Person {name: "Rose"})-[:HAS_CHILD]->(:Person)-[:HAS_CHILD]->(gc:Person)
RETURN gc.name
```

Or if you stored the derived edge:
```cypher
MATCH (rose:Person {name: "Rose"})<-[:IS_GRANDCHILD_OF]-(gc:Person)
RETURN gc.name
```

Neo4j has a free community edition and a Python driver (`neo4j`).

**Option 2: in-memory with NetworkX**

For a small project, you don't need a database at all. NetworkX is a Python library for graphs.

```python
import networkx as nx

G = nx.DiGraph()
G.add_node("Rose", type="Person")
G.add_node("Sean Cosenza", type="Person")
G.add_edge("Sean Cosenza", "Rose", relationship="IS_GRANDCHILD_OF")

# Query: who are Rose's grandchildren?
grandchildren = [n for n, d in G.in_edges("Rose", data=True)
                 if d["relationship"] == "IS_GRANDCHILD_OF"]
```

For ~100 people across 3 bio files, NetworkX is completely sufficient. You'd serialize it to JSON or pickle between runs.

---

## Integrating the graph with RAG

The combined system works like this:

```
user question
    │
    ├─► entity detection: does the question mention a known node?
    │       "grandchildren" → Rose; "Sean" → Sean Cosenza
    │
    ├─► graph query: answer relational questions directly
    │       MATCH (rose)<-[:IS_GRANDCHILD_OF]-(gc) RETURN gc.name
    │       → ["Sean", "Zac", "Julia", "Jennifer"]
    │
    └─► vector search: find relevant prose for context
            → chunks about Sean's birth, Zac's illness, etc.
    
    both results → LLM → answer
```

The LLM now has two inputs:
1. **Structured facts** from the graph: "Sean Cosenza (grandson), born June 8 1994, parents Laura and John"
2. **Prose context** from the vector store: the actual memoir passages

The structured facts anchor the answer; the prose provides depth and color.

---

## The honest tradeoffs

**Worth it for this project if:**
- You want precise answers to relationship questions ("how is X related to Rose?")
- The data is relatively stable (memoirs don't change often)
- You enjoy the engineering — it's a meaningful complexity increase

**Not worth it if:**
- The current hybrid search + reranking is good enough after tuning
- You'd rather invest the time in the API and frontend

**The minimum viable version for this project:**
1. Write `extract.py` — call Groq with each bio file, get back nodes + edges as JSON
2. Build the graph in NetworkX — load the JSON, add nodes and edges
3. At query time — detect if the question is relational, query the graph, inject the result into the LLM context alongside the vector results

That's maybe 150 lines of code and no external database. A good next step if retrieval quality is still lacking after the contextual index and reranker are fully tuned.
