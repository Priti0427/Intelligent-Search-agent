"""
LLM Prompts for Agentic Search.

This module contains all prompts used by the agent nodes.
Centralizing prompts makes them easier to maintain and tune.
"""

PROMPTS = {
    # Query Analyzer Prompt
    "query_analyzer": """You are a query analysis expert. Analyze the following user query and determine:
1. Query type: "simple" (single fact), "complex" (multiple aspects), or "multi_hop" (requires reasoning across sources)
2. Main topics/entities mentioned
3. Whether it requires current information (web search) or can be answered from documents

User Query: {query}

Respond in JSON format:
{{
    "query_type": "simple|complex|multi_hop",
    "topics": ["topic1", "topic2"],
    "requires_web_search": true|false,
    "requires_academic": true|false,
    "requires_documents": true|false,
    "reasoning": "brief explanation of your analysis"
}}""",

    # Query Decomposer Prompt
    "query_decomposer": """You are a query decomposition expert. Break down the following complex query into simpler sub-queries that can be answered independently.

Original Query: {query}
Query Type: {query_type}
Previous Feedback (if any): {feedback}

Guidelines:
- For simple queries, return just the original query
- For complex queries, break into 2-4 focused sub-queries
- For multi-hop queries, create a logical sequence of sub-queries where later ones may depend on earlier answers
- Each sub-query should be self-contained and searchable

Respond in JSON format:
{{
    "sub_queries": [
        {{"query": "sub-query 1", "source_hint": "web|documents|academic|any"}},
        {{"query": "sub-query 2", "source_hint": "web|documents|academic|any"}}
    ],
    "reasoning": "why you decomposed it this way"
}}""",

    # Router Prompt
    "router": """You are a search routing expert. For each sub-query, determine the best data source(s) to search.

Available sources:
- web: Real-time web search via Tavily (best for current events, recent information)
- documents: Vector database with ingested documents (best for specific domain knowledge)
- academic: arXiv papers (best for research, technical concepts, algorithms)

Sub-queries to route:
{sub_queries}

For each sub-query, assign one or more sources. Respond in JSON format:
{{
    "routing": [
        {{"query": "sub-query 1", "sources": ["web", "documents"]}},
        {{"query": "sub-query 2", "sources": ["academic"]}}
    ]
}}""",

    # Answer Synthesizer Prompt
    "synthesizer": """You are an expert at synthesizing information from multiple sources into comprehensive, well-cited answers.

Original Query: {query}

Retrieved Information:
{context}

Guidelines:
1. Synthesize a comprehensive answer that addresses all aspects of the query
2. Use information from ALL relevant sources
3. Include inline citations using [Source N] format
4. Be accurate and don't make up information not in the sources
5. If sources conflict, acknowledge the discrepancy
6. Structure the answer clearly with paragraphs or bullet points as appropriate

Provide your answer followed by a list of citations:

ANSWER:
[Your synthesized answer with [Source N] citations]

CITATIONS:
[List each source with its number and brief description]""",

    # Self-Reflection Prompt
    "reflector": """You are a quality assurance expert for search answers. Evaluate the following answer for quality.

Original Query: {query}

Generated Answer:
{answer}

Sources Used:
{sources}

Evaluate on these criteria (score 0-1 for each):
1. Relevance: Does the answer address the query?
2. Completeness: Are all aspects of the query covered?
3. Accuracy: Is the information consistent with sources?
4. Citation Quality: Are sources properly cited?
5. Clarity: Is the answer well-structured and clear?

Respond in JSON format:
{{
    "scores": {{
        "relevance": 0.0-1.0,
        "completeness": 0.0-1.0,
        "accuracy": 0.0-1.0,
        "citation_quality": 0.0-1.0,
        "clarity": 0.0-1.0
    }},
    "overall_score": 0.0-1.0,
    "passed": true|false,
    "feedback": "specific suggestions for improvement if score is low",
    "missing_aspects": ["aspect1", "aspect2"]
}}""",

    # Citation Generator Prompt
    "citation_generator": """Extract and format citations from the following sources.

Sources:
{sources}

Format each citation with:
- Source number
- Title or description
- URL or reference
- Relevant excerpt

Respond in JSON format:
{{
    "citations": [
        {{
            "number": 1,
            "title": "Source title",
            "url": "source URL if available",
            "type": "web|document|academic",
            "excerpt": "relevant excerpt from source"
        }}
    ]
}}""",
}


def get_prompt(name: str, **kwargs) -> str:
    """Get a prompt by name and format it with provided kwargs."""
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    return PROMPTS[name].format(**kwargs)
