from langchain_core.prompts import ChatPromptTemplate

PERSONAS = {
    "default": "You are a helpful, precise assistant that answers strictly from the given context.",
    "beginner_tutor": "You are a patient tutor explaining concepts to someone new to the topic. Use simple language and short sentences.",
    "technical_expert": "You are a terse technical assistant speaking to a domain expert. Be precise, use correct terminology, skip basic explanations.",
}

FEW_SHOT_EXAMPLES = """
Example 1
Q: What is the refund window mentioned in the document?
A: The document states a 30-day refund window from the date of purchase.

Example 2
Q: Does the document mention international shipping?
A: The provided context does not mention international shipping.
"""

def build_prompt(persona: str = "default", use_few_shot: bool = True):
    role_instruction = PERSONAS.get(persona, PERSONAS["default"])

    system_template = f"""{role_instruction}

Rules (zero-shot instructions):
- Answer ONLY using the information in the context below.
- If the answer is not in the context, say "The provided context does not mention this."
- Never invent facts not present in the context.
- Cite the page number when possible.

{FEW_SHOT_EXAMPLES if use_few_shot else ""}

Context:
{{context}}
"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}"),
    ])