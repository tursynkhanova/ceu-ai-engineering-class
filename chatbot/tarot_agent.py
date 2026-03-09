from pathlib import Path
import random

import chromadb
from pydantic import BaseModel
from agents import (
    Agent,
    FunctionTool,
    function_tool,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
)

import dotenv
dotenv.load_dotenv()

MODEL = "litellm/bedrock/eu.amazon.nova-micro-v1:0"


def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )


chroma_path = Path("chroma")
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
tarot_cards_rag = chroma_client.get_collection(name="tarot_cards_rag")


TAROT_DECK = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
    "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups",
    "Six of Cups", "Seven of Cups", "Eight of Cups", "Nine of Cups", "Ten of Cups",
    "Page of Cups", "Knight of Cups", "Queen of Cups", "King of Cups",
    "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles",
    "Five of Pentacles", "Six of Pentacles", "Seven of Pentacles", "Eight of Pentacles",
    "Nine of Pentacles", "Ten of Pentacles", "Page of Pentacles", "Knight of Pentacles",
    "Queen of Pentacles", "King of Pentacles",
    "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords",
    "Six of Swords", "Seven of Swords", "Eight of Swords", "Nine of Swords", "Ten of Swords",
    "Page of Swords", "Knight of Swords", "Queen of Swords", "King of Swords",
    "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands",
    "Six of Wands", "Seven of Wands", "Eight of Wands", "Nine of Wands", "Ten of Wands",
    "Page of Wands", "Knight of Wands", "Queen of Wands", "King of Wands"
]


@function_tool
def draw_tarot_cards_tool(topic: str = "general", n_cards: int = 3) -> str:
    """
    Draw random tarot cards for a user reading.
    """
    if n_cards < 1 or n_cards > 10:
        return "Please choose between 1 and 10 cards."

    chosen = random.sample(TAROT_DECK, n_cards)

    results = []
    for i, card in enumerate(chosen, start=1):
        orientation = random.choice(["upright", "reversed"])
        results.append(f"{i}. {card} ({orientation})")

    return f"Tarot reading topic: {topic}\nDrawn cards:\n" + "\n".join(results)


@function_tool
def tarot_lookup_tool(card_name: str, max_results: int = 1) -> str:
    """
    Look up meanings and details for a specific Tarot card.
    """
    results = tarot_cards_rag.query(query_texts=[card_name], n_results=max_results)

    if not results["documents"][0]:
        return f"No information found for card: {card_name}"

    formatted_results = []
    for doc in results["documents"][0]:
        formatted_results.append(doc)

    return f"Meaning for {card_name}:\n" + "\n\n---\n\n".join(formatted_results)


class TarotGuardrailOutput(BaseModel):
    only_about_tarot: bool
    topic: str
    reason: str


guardrail_agent = Agent(
    name="Tarot Guardrail Check",
    instructions="""
    You are a guardrail classifier for a tarot reading assistant.

    Decide whether the user's request is appropriate for a tarot assistant.
    Allowed:
    - tarot readings
    - reflective questions about love, career, relationships, emotions, self-growth
    - entertainment-style mystical
    questions
    - questions about tarot card meanings

    Not allowed:
    - medical or mental health advice
    - legal advice
    - financial or investment advice
    - pregnancy or fertility diagnosis/prediction
    - crisis, self-harm, suicide, emergencies
    - guaranteed future outcomes, exact death predictions
    - clearly off-topic requests unrelated to tarot

    Return:
    - only_about_tarot = true if the request is allowed
    - only_about_tarot = false otherwise
    - topic = short label
    - reason = short explanation
    """,
    output_type=TarotGuardrailOutput,
    model=MODEL,
)


@input_guardrail
async def tarot_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_tarot),
    )


tarot_agent = Agent(
    name="Tarot predictions",
    instructions="""
    You are a tarot reading assistant for reflection and entertainment.

    You answer the user's question using Tarot cards.
    You randomly draw 3 cards out of 78 and interpret them together in a way that is relevant to the user's question.
    If you need to randomly choose cards, use the tool: draw_tarot_cards_tool
    If you need to look up card meanings, use the tool: tarot_lookup_tool

    Important rules:
    - Tarot readings are for reflection and entertainment, not factual certainty.
    - Do not provide medical, mental-health, legal, financial, or pregnancy-related advice.
    - Do not claim guaranteed outcomes.
    - Keep the reading thoughtful, clear, and relevant to the user's question.
    """,
    model=MODEL,
    input_guardrails=[tarot_topic_guardrail],
    tools=[
        bedrock_tool(draw_tarot_cards_tool.__dict__),
        bedrock_tool(tarot_lookup_tool.__dict__)
    ],
)
