import pandas as pd
from pathlib import Path

def create_tarot_text_database(csv_path: str, output_path: str):
    """
    Convert tarot CSV into a text file with formatted documents.
    Each tarot card becomes a formatted text entry suitable for RAG.
    """
    # Read the CSV
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    documents = []

    for index, row in df.iterrows():
        # Handle potential NaN values safely by converting to empty string if missing
        name = row.get('name', '')
        number = row.get('number', '')
        arcana = row.get('arcana', '')
        fortune = row.get('fortune_telling', '')
        light = row.get('meanings_light', '')
        shadow = row.get('meanings_shadow', '')
        questions = row.get('Questions to Ask', '')

        # Create rich document text for semantic search
        document_text = f"""Tarot Card: {name}
Number: {number}
Arcana: {arcana}

Fortune Telling:
{fortune}

Light Meanings:
{light}

Shadow Meanings:
{shadow}

Questions to Ask:
{questions}"""

        documents.append(document_text)

    # Write all documents to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documents):
            f.write(doc)
            # Add separator between documents (except for the last one)
            if i < len(documents) - 1:
                f.write('\n\n---\n\n')

    print(f"Successfully created {output_path}")
    print(f"Converted {len(documents)} tarot cards from {csv_path}")
    return len(documents)

if __name__ == "__main__":

    script_dir = Path(__file__).parent

    csv_path = script_dir.parent / "data" / "tarot_cards_subset.csv"
    
    output_path = script_dir.parent / "data" / "tarot_database_rag.txt"

    num_items = create_tarot_text_database(str(csv_path), str(output_path))
    
    print(f"\nOutput file location: {output_path}")
    print(f"Total cards processed: {num_items}")