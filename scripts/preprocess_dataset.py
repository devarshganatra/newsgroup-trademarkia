import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

def load_documents(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Recursively loads all files inside the dataset directory.
    
    Returns:
        List[Dict[str, Any]]: List of documents with id, newsgroup, and raw_text.
    """
    documents = []
    doc_id = 0
    base_path = Path(dataset_path)
    
    if not base_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return []

    # Iterate through each newsgroup directory
    for newsgroup_dir in sorted(os.listdir(base_path)):
        group_path = base_path / newsgroup_dir
        if group_path.is_dir():
            # Iterate through each file in the newsgroup directory
            for file_name in sorted(os.listdir(group_path)):
                file_path = group_path / file_name
                if file_path.is_file():
                    try:
                        # Using 'latin-1' encoding as Usenet messages often contain non-UTF-8 characters
                        with open(file_path, 'r', encoding='latin-1') as f:
                            raw_text = f.read()
                            documents.append({
                                "id": doc_id,
                                "newsgroup": newsgroup_dir,
                                "raw_text": raw_text
                            })
                            doc_id += 1
                    except Exception as e:
                        print(f"Warning: Could not read file {file_path}: {e}")
                        
    return documents

def remove_headers(text: str) -> str:
    """
    Removes email headers from the text.
    Headers end at the first blank line.
    """
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    if len(parts) > 1:
        return parts[1]
    return text

def extract_subject(text: str) -> str:
    """
    Extracts the subject line from the header section of the raw text.
    """
    match = re.search(r'^Subject:\s*(.*)$', text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""

def remove_quotes(text: str) -> str:
    """
    Removes lines beginning with '>'.
    """
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('>')]
    return '\n'.join(cleaned_lines)

def remove_signature(text: str) -> str:
    """
    Removes signature blocks starting with '--'.
    """
    # Look for a line that is exactly '--' or starts with '--' followed by whitespace
    parts = re.split(r'\n--\s*\n|\n--$', text, maxsplit=1)
    return parts[0]

def remove_metadata_artifacts(text: str) -> str:
    """
    Removes common FAQ metadata artifacts like archive-name, last-modified, etc.
    These often appear at the start of the body or pinned to the beginning of lines.
    """
    # Patterns to match at the beginning of lines
    # Using more aggressive matching for archive-name to catch prefixes
    metadata_patterns = [
        r'^.*archive-name:.*$',
        r'^last-modified:.*$',
        r'^version:.*$',
        r'^expires:.*$',
        r'^summary:.*$',
        r'^keywords:.*$',
        r'^-+begin pgp.*-+$',
        r'^-+end pgp.*-+$'
    ]
    
    for pattern in metadata_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
    return text

def normalize_text(text: str) -> str:
    """
    Applies basic normalization: lowercase, collapse whitespace, strip.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_document(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Applies the full preprocessing pipeline to a single document.
    """
    raw_text = doc['raw_text']
    
    # 1. Extract Subject from headers BEFORE removing headers
    subject = extract_subject(raw_text)
    
    # 2. Remove Headers
    text = remove_headers(raw_text)
    
    # 3. Prepend Subject
    if subject:
        text = f"{subject}\n\n{text}"
    
    # 4. Remove Quotes
    text = remove_quotes(text)
    
    # 5. Remove Signature
    text = remove_signature(text)
    
    # 6. Remove Metadata Artifacts (Internal FAQ headers, PGP blocks)
    text = remove_metadata_artifacts(text)
    
    # 7. Normalize
    text = normalize_text(text)
    
    # 8. Length Calculation
    char_length = len(text)
    token_length = len(text.split())
    
    # 9. Filtering Rule (Refined to token count)
    if token_length < 5:
        return None
        
    return {
        "id": doc['id'],
        "text": text,
        "newsgroup": doc['newsgroup'],
        "char_length": char_length,
        "token_length": token_length
    }

def main():
    dataset_path = "data/20_newsgroups"
    output_path = "data/processed_corpus.json"
    
    print(f"Loading documents from {dataset_path}...")
    raw_docs = load_documents(dataset_path)
    print(f"Total raw documents loaded: {len(raw_docs)}")
    
    processed_docs = []
    removed_count = 0
    
    print("Processing documents...")
    for doc in raw_docs:
        processed = process_document(doc)
        if processed:
            processed_docs.append(processed)
        else:
            removed_count += 1
            
    # Calculate statistics
    token_lengths = [doc['token_length'] for doc in processed_docs]
    if token_lengths:
        avg_tokens = sum(token_lengths) / len(token_lengths)
        max_tokens = max(token_lengths)
        min_tokens = min(token_lengths)
    else:
        avg_tokens = max_tokens = min_tokens = 0

    print(f"Documents removed (short): {removed_count}")
    print(f"Final document count: {len(processed_docs)}")
    print(f"Average token length: {avg_tokens:.2f}")
    print(f"Maximum token length: {max_tokens}")
    print(f"Minimum token length: {min_tokens}")
    
    # Save to JSON
    print(f"Saving processed corpus to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
    # Sample display
    if processed_docs:
        print("\n" + "="*50)
        print("Sample cleaned documents:")
        for doc in processed_docs[:3]:
            print(f"ID: {doc['id']}")
            print(f"Group: {doc['newsgroup']}")
            print(f"Char Length: {doc['char_length']}")
            print(f"Token Length: {doc['token_length']}")
            print(f"Text: {doc['text'][:200]}...")
            print("-" * 30)
        print("="*50)

if __name__ == "__main__":
    main()
