from app.models import Document


CONTEXT_VERSION = 1
CONTEXT_START = "[Document context]"
CONTEXT_END = "[/Document context]"


def base_semantic_content(content: str) -> str:
    if content.startswith(CONTEXT_START) and CONTEXT_END in content:
        return content.split(CONTEXT_END, 1)[1].lstrip()
    return content


def enrich_semantic_content(content: str, document: Document) -> str:
    base_content = base_semantic_content(content)
    lines = [CONTEXT_START]
    if document.company:
        lines.append(f"Company: {document.company}.")
    if document.fiscal_year:
        lines.append(f"Fiscal year: {document.fiscal_year}.")
    if document.document_type:
        lines.append(f"Document type: {document.document_type.replace('_', ' ')}.")
    if document.title:
        lines.append(f"Report: {document.title}.")
    lines.append(CONTEXT_END)
    lines.append(base_content)
    return "\n".join(lines)


def document_context_metadata(document: Document) -> dict[str, object]:
    return {
        "context_version": CONTEXT_VERSION,
        "company": document.company,
        "fiscal_year": document.fiscal_year,
        "document_type": document.document_type,
    }
