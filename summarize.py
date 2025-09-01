import re

def summarize_rent_agreement(document_text: str, file_name: str = "Unknown File") -> dict:
    """
    Summarizes a Rent/Leave & License Agreement into key details.
    Works even if OCR introduces some noise.
    """

    summary = {
        "file_info": {
            "name": file_name,
            "type": "Unknown",
        },
        "date_of_document": "Not Found",
        "parties": {
            "licensor": "Not Found",
            "tenant": "Not Found"
        },
        "premises": "Not Found",
        "deposit_amount": "Not Found",
        "monthly_rent": "Not Found",
        "duration": "Not Found",
        "key_terms": []
    }

    text = document_text.replace("\n", " ")
    text_lower = text.lower()

    # --- Type ---
    document_text_lower = document_text.lower()

    if re.search(r"\b(employment|job agreement|service contract)\b", document_text_lower):
        summary["file_info"]["type"] = "Employment Agreement"
    elif re.search(r"\b(rent agreement|lease agreement|tenancy agreement)\b", document_text_lower):
        summary["file_info"]["type"] = "Rent/Lease Agreement"
    elif re.search(r"\b(purchase agreement|sale deed|bill of sale|acquisition agreement)\b", document_text_lower):
        summary["file_info"]["type"] = "Purchase/Sale Agreement"
    elif re.search(r"\b(non-disclosure agreement|nda)\b", document_text_lower):
        summary["file_info"]["type"] = "Non-Disclosure Agreement (NDA)"
    elif re.search(r"\b(partnership agreement)\b", document_text_lower):
        summary["file_info"]["type"] = "Partnership Agreement"
    elif re.search(r"\b(loan agreement|promissory note)\b", document_text_lower):
        summary["file_info"]["type"] = "Loan Agreement"
    elif re.search(r"\b(terms and conditions|terms of service)\b", document_text_lower):
        summary["file_info"]["type"] = "Terms and Conditions"
    elif re.search(r"\b(will and testament|last will)\b", document_text_lower):
        summary["file_info"]["type"] = "Will and Testament"
    elif re.search(r"\b(affidavit|declaration)\b", document_text_lower):
        summary["file_info"]["type"] = "Affidavit/Declaration"
    elif re.search(r"\b(power of attorney)\b", document_text_lower):
        summary["file_info"]["type"] = "Power of Attorney"
    elif re.search(r"\b(memorandum of understanding|mou)\b", document_text_lower):
        summary["file_info"]["type"] = "Memorandum of Understanding (MOU)"
    else:
        if re.search(r"\b(agreement|contract)\b", document_text_lower):
            summary["file_info"]["type"] = "General Agreement/Contract"

    # --- Date ---
    date_match = re.search(r"(?:\d{1,2}\s+\w+\s+\d{4})", text)
    if date_match:
        summary["date_of_document"] = date_match.group(0)

    # --- Parties ---
    licensor_match = re.search(r"Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+", text)
    tenant_match = re.search(r"Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+", text[text.find("AND"):])
    if licensor_match:
        summary["parties"]["licensor"] = licensor_match.group(0)
    if tenant_match:
        summary["parties"]["tenant"] = tenant_match.group(0)

    # --- Premises ---
    premises_match = re.search(r"Flat\s+No\s*[-]?\s*\w+.*?Pune\s*[-]?\s*\d{6}", text, re.IGNORECASE)
    if premises_match:
        summary["premises"] = premises_match.group(0)

    # --- Deposit ---
    deposit_match = re.search(r"deposit\s+Rs\.?\s*[\d,]+", text, re.IGNORECASE)
    if deposit_match:
        summary["deposit_amount"] = deposit_match.group(0)

    # --- Rent ---
    rent_match = re.search(
    r"(Rs\.?\s*[\d,]+\s*\(Rupees.*?Thousand.*?\)|sum\s*of\s*Rs\.?\s*[\d,]+)", text, re.IGNORECASE)
    if rent_match:
        summary["monthly_rent"] = rent_match.group(0)


    # --- Duration ---
    duration_match = re.search(r"valid for\s+\d+\s+months", text, re.IGNORECASE)
    if duration_match:
        summary["duration"] = duration_match.group(0)

    # --- RISK ---
    if "risk_assessment" not in summary:
        summary["risk_assessment"] = {}

    RISK_KEYWORDS = [
        "unlimited liability", "personal guarantee", "non-compete",
        "confidentiality breach", "penalty", "termination",
        "indemnification", "force majeure", "breach of contract",
        "liquidated damages", "arbitration", "governing law"
    ]

    document_text_lower = document_text.lower()
    found_risks = []

    for keyword in RISK_KEYWORDS:
        if keyword.lower() in document_text_lower:
            # Find context around the keyword
            context_match = re.search(
                f".{{0,100}}{re.escape(keyword)}.{{0,100}}",
                document_text, re.IGNORECASE | re.DOTALL
            )
            if context_match:
                found_risks.append({
                    "keyword": keyword,
                    "context": context_match.group(0).strip().replace('\n', ' ')
                })
            else:
                found_risks.append({"keyword": keyword, "context": "Context not fully captured."})

    if found_risks:
        summary["identified_risks"] = found_risks
        summary["risk_assessment"]["summary"] = "⚠️ Potentially risky clauses identified."
        summary["risk_assessment"]["recommendation"] = (
            "✅ **Recommendation:** These keywords indicate clauses that may have significant implications. "
            "You **should** thoroughly review these sections with legal counsel. "
            "You **should not** sign or agree to the document without fully understanding the impact "
            "of these clauses on your rights and obligations."
        )
    else:
        summary["risk_assessment"]["summary"] = "✔️ No obvious high-risk keywords detected."
        summary["risk_assessment"]["recommendation"] = (
            "✅ **Recommendation:** While no major risk keywords were flagged, "
            "this summarizer is not a substitute for legal advice. "
            "You **should** always have a qualified legal professional review any important document "
            "before signing or making commitments. "
            "You **should not** assume the document is risk-free based solely on this automated summary."
        )

    return summary
