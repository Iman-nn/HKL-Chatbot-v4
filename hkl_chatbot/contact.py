"""Curated general contact answer; keep department-specific queries in the corpus.

Source checked 2026-10-19: https://hkl.moh.gov.my/en/ (CONTACT US).
Review this record with the hospital content owner when contact details change.
"""

import re


_HOSPITAL = r"(?:hkl|hospital|hospital kuala lumpur|the hospital)"
_INTENTS = (
    rf"(?:how (?:can|do) i|how to) contact {_HOSPITAL}",
    rf"(?:contact|call|phone|telephone|contact number|phone number|telephone number) (?:for |of )?{_HOSPITAL}",
    rf"{_HOSPITAL} (?:contact|contact details|contact number|phone|phone number|telephone number|website)",
    rf"(?:what is|what s) (?:the )?(?:contact|phone|telephone) number (?:for |of )?{_HOSPITAL}",
    rf"(?:official website|website) (?:for |of )?{_HOSPITAL}",
    rf"(?:bagaimana (?:cara |untuk )?|cara )?hubungi {_HOSPITAL}",
    rf"(?:apakah |apa )?(?:nombor (?:telefon|hubungan)|telefon|laman web rasmi|laman web) {_HOSPITAL}",
)


def contact_answer(cleaned: str):
    if not any(re.fullmatch(pattern, cleaned) for pattern in _INTENTS):
        return None
    malay = bool(re.search(r"\b(hubungi|nombor|telefon|laman)\b", cleaned))
    response = (
        "Anda boleh menghubungi Hospital Kuala Lumpur melalui:\n\n"
        "**Telefon utama:** 03-2615 5555\n"
        "**Laman web rasmi:** https://hkl.moh.gov.my/\n\n"
        "Untuk menghubungi klinik atau jabatan tertentu, minta disambungkan melalui talian utama."
        if malay else
        "You can contact Hospital Kuala Lumpur through:\n\n"
        "**Main switchboard:** 03-2615 5555\n"
        "**Official website:** https://hkl.moh.gov.my/\n\n"
        "For a specific clinic or department, ask the switchboard to connect you."
    )
    return {"response": response, "suggestions": [], "source": "official_contact",
            "references": [{"title": "HKL official website", "url": "https://hkl.moh.gov.my/en/",
                            "checked_on": "2026-10-19"}]}
