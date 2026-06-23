DEMO_QUESTIONS: list[dict[str, str]] = [
    {
        "label": "Contractor production access",
        "question": "Can contractors access production data?",
        "expected_behavior": "answer_with_sources",
    },
    {
        "label": "Phishing response",
        "question": "What should an employee do after a phishing email?",
        "expected_behavior": "answer_with_sources",
    },
    {
        "label": "MFA for admins",
        "question": "Do privileged accounts need MFA?",
        "expected_behavior": "answer_with_sources",
    },
    {
        "label": "Out-of-scope refusal",
        "question": "What is the CEO's favorite restaurant?",
        "expected_behavior": "refuse_not_in_sources",
    },
]


def get_demo_questions() -> list[dict[str, str]]:
    """Return predefined demo questions for the Streamlit UI."""

    return DEMO_QUESTIONS.copy()


def get_default_question() -> str:
    """Return the default question shown in the UI."""

    return DEMO_QUESTIONS[0]["question"]