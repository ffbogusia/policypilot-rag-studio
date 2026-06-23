from app.demo_questions import get_default_question, get_demo_questions


def test_demo_questions_have_required_fields() -> None:
    questions = get_demo_questions()

    assert len(questions) >= 4

    for question in questions:
        assert question["label"]
        assert question["question"]
        assert question["expected_behavior"]


def test_demo_questions_include_refusal_case() -> None:
    questions = get_demo_questions()

    refusal_questions = [
        question
        for question in questions
        if question["expected_behavior"] == "refuse_not_in_sources"
    ]

    assert len(refusal_questions) >= 1


def test_default_question_is_available() -> None:
    default_question = get_default_question()

    assert default_question == "Can contractors access production data?"