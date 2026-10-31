from app.eval.asqa_answers import short_answer_recall


def test_short_answer_presence_counts_one_alias_per_question():
    pairs = [{"short_answers": ["Edward Furlong", "Furlong"]}, {"short_answers": ["Nick Stahl"]}]
    assert short_answer_recall("Edward Furlong played him.", pairs) == {"matched": 1, "eligible": 2, "recall": .5}


def test_short_answer_presence_is_not_entailment_and_excludes_empty_labels():
    result = short_answer_recall("It was not Edward Furlong.", [{"short_answers": ["Edward Furlong"]}, {"short_answers": []}])
    assert result == {"matched": 1, "eligible": 1, "recall": 1}
    assert short_answer_recall("any", [{"short_answers": []}])["recall"] is None
