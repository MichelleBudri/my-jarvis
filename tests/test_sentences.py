from backend.audio.sentences import SentenceSplitter, clean_for_speech


def feed_all(splitter, tokens):
    out = []
    for t in tokens:
        out += splitter.feed(t)
    if rest := splitter.flush():
        out.append(rest)
    return out


def test_splits_streamed_tokens_into_sentences():
    tokens = ["Boa noite, ", "senhora. A tempe", "ratura é de 18 graus! ", "Deseja algo mais?"]
    assert feed_all(SentenceSplitter(10), tokens) == [
        "Boa noite, senhora.",
        "A temperatura é de 18 graus!",
        "Deseja algo mais?",
    ]


def test_first_sentence_goes_out_early_then_short_ones_merge():
    tokens = ["Boa noite. ", "Sim. ", "Claro, senhora. ", "Já vou."]
    assert feed_all(SentenceSplitter(20), tokens) == [
        "Boa noite.",
        "Sim. Claro, senhora.",
        "Já vou.",
    ]


def test_long_first_sentence_breaks_at_a_comma():
    tokens = [
        "A lua é um satélite natural da Terra, ",
        "que completa uma órbita ",
        "em 27,3 dias. ",
    ]
    tokens += ["Sua gravidade é menor, ", "cerca de um sexto."]
    assert feed_all(SentenceSplitter(20), tokens) == [
        "A lua é um satélite natural da Terra,",
        "que completa uma órbita em 27,3 dias.",
        "Sua gravidade é menor, cerca de um sexto.",
    ]


def test_abbreviations_and_decimals_do_not_split():
    text = "O Dr. Silva disse que o modelo qwen3.5 roda bem. Fim da história."
    assert feed_all(SentenceSplitter(5), [text]) == [
        "O Dr. Silva disse que o modelo qwen3.5 roda bem.",
        "Fim da história.",
    ]


def test_clean_for_speech():
    assert clean_for_speech("**Olá**, senhora! 😀 Veja `isto`.") == "Olá, senhora! Veja isto."
