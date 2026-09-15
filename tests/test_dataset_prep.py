from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from vc_clone_graph.dataset_prep import (
    _audit_question_reveals_panel_stance,
    _is_investor_question,
    build_pitch_only_document,
    compile_review_rows,
    locate_investor_turn,
    package_relative_path,
    write_pitch_package,
)
from vc_clone_graph.precedent_builder import parse_speaker_turns


_PANEL_VERDICT_QUESTIONS = (
    "So wait, are you still passing? Or did Howie convince you?",
    "So you're out? You're passing? You're done?",
    "So you know what I love about this?",
    "You gonna invest in this?",
    "Is Howie in?",
    "Did she pass on this?",
    "Are they going to invest in the company?",
    "What's your offer?",
    "Does he hate this deal?",
    "Have you been convinced by the pitch?",
    "Howie, you're gonna go in for 150?",
    "So are you in?",
    "How much are you in for?",
    "Are you in for 500?",
    "You're in for 250?",
    "How much did you come in at?",
    "And you would invest in the company?",
    "You want to go in, don't you?",
    "How much are you going to put in?",
    "How much are you going to invest?",
    "Phil, do you, have you invested?",
    "Did Sarah pass on this?",
    "Will Marcus invest in the company?",
    "How much is Phil going to put in?",
    "Should I pass on this?",
)

_FACTUAL_VERDICT_VOCABULARY_QUESTIONS = (
    "What are the margins like on a product like this?",
    "Have you invested in apparel?",
    "How much are you going to invest in customer acquisition?",
    "Are you in Walgreens?",
    "How many users pass through checkout?",
    "What offer do customers see?",
    "What do you like about your current investors?",
    "Are you done building the prototype?",
    "Did the pilot convince you to change pricing?",
    "What did you invest in, what are we investing in now?",
    "How much did he put in?",
    "How much of that is done?",
    (
        "Who is responsible for thinking up all the use cases, do those come "
        "from you? Like your imagination of the cleaning robot? Are you like "
        "the idea generator of like this, basically the skills that need to be built?"
    ),
)

_MULTIWORD_PANEL_NAMES = ("Elizabeth", "Sarah Jane")
_PARITY_PANEL_NAMES = ("Elizabeth", "Howie", "Sarah", "Marcus", "Phil")
_MULTIWORD_PANEL_VERDICT_QUESTIONS = (
    "Did Sarah Jane pass on this?",
    "Will Sarah Jane invest in the company?",
    "How much is Sarah Jane going to put in?",
)
_ACTORLESS_VERDICT_REMNANTS = (
    "Did pass on this?",
    "Will invest in the company?",
    "How much is going to put in?",
    "Did convince you?",
)
_MULTIWORD_PANEL_FACTUAL_QUESTIONS = (
    ("Did Sarah Jane pass through checkout?", "Did pass through checkout?"),
    (
        "Will Sarah Jane invest in customer acquisition?",
        "Will invest in customer acquisition?",
    ),
    (
        "How much is Sarah Jane going to put into customer acquisition?",
        "How much is going to put into customer acquisition?",
    ),
    (
        "Did Sarah Jane convince you to change pricing?",
        "Did convince you to change pricing?",
    ),
)


@pytest.mark.parametrize(
    "founder_text",
    (
        "We just passed 166,000 users this month.",
        "Our conversion rate is out of 166,000 users.",
        "Martim is a mechanical engineer.",
        "Her mom then passed away, which inspired the company.",
        "Here's the good news. I've already sold 5000 units.",
        "Kalle has ten years of coffee experience.",
    ),
)
def test_pitch_document_preserves_legitimate_founder_language_that_resembles_narration(
    founder_text: str,
) -> None:
    turns = parse_speaker_turns(
        f"Founder: Opening evidence.\nFounder: {founder_text}\nElizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder", "Martim", "Kalle"),
        panel_names=("Elizabeth",),
        endpoint=2,
    )

    assert pitch == f"Founder: Opening evidence.\nFounder: {founder_text}\n"


def test_cross_investor_split_speaker_preserves_founder_evidence_without_leakage() -> None:
    turns = parse_speaker_turns(
        "Iñaki\n: We just passed 166,000 users.\n"
        "Elizabeth Yin\n: What is your conversion rate?\n"
        "Iñaki\n: Our conversion rate is out of 166,000 users.\n"
        "Elizabeth Yin\n: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Iñaki",),
        panel_names=("Elizabeth Yin",),
        endpoint=3,
    )

    assert pitch == (
        "Iñaki: We just passed 166,000 users.\n"
        "Investor question: What is your conversion rate?\n"
        "Iñaki: Our conversion rate is out of 166,000 users.\n"
    )
    assert "Elizabeth" not in pitch
    assert "I am out" not in pitch


@pytest.mark.parametrize(
    ("episode", "transcript", "founders", "panel", "endpoint", "expected", "prohibited"),
    (
        (
            3,
            "Phil: Okay. When you're investing your money, do the right thing.\n"
            "But what happens when someone tries to reinvent an industry as old as trash itself?\n"
            "Amanda: Revenue is $100k.\nElizabeth: I am out.",
            ("Amanda",),
            ("Phil", "Elizabeth"),
            3,
            "Amanda: Revenue is $100k.\n",
            "what happens when",
        ),
        (
            9,
            "Alex: Our answer is obvious.\n"
            "I gotta say, calling that answer obvious takes some confidence.\n"
            "Elizabeth: I am out.",
            ("Alex",),
            ("Elizabeth",),
            1,
            "Alex: Our answer is obvious.\n",
            "takes some confidence",
        ),
        (
            26,
            "Maria: We make physical therapy effective at home.\n"
            "But Maria will have to convince investors she should be leading the company.\n"
            "Elizabeth: I am out.",
            ("Maria",),
            ("Elizabeth",),
            1,
            "Maria: We make physical therapy effective at home.\n",
            "convince investors",
        ),
        (
            97,
            "Charles: Nice to meet you.\n"
            "Christopher confidently strides in and shakes the investors hands, remember when that was a thing?\n"
            "Christopher: The workflow has 70,000 MRR.\n"
            "If you are thinking this pitch is confusing, you are not alone.\n"
            "Elizabeth: I am out.",
            ("Christopher",),
            ("Charles", "Elizabeth"),
            3,
            "Christopher: The workflow has 70,000 MRR.\n",
            "remember when",
        ),
        (
            10,
            "Josh: Our peak was 500 subscribers and now we're closer to 400.\n"
            "Did you catch that? A declining base is not an encouraging sign.\n"
            "Elizabeth: What is retention?\nJosh: It is 80%.\nElizabeth: I am out.",
            ("Josh",),
            ("Elizabeth",),
            4,
            "Josh: Our peak was 500 subscribers and now we're closer to 400.\n"
            "Investor question: What is retention?\nJosh: It is 80%.\n",
            "not an encouraging sign",
        ),
        (
            6,
            "Evan: Today is our first day fundraising.\n"
            "Right at the end of the pitch, the founders drop a bombshell: the investors are leaning in.\n"
            "Elizabeth: You in or you out?\n"
            "Troy: We want direct feedback.\nElizabeth: I am out.",
            ("Evan", "Troy"),
            ("Elizabeth",),
            3,
            "Evan: Today is our first day fundraising.\nTroy: We want direct feedback.\n",
            "leaning in",
        ),
        (
            32,
            "Michael: I am passing.\nXiao: Thank you.\nMichael's out. Here's Daniel.\n"
            "Daniel: Is it a data play?\nXiao: Yes.\nJillian: I am in.",
            ("Xiao",),
            ("Michael", "Daniel", "Jillian"),
            5,
            "Investor question: Is it a data play?\nXiao: Yes.\n",
            "out. Here's",
        ),
        (
            61,
            "Jen: The contract collapsed after a stakeholder left.\n"
            "It sounds like Hearken is having a tough time and the investors are nervous.\n"
            "Michael: How will you reduce churn?\nJen: We train more stakeholders.\n"
            "Pretty sure Jen just told Michael she doesn't want his money.\n"
            "Sarah: I am out.",
            ("Jen",),
            ("Michael", "Sarah"),
            4,
            "Jen: The contract collapsed after a stakeholder left.\n"
            "Investor question: How will you reduce churn?\n"
            "Jen: We train more stakeholders.\n",
            "investors are nervous",
        ),
    ),
)
def test_pitch_document_removes_structural_editorial_fragments_from_reviewed_episodes(
    episode: int,
    transcript: str,
    founders: tuple[str, ...],
    panel: tuple[str, ...],
    endpoint: int,
    expected: str,
    prohibited: str,
) -> None:
    pitch, _ = build_pitch_only_document(
        turns=parse_speaker_turns(transcript),
        founder_names=founders,
        panel_names=panel,
        endpoint=endpoint,
    )

    assert pitch == expected, f"episode {episode}"
    assert prohibited.casefold() not in pitch.casefold()


@pytest.mark.parametrize(
    "narration",
    (
        "In a manner familiar to anyone who's ever been late, Amanda tries to recover.",
        "So, Amanda explains, the waste produces methane.",
        "Amanda says that she already secured a large check.",
        "Eventually Amanda moved on from life near the landfill.",
        "And this is when Amanda pulls out a clear ziplocked bag.",
        "It's a clear sign that Amanda is hitting her stride.",
        "So Amanda's plan is to start small.",
        "When Amanda says further down the chain, she means prices could fall.",
        "Amanda's trying to say they are not up and running yet.",
        "After seeing these broken food systems up close, Alex teamed up with Meghan.",
        "For a lot of venture capitalists, this is the sweet spot.",
        "Let's find out if our investors think the founders can deliver.",
        "Basically, what Maria is saying is the Flexdot should sell itself.",
        "But Maria actually sees a bigger opportunity.",
        "It's one thing when investors are grilling you about your product. "
        "They are asking whether Maria should even be leading the company.",
        "Christopher's first bare-bones product has 70K in monthly revenue.",
        "Christopher has already proven that he can raise money.",
        "Kalle got the title by making the best tasting coffee.",
        "But, for Kalle, coffee-making is not just about winning awards.",
        "Kalle is not just any barista.",
        "So he had an idea - he'll sell instant coffee.",
        "Joshua Zloof is the business side of Sudden Coffee.",
        "Evan explains that casinos rely on third party vendors.",
        "50-50, 80-20, 70-30... the point is, there will be some kind of split.",
        "Troy says the game is like a regular slot machine.",
        "Once Xiao began to dig into this question, he found a complex problem.",
        "All right, so the way Boundless works is through an online application.",
        "In short, it's TurboTax for immigration.",
        "So if you actually know what all those forms are, you're probably a lawyer.",
        "In this case, Xiao is showing investors how Boundless simplifies applications.",
        "Xiao's pushing the investors to look beyond today's numbers.",
        "Xiao's vision for Boundless is to own the entire space.",
        "With both Phil and Michael in! The last up is Jillian.",
        "And while that might be an honorable mission, investors must decide.",
        "What she's built with Hearken is a suite of tools.",
        "When a newsroom is using Hearken, readers can suggest questions.",
        "When Jen says turnkey, she's talking about self service.",
        "Today, the Zebra startup asks whether the company is venture-backable.",
        "Here's .",
    ),
)
def test_pitch_document_removes_actual_reviewed_episode_narration(
    narration: str,
) -> None:
    turns = parse_speaker_turns(
        f"Founder: Founder evidence.\n{narration}\nElizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=(
            "Founder",
            "Amanda",
            "Alex",
            "Maria",
            "Christopher",
            "Kalle",
            "Joshua",
            "Evan",
            "Troy",
            "Xiao",
            "Jen",
        ),
        panel_names=("Elizabeth", "Phil", "Michael", "Jillian"),
        endpoint=1,
    )

    assert pitch == "Founder: Founder evidence.\n"


def test_pitch_document_removes_split_emphasis_before_structural_narration() -> None:
    turns = parse_speaker_turns(
        "Maria: We provide physical therapy oversight at home.\n"
        "At home.\n"
        "This\n"
        "is where companies are targeting consumers these days.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Maria",),
        panel_names=("Elizabeth",),
        endpoint=1,
    )

    assert pitch == "Maria: We provide physical therapy oversight at home.\n"


def test_pitch_document_rechecks_structure_after_panel_name_redaction() -> None:
    turns = parse_speaker_turns(
        "Founder: Founder evidence.\n"
        "Here's Jillian.\n"
        "Founder: More founder evidence.\n"
        "Jillian: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Jillian",),
        endpoint=2,
    )

    assert pitch == "Founder: Founder evidence.\nFounder: More founder evidence.\n"


def test_pitch_document_keeps_founder_words_and_only_neutral_investor_questions() -> None:
    transcript = """\
Founder: We sell workflow software to clinics.
Elizabeth: I love the mission. What is your annual recurring revenue? This looks great.
Founder: Our ARR is $120,000.
Charles: I'm in for $50k.
Founder: Thank you!
Other Investor: Why do customers renew?
Founder: They save ten hours every week.
Elizabeth: The market is too narrow, so I'm out.
Founder: We later doubled revenue.
"""
    turns = parse_speaker_turns(transcript)

    pitch, segments = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth", "Charles", "Other Investor"),
        endpoint=7,
    )

    assert pitch == (
        "Founder: We sell workflow software to clinics.\n"
        "Investor question: What is your annual recurring revenue?\n"
        "Founder: Our ARR is $120,000.\n"
        "Investor question: Why do customers renew?\n"
        "Founder: They save ten hours every week.\n"
    )
    assert "love the mission" not in pitch
    assert "I'm in" not in pitch
    assert "Thank you" not in pitch
    assert "Elizabeth" not in pitch
    assert "doubled revenue" not in pitch
    assert any(row["action"] == "neutralized" for row in segments)


def test_locate_investor_turn_binds_exact_source_wording() -> None:
    turns = parse_speaker_turns(
        "Founder: We are raising.\n"
        "Elizabeth Yin: Assuming we agree on valuation, I'm in for $50k.\n"
    )

    index, quote = locate_investor_turn(
        turns,
        investor_aliases=("Elizabeth", "Elizabeth Yin"),
        evidence_contains="I'm in for $50k",
    )

    assert index == 1
    assert quote == "Assuming we agree on valuation, I'm in for $50k."


def test_locate_investor_turn_can_select_reviewed_duplicate_occurrence() -> None:
    turns = parse_speaker_turns(
        "Elizabeth: I am out. Host narration.\n"
        "Elizabeth: I am out.\n"
    )

    index, quote = locate_investor_turn(
        turns,
        investor_aliases=("Elizabeth",),
        evidence_contains="I am out",
        occurrence=2,
    )

    assert index == 1
    assert quote == "I am out."


def test_pitch_document_strips_host_bridge_appended_to_founder_turn() -> None:
    turns = parse_speaker_turns(
        "Founder: We have ten customers. [BREAK] Welcome back. The investors all passed.\n"
        "Elizabeth: I'm out.\n"
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=1,
    )

    assert pitch == "Founder: We have ten customers.\n"
    assert "passed" not in pitch


def test_pitch_document_removes_target_names_and_nonquestion_narration() -> None:
    turns = parse_speaker_turns(
        "Founder: Hi Elizabeth. We have ten customers. The investors are trying to decide.\n"
        "Elizabeth: What Elizabeth is offering the founder is a loan.\n"
        "Charles: Elizabeth, was that your question?\n"
        "Founder: Our answer is yes.\n"
        "Elizabeth: I am out.\n"
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth Yin", "Charles Hudson"),
        endpoint=4,
    )

    assert pitch == (
        "Founder: We have ten customers.\n"
        "Investor question: was that your question?\n"
        "Founder: Our answer is yes.\n"
    )
    assert "Elizabeth" not in pitch
    assert "investors are trying" not in pitch
    assert "offering the founder" not in pitch


def test_pitch_document_drops_punctuation_left_by_panel_name_redaction() -> None:
    pitch, segments = build_pitch_only_document(
        turns=parse_speaker_turns("Jim: Charles.\nElizabeth: I am out."),
        founder_names=("Jim",),
        panel_names=("Charles", "Elizabeth"),
        endpoint=1,
    )

    assert pitch == ""
    assert segments[0]["action"] == "removed"


def test_pitch_document_drops_question_destroyed_by_panel_name_redaction() -> None:
    pitch, segments = build_pitch_only_document(
        turns=parse_speaker_turns("Mac: How about you, Beck?\nJillian: I am out."),
        founder_names=("Founder",),
        panel_names=("Mac", "Beck", "Jillian"),
        endpoint=1,
    )

    assert pitch == ""
    assert segments[0]["action"] == "removed"


def test_pitch_document_removes_embedded_editorial_narration() -> None:
    turns = parse_speaker_turns(
        "Founder: We have ten customers.\n"
        "The investors sound excited, but will they invest?\n"
        "Founder\n: Revenue is $100k.\n"
        "Elizabeth: What is retention? But the investors want to know if this is real?\n"
        "Founder: Retention is 90%.\n"
        "The founder has strong numbers, but the market may be too small.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=4,
    )

    assert pitch == (
        "Founder: We have ten customers.\n"
        "Founder: Revenue is $100k.\n"
        "Investor question: What is retention?\n"
        "Founder: Retention is 90%.\n"
    )
    assert "investors" not in pitch
    assert "market may be" not in pitch


def test_pitch_document_rejects_host_question_about_investors() -> None:
    turns = parse_speaker_turns(
        "Founder: We built a prototype.\n"
        "Elizabeth: Can the investors get past the prototype and see the vision?\n"
        "Elizabeth: How many customers use it?\n"
        "Founder: Ten customers.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=4,
    )

    assert pitch == (
        "Founder: We built a prototype.\n"
        "Investor question: How many customers use it?\n"
        "Founder: Ten customers.\n"
    )


@pytest.mark.parametrize(
    "decision_question",
    _PANEL_VERDICT_QUESTIONS,
)
def test_pitch_document_rejects_panel_verdict_and_evaluation_questions(
    decision_question: str,
) -> None:
    turns = parse_speaker_turns(
        "Founder: We have ten customers.\n"
        f"Elizabeth: {decision_question}\n"
        "Founder: Retention is 90%.\n"
        "Elizabeth: I am out."
    )

    pitch, segments = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=_PARITY_PANEL_NAMES,
        endpoint=3,
    )

    assert pitch == (
        "Founder: We have ten customers.\n"
        "Founder: Retention is 90%.\n"
    )
    assert segments[1]["action"] == "removed"


def test_pitch_document_preserves_factual_questions_with_verdict_vocabulary() -> None:
    questions = _FACTUAL_VERDICT_VOCABULARY_QUESTIONS
    transcript = "Founder: We have ten customers.\n" + "".join(
        f"Elizabeth: {question}\n" for question in questions
    ) + "Elizabeth: I am out."
    turns = parse_speaker_turns(transcript)

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=len(questions) + 1,
    )

    assert pitch == "Founder: We have ten customers.\n" + "".join(
        f"Investor question: {question}\n" for question in questions
    )


@pytest.mark.parametrize("question", _FACTUAL_VERDICT_VOCABULARY_QUESTIONS)
def test_question_filter_and_independent_audit_agree_on_factual_questions(
    question: str,
) -> None:
    assert _is_investor_question(question) is True
    assert _audit_question_reveals_panel_stance(question) is False


@pytest.mark.parametrize("question", _PANEL_VERDICT_QUESTIONS)
def test_question_filter_and_independent_audit_agree_on_panel_verdicts(
    question: str,
) -> None:
    assert _is_investor_question(question, _PARITY_PANEL_NAMES) is False
    assert _audit_question_reveals_panel_stance(question, _PARITY_PANEL_NAMES) is True


@pytest.mark.parametrize("question", _MULTIWORD_PANEL_VERDICT_QUESTIONS)
def test_question_filter_and_audit_reject_exact_multiword_panel_verdicts(
    question: str,
) -> None:
    assert (
        _is_investor_question(question, panel_names=_MULTIWORD_PANEL_NAMES) is False
    )
    assert (
        _audit_question_reveals_panel_stance(
            question, panel_names=_MULTIWORD_PANEL_NAMES
        )
        is True
    )


@pytest.mark.parametrize("question", _ACTORLESS_VERDICT_REMNANTS)
def test_question_filter_and_audit_fail_closed_on_actorless_verdicts(
    question: str,
) -> None:
    assert _is_investor_question(question) is False
    assert _audit_question_reveals_panel_stance(question) is True


@pytest.mark.parametrize(
    ("question", "sanitized"), _MULTIWORD_PANEL_FACTUAL_QUESTIONS
)
def test_question_filter_and_audit_allow_factual_multiword_panel_questions(
    question: str, sanitized: str
) -> None:
    assert _is_investor_question(question, panel_names=_MULTIWORD_PANEL_NAMES) is True
    assert (
        _audit_question_reveals_panel_stance(
            question, panel_names=_MULTIWORD_PANEL_NAMES
        )
        is False
    )
    assert _is_investor_question(sanitized) is True
    assert _audit_question_reveals_panel_stance(sanitized) is False


def test_pitch_document_truncates_show_narration_after_real_content() -> None:
    turns = parse_speaker_turns(
        "Founder: We raised at an $11m cap.\n"
        "You can almost see the gears turning in the investors heads.\n"
        "Elizabeth: Can you explain Thryft? Wait... what? Before Dressd, there was Thryft? "
        "Have the investors written her off too soon?\n"
        "Founder: It was our prior product.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=3,
    )

    assert pitch == (
        "Founder: We raised at an $11m cap.\n"
        "Investor question: Can you explain Thryft?\n"
        "Founder: It was our prior product.\n"
    )


def test_pitch_document_removes_show_intros_and_outcome_commentary() -> None:
    turns = parse_speaker_turns(
        "Founder: My name is Ari.\n"
        "I'm Josh Muccio and this is The Pitch.\n"
        "Founder: We help employees.\n"
        "Charles: I would like to invest $25k.\n"
        "Founder: Thank you, I appreciate it.\n"
        "Well... that was fast. Charles is in on the company.\n"
        "Elizabeth: What is retention?\n"
        "Founder: It is 90%.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Charles", "Elizabeth"),
        endpoint=6,
    )

    assert pitch == (
        "Founder: My name is Ari.\n"
        "Founder: We help employees.\n"
        "Investor question: What is retention?\n"
        "Founder: It is 90%.\n"
    )


def test_pitch_document_preserves_multiline_founder_pitch() -> None:
    turns = parse_speaker_turns(
        "Founder: We help employees manage money.\n"
        "Founder: That's why we're building a coaching platform.\n"
        "Founder: There will be 100 million target users by 2030.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=3,
    )

    assert pitch == (
        "Founder: We help employees manage money.\n"
        "Founder: That's why we're building a coaching platform.\n"
        "Founder: There will be 100 million target users by 2030.\n"
    )


@pytest.mark.parametrize(
    ("episode_slug", "unattributed_continuation"),
    (
        (
            "91-if-we-dont-get-the-money-by-friday",
            "The investors went back and forth with Cody. But then passed on the deal, "
            "and one after the other, all the investors said no.",
        ),
        (
            "39-this-pitch-is-damn-near-perfect",
            "It's wild to me that Stefan was ready with an answer. I think he nailed it!",
        ),
        ("7-shimmur", "Ah, Graycroft is in."),
        (
            "18-rowvigor",
            "Even though Charles sounds like he's going to pass, Kevin keeps pitching.",
        ),
        (
            "11-tesloop",
            "Haydn is still the founder, and his job today is to sell investors.",
        ),
        (
            "51-this-bot-can-fight-your-atm-fees",
            "The investors don't like the price tag Paul just threw out.",
        ),
        (
            "119-amateur-golf-society-venture-capital-vs-private-equity",
            "Okay. If you're confused here's a quick recap of the ownership.",
        ),
        (
            "87-uber-for-pets",
            "Aparna says Spot On has groomed its service just for pets.",
        ),
        (
            "50-ticket-scalpers-beware-blockchain-is-coming-for-you",
            "With Jillian's pass, that leaves 2 investors.",
        ),
    ),
)
def test_pitch_document_removes_unattributed_continuations_from_reported_episodes(
    episode_slug: str,
    unattributed_continuation: str,
) -> None:
    turns = parse_speaker_turns(
        "Founder: We have a working product.\n"
        f"{unattributed_continuation}\n"
        "Elizabeth: What is revenue?\n"
        "Founder: Revenue is $100k.\n"
        "Elizabeth: I am out."
    )

    pitch, segments = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth", "Jillian", "Charles"),
        endpoint=4,
    )

    assert pitch == (
        "Founder: We have a working product.\n"
        "Investor question: What is revenue?\n"
        "Founder: Revenue is $100k.\n"
    ), episode_slug
    assert unattributed_continuation.casefold() not in pitch.casefold()
    assert segments[0]["unattributed_continuation_count"] == 1


def test_pitch_document_removes_unknown_unattributed_continuation_without_regex() -> None:
    turns = parse_speaker_turns(
        "Founder: Explicitly attributed evidence.\n"
        "Quartz meadow lanterns remain pleasantly ordinary.\n"
        "Elizabeth: I am out."
    )

    pitch, segments = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=1,
    )

    assert pitch == "Founder: Explicitly attributed evidence.\n"
    assert segments[0]["unattributed_continuation_count"] == 1


@pytest.mark.parametrize(
    "transcript",
    (
        "Founder:\nActual founder evidence.\nUnattributed narration.\nElizabeth: I am out.",
        "Founder\n: Actual founder evidence.\nUnattributed narration.\nElizabeth: I am out.",
    ),
)
def test_pitch_document_attributes_first_content_after_empty_or_split_marker(
    transcript: str,
) -> None:
    pitch, segments = build_pitch_only_document(
        turns=parse_speaker_turns(transcript),
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=1,
    )

    assert pitch == "Founder: Actual founder evidence.\n"
    assert segments[0]["unattributed_continuation_count"] == 1


def test_pitch_document_removes_named_stage_directions_and_offer_teaser() -> None:
    turns = parse_speaker_turns(
        "Founder: We are open to offers.\n"
        "Kate makes an offer, after this.\n"
        "Founder: I love direct.\n"
        "[Charles coughs]\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Kate", "Charles", "Elizabeth"),
        endpoint=2,
    )

    assert pitch == (
        "Founder: We are open to offers.\n"
        "Founder: I love direct.\n"
    )


@pytest.mark.parametrize(
    "outcome_narration",
    (
        "Babyscripts needs another million. Has Juan-Pablo convinced investors to take a chance?",
        "Three out of four investors have passed. Only Jillian remains.",
        "So Phil just passed, or wait, was it Laura that just passed?",
        "Phil is out. Jillian is the last investor left.",
        "Everyone is out except Jillian. But does she really want to invest?",
        "So Charles is out. He might invest after more work.",
        "Michael passed on the deal, and one after another, all the investors said no.",
        "So Charles is in. Here's Jillian.",
        "Both Phil and Michael are in with conditions. Here's Jillian.",
        "Jillian is the only investor left.",
        "It's decision time. Jillian is first.",
        "Let's see if their first try was enough to hook our investors.",
        "Let's see how this plays out.",
        "Have the founders done enough to get investors on board?",
        "So investors are thinking about whether they should invest.",
        "Will he pass the test with our investors?",
        "Will investors decide that their device is a heavyweight?",
        "Now can they get investors to put their bucks behind their beans?",
    ),
)
def test_pitch_document_truncates_embedded_outcome_narration(
    outcome_narration: str,
) -> None:
    turns = parse_speaker_turns(
        "Founder: Revenue is $100k.\n"
        f"{outcome_narration}\n"
        "Elizabeth: What is retention?\n"
        "Founder: Retention is 90%.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Elizabeth",),
        endpoint=3,
    )

    assert pitch == (
        "Founder: Revenue is $100k.\n"
        "Investor question: What is retention?\n"
        "Founder: Retention is 90%.\n"
    )


def test_pitch_document_rejects_outcome_question_embedded_in_panel_turn() -> None:
    turns = parse_speaker_turns(
        "Founder: Revenue is $100k.\n"
        "Charles: I like the business. Charles passed? Let's see who else invests.\n"
        "Elizabeth: What is retention?\n"
        "Founder: Retention is 90%.\n"
        "Elizabeth: I am out."
    )

    pitch, _ = build_pitch_only_document(
        turns=turns,
        founder_names=("Founder",),
        panel_names=("Charles", "Elizabeth"),
        endpoint=4,
    )

    assert pitch == (
        "Founder: Revenue is $100k.\n"
        "Investor question: What is retention?\n"
        "Founder: Retention is 90%.\n"
    )


def test_compile_review_rows_materializes_exact_decision_evidence(tmp_path) -> None:
    transcripts = tmp_path / "episodes"
    transcripts.mkdir()
    (transcripts / "10-example.json").write_text(
        '{"panel":[{"name":"Elizabeth Yin","slug":"elizabeth-yin"}],'
        '"founders":[{"name":"Founder"}],'
        '"transcript":"Founder: We are raising.\\n'
        'Elizabeth: I am in for $25k."}',
        encoding="utf-8",
    )

    rows = compile_review_rows(
        review_rows=[
            {
                "episode_slug": "10-example",
                "final_decision": "Out",
                "pitch_window_decision": "In",
                "decision_context": "later_diligence",
                "initial_response": "In",
                "evidence_contains": "in for $25k",
                "founder_aliases": ["Founder Alias"],
                "label_basis": "Concrete pitch-room commitment.",
                "audit_notes": "Later outcome does not replace the first response.",
            }
        ],
        transcript_root=transcripts,
        vc_slug="elizabeth-yin",
        investor_aliases=("Elizabeth", "Elizabeth Yin"),
    )

    assert rows[0]["evidence_quote"] == "I am in for $25k."
    assert rows[0]["evidence_turn_index"] == 1
    assert rows[0]["evaluation_eligible"] is True
    assert rows[0]["on_panel"] is True
    assert rows[0]["founder_aliases"] == ["Founder Alias"]


def test_compile_review_rows_rejects_unknown_decision_context(tmp_path) -> None:
    transcripts = tmp_path / "episodes"
    transcripts.mkdir()
    (transcripts / "10-example.json").write_text(
        '{"panel":[],"founders":[],"transcript":""}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid decision context"):
        compile_review_rows(
            review_rows=[
                {
                    "episode_slug": "10-example",
                    "final_decision": "Out",
                    "pitch_window_decision": "Unobserved",
                    "decision_context": "non_comparable",
                    "initial_response": "NotPresent",
                    "label_basis": "No decision.",
                    "audit_notes": "No decision.",
                }
            ],
            transcript_root=transcripts,
            vc_slug="elizabeth-yin",
            investor_aliases=("Elizabeth",),
        )


def test_write_pitch_package_creates_hash_bound_clean_audit(tmp_path) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        '{"founders":[{"name":"Founder"}],'
        '"panel":[{"name":"Elizabeth Yin"},{"name":"Charles Hudson"}],'
        '"transcript":"Founder: We have ten customers.\\n'
        'Charles: What is revenue?\\nFounder: Revenue is $100k.\\n'
        'Narrator: Host commentary.\\nElizabeth: I am out."}',
        encoding="utf-8",
    )
    review = {
        "episode_slug": "10-example",
        "pitch_window_decision": "Out",
        "evidence_turn_index": 4,
    }

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review=review,
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth-yin",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == (
        "Founder: We have ten customers.\n"
        "Investor question: What is revenue?\n"
        "Founder: Revenue is $100k.\n"
    )
    audit = json.loads(paths.audit.read_text())
    assert audit["status"] == "audited"
    assert audit["decision_window_endpoint"] == 4
    assert all(value is False for value in audit["leakage_checklist"].values())
    assert audit["transformation_counts"] == {
        "neutralized": 1,
        "removed": 1,
        "removed_unattributed_continuations": 0,
        "retained": 2,
    }


def test_write_pitch_package_fails_closed_on_unclassified_rendered_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        '{"founders":[{"name":"Founder"}],'
        '"panel":[{"name":"Elizabeth Yin"}],'
        '"transcript":"Founder: Revenue is $100k.\\nElizabeth: I am out."}',
        encoding="utf-8",
    )

    def leaky_renderer(**_: object) -> tuple[str, list[dict[str, object]]]:
        return (
            "Founder: Revenue is $100k.\n"
            "Investor question: I'm out.\n"
            "It sounds like the investors are nervous.\n",
            [
                {
                    "action": "retained",
                    "reason": "founder pitch",
                    "rendered_text": "Founder: Revenue is $100k.",
                    "source_turn_index": 0,
                    "speaker": "Founder",
                },
                {
                    "action": "neutralized",
                    "reason": "neutralized investor question",
                    "rendered_text": "Investor question: I'm out.",
                    "source_turn_index": 1,
                    "speaker": "Elizabeth",
                },
            ],
        )

    monkeypatch.setattr(
        "vc_clone_graph.dataset_prep.build_pitch_only_document", leaky_renderer
    )
    with pytest.raises(ValueError, match="structural provenance audit"):
        write_pitch_package(
            episode_path=episode_path,
            compiled_review={
                "episode_slug": "10-example",
                "pitch_window_decision": "Out",
                "evidence_turn_index": 1,
            },
            investor_root=tmp_path / "investor",
            vc_slug="elizabeth-yin",
            audit_source="evaluation/labels/example.json",
        )


@pytest.mark.parametrize("question", _PANEL_VERDICT_QUESTIONS)
def test_write_pitch_package_independently_audits_panel_verdict_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": name} for name in _PARITY_PANEL_NAMES],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    def leaky_renderer(**_: object) -> tuple[str, list[dict[str, object]]]:
        founder_text = "Revenue is $100k."
        return (
            "Founder: Revenue is $100k.\n"
            f"Investor question: {question}\n",
            [
                {
                    "action": "retained",
                    "attributed_source_line_index": 0,
                    "attributed_source_line_sha256": sha256(
                        founder_text.encode("utf-8")
                    ).hexdigest(),
                    "reason": "founder pitch",
                    "rendered_text": "Founder: Revenue is $100k.",
                    "retained_content_class": "explicit_founder",
                    "source_turn_index": 0,
                    "speaker": "Founder",
                    "unattributed_continuation_count": 0,
                },
                {
                    "action": "neutralized",
                    "attributed_source_line_index": 0,
                    "attributed_source_line_sha256": sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                    "reason": "neutralized investor question",
                    "rendered_text": f"Investor question: {question}",
                    "retained_content_class": "explicit_investor_question",
                    "source_turn_index": 1,
                    "speaker": "Elizabeth",
                    "unattributed_continuation_count": 0,
                },
            ],
        )

    monkeypatch.setattr(
        "vc_clone_graph.dataset_prep.build_pitch_only_document", leaky_renderer
    )
    with pytest.raises(ValueError, match="structural provenance audit"):
        write_pitch_package(
            episode_path=episode_path,
            compiled_review={
                "episode_slug": "10-example",
                "pitch_window_decision": "Out",
                "evidence_turn_index": 2,
            },
            investor_root=tmp_path / "investor",
            vc_slug="elizabeth",
            audit_source="evaluation/labels/example.json",
        )


@pytest.mark.parametrize("question", _FACTUAL_VERDICT_VOCABULARY_QUESTIONS)
def test_write_pitch_package_allows_factual_verdict_vocabulary_end_to_end(
    tmp_path: Path, question: str
) -> None:
    episode_path = tmp_path / "7-example.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": "Elizabeth"}],
                "transcript": (
                    "Founder: We have a committed round.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review={
            "episode_slug": "7-example",
            "pitch_window_decision": "Out",
            "evidence_turn_index": 2,
        },
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == (
        "Founder: We have a committed round.\n"
        f"Investor question: {question}\n"
    )


@pytest.mark.parametrize("question", _PANEL_VERDICT_QUESTIONS)
def test_write_pitch_package_removes_panel_verdict_questions_end_to_end(
    tmp_path: Path, question: str
) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [
                    {"name": name}
                    for name in ("Elizabeth", "Howie", "Sarah", "Marcus", "Phil")
                ],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review={
            "episode_slug": "10-example",
            "pitch_window_decision": "Out",
            "evidence_turn_index": 2,
        },
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == "Founder: Revenue is $100k.\n"
    audit = json.loads(paths.audit.read_text())
    assert audit["segments"][1]["action"] == "removed"


@pytest.mark.parametrize("question", _MULTIWORD_PANEL_VERDICT_QUESTIONS)
def test_write_pitch_package_removes_exact_multiword_panel_verdicts(
    tmp_path: Path, question: str
) -> None:
    episode_path = tmp_path / "10-multiword.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": name} for name in _MULTIWORD_PANEL_NAMES],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review={
            "episode_slug": "10-multiword",
            "pitch_window_decision": "Out",
            "evidence_turn_index": 2,
        },
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == "Founder: Revenue is $100k.\n"


@pytest.mark.parametrize("question", _MULTIWORD_PANEL_VERDICT_QUESTIONS)
def test_write_pitch_package_independently_audits_multiword_panel_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    episode_path = tmp_path / "10-multiword.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": name} for name in _MULTIWORD_PANEL_NAMES],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    def leaky_renderer(**_: object) -> tuple[str, list[dict[str, object]]]:
        founder_text = "Revenue is $100k."
        return (
            "Founder: Revenue is $100k.\n"
            f"Investor question: {question}\n",
            [
                {
                    "action": "retained",
                    "attributed_source_line_index": 0,
                    "attributed_source_line_sha256": sha256(
                        founder_text.encode("utf-8")
                    ).hexdigest(),
                    "reason": "founder pitch",
                    "rendered_text": "Founder: Revenue is $100k.",
                    "retained_content_class": "explicit_founder",
                    "source_turn_index": 0,
                    "speaker": "Founder",
                    "unattributed_continuation_count": 0,
                },
                {
                    "action": "neutralized",
                    "attributed_source_line_index": 0,
                    "attributed_source_line_sha256": sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                    "reason": "neutralized investor question",
                    "rendered_text": f"Investor question: {question}",
                    "retained_content_class": "explicit_investor_question",
                    "source_turn_index": 1,
                    "speaker": "Elizabeth",
                    "unattributed_continuation_count": 0,
                },
            ],
        )

    monkeypatch.setattr(
        "vc_clone_graph.dataset_prep.build_pitch_only_document", leaky_renderer
    )
    with pytest.raises(ValueError, match="structural provenance audit"):
        write_pitch_package(
            episode_path=episode_path,
            compiled_review={
                "episode_slug": "10-multiword",
                "pitch_window_decision": "Out",
                "evidence_turn_index": 2,
            },
            investor_root=tmp_path / "investor",
            vc_slug="elizabeth",
            audit_source="evaluation/labels/example.json",
        )


@pytest.mark.parametrize(
    ("question", "sanitized"), _MULTIWORD_PANEL_FACTUAL_QUESTIONS
)
def test_write_pitch_package_allows_factual_multiword_panel_questions(
    tmp_path: Path, question: str, sanitized: str
) -> None:
    episode_path = tmp_path / "10-multiword.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": name} for name in _MULTIWORD_PANEL_NAMES],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    f"Elizabeth: {question}\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review={
            "episode_slug": "10-multiword",
            "pitch_window_decision": "Out",
            "evidence_turn_index": 2,
        },
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == (
        "Founder: Revenue is $100k.\n"
        f"Investor question: {sanitized}\n"
    )


def test_write_pitch_package_audits_retained_question_not_surrounding_evaluation(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "10-question-clause.json"
    episode_path.write_text(
        json.dumps(
            {
                "founders": [{"name": "Founder"}],
                "panel": [{"name": "Elizabeth"}],
                "transcript": (
                    "Founder: Revenue is $100k.\n"
                    "Elizabeth: I like this. What is retention?\n"
                    "Elizabeth: I am out."
                ),
            }
        ),
        encoding="utf-8",
    )

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review={
            "episode_slug": "10-question-clause",
            "pitch_window_decision": "Out",
            "evidence_turn_index": 2,
        },
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == (
        "Founder: Revenue is $100k.\n"
        "Investor question: What is retention?\n"
    )


def test_write_pitch_package_uses_reviewed_founder_aliases(tmp_path) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        '{"founders":[{"name":"Jennifer Brandel"}],'
        '"panel":[{"name":"Elizabeth Yin"}],'
        '"transcript":"Jen: We connect local newsrooms.\\n'
        'Elizabeth: I am out."}',
        encoding="utf-8",
    )
    review = {
        "episode_slug": "10-example",
        "pitch_window_decision": "Out",
        "evidence_turn_index": 1,
        "founder_aliases": ["Jen"],
    }

    paths = write_pitch_package(
        episode_path=episode_path,
        compiled_review=review,
        investor_root=tmp_path / "investor",
        vc_slug="elizabeth-yin",
        audit_source="evaluation/labels/example.json",
    )

    assert paths is not None
    assert paths.pitch.read_text() == "Jen: We connect local newsrooms.\n"


def test_write_pitch_package_requires_a_retained_founder_turn(tmp_path) -> None:
    episode_path = tmp_path / "10-example.json"
    episode_path.write_text(
        '{"founders":[{"name":"Founder"}],'
        '"panel":[{"name":"Elizabeth Yin"},{"name":"Charles Hudson"}],'
        '"transcript":"Charles: What is revenue?\\nElizabeth: I am out."}',
        encoding="utf-8",
    )
    review = {
        "episode_slug": "10-example",
        "pitch_window_decision": "Out",
        "evidence_turn_index": 1,
    }

    with pytest.raises(ValueError, match="at least one founder turn"):
        write_pitch_package(
            episode_path=episode_path,
            compiled_review=review,
            investor_root=tmp_path / "investor",
            vc_slug="elizabeth-yin",
            audit_source="evaluation/labels/example.json",
        )


def test_package_relative_path_normalizes_relative_input(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    relative = __import__("pathlib").Path("evaluation/labels/audit.json")

    assert package_relative_path(relative, tmp_path) == "evaluation/labels/audit.json"
