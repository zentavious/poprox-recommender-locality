from uuid import uuid4

from lenskit.pipeline import Pipeline

from poprox_concepts.domain import Article, CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Concatenate, Fill, Interleave
from poprox_recommender.components.samplers import UniformSampler


def build_pipeline(total_slots):
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=total_slots)

    pipeline = Pipeline(name="random_concat")
    in_cand = pipeline.create_input("candidate", CandidateSet)
    in_prof = pipeline.create_input("profile", InterestProfile)
    tf = pipeline.add_component("topic-filter", topic_filter, candidate=in_cand, interest_profile=in_prof)
    pipeline.add_component("sampler", sampler, candidates1=tf, candidates2=in_cand)

    return pipeline


total_slots = 10
inputs = {
    "candidate": CandidateSet(
        articles=[Article(article_id=uuid4(), headline="headline") for _ in range(2 * total_slots)]
    ),
    "profile": InterestProfile(click_history=[], onboarding_topics=[]),
}


def test_concat_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    concat = Concatenate()

    pipeline = build_pipeline(int(total_slots / 2))
    s2 = pipeline.add_component(
        "sampler2", sampler2, candidates1=pipeline.node("topic-filter"), candidates2=pipeline.node("candidate")
    )
    pipeline.add_component("joiner", concat, recs1=pipeline.node("sampler"), recs2=s2)
    pipeline.alias("recommender", "joiner")

    outputs = pipeline.run_all(**inputs)

    expected_articles = outputs["sampler"].articles + outputs["sampler2"].articles
    expected_article_ids = [article.article_id for article in expected_articles]
    assert set(article.article_id for article in outputs["recommender"].articles) == set(expected_article_ids)


def test_interleave_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    joiner = Interleave()

    pipeline = build_pipeline(int(total_slots / 2))
    s2 = pipeline.add_component(
        "sampler2", sampler2, candidates1=pipeline.node("topic-filter"), candidates2=pipeline.node("candidate")
    )
    pipeline.add_component("joiner", joiner, recs1=pipeline.node("sampler"), recs2=s2)
    pipeline.alias("recommender", "joiner")

    outputs = pipeline.run_all(**inputs)

    recs1_article_ids = [article.article_id for article in outputs["sampler"].articles]
    recs2_article_ids = [article.article_id for article in outputs["sampler2"].articles]

    assert all([article.article_id in recs1_article_ids for article in outputs["recommender"].articles[::2]])
    assert all([article.article_id in recs2_article_ids for article in outputs["recommender"].articles[1::2]])


def test_fill_out_one_recs_list_from_another():
    sampled_slots = 7

    joiner = Fill(num_slots=total_slots, deduplicate=False)

    pipeline = build_pipeline(sampled_slots)
    pipeline.add_component("joiner", joiner, recs1=pipeline.node("sampler"), recs2=pipeline.node("candidate"))
    pipeline.alias("recommender", "joiner")

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["sampler"].articles) == sampled_slots
    assert len(outputs["recommender"].articles) == total_slots

    assert outputs["recommender"].articles[:sampled_slots] == outputs["sampler"].articles
    assert (
        outputs["recommender"].articles[sampled_slots:] == inputs["candidate"].articles[: total_slots - sampled_slots]
    )


def test_fill_removes_duplicates():
    inputs = {
        "recs": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots / 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    fill = Fill(num_slots=total_slots)

    pipeline = Pipeline(name="duplicate_concat")
    recs_input = pipeline.create_input("recs", RecommendationList)

    pipeline.add_component("fill", fill, recs1=recs_input, recs2=recs_input)
    pipeline.alias("recommender", "fill")

    outputs = pipeline.run_all(**inputs)

    article_ids = [article.article_id for article in outputs["recommender"].articles]

    assert len(outputs["recommender"].articles) == 5
    assert len(article_ids) == len(set(article_ids))
