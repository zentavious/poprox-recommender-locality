import asyncio
import json
import logging
from datetime import datetime, timedelta

import numpy as np
from lenskit.pipeline import Component
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts.domain import (
    Article,
    CandidateSet,
    InterestProfile,
    RecommendationList,
)
from poprox_recommender.components.diversifiers.locality_calibration import (
    LocalityCalibrator,
)
from poprox_recommender.paths import model_file_path

MAX_RETRIES = 3
DELAY = 2
SEMANTIC_THRESHOLD = 0.2
BASELINE_THETA_TOPIC = 0.3
NUM_TOPICS = 3
DAYS = 14

logger = logging.getLogger(__name__)


event_system_prompt = (
    "You are an Associated Press editor tasked to rewrite a news preview in a factual tone. "
    "You are provided with [[MAIN_NEWS]] and [[RELATED_NEWS]] each with a HEADLINE, SUB_HEADLINE, and BODY. "
    "The [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE should be rewritten using the following rules. "
    "Rules: "
    "1. ***Explicitly*** integrate key themes or implications from the [[RELATED_NEWS]] HEADLINE, SUB_HEADLINE, "
    "and BODY to rewrite the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE. The connection should be meaningful, "
    "not just mentioned in passing. "
    "2. Reframe an element of the [[MAIN_NEWS]] BODY in the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE to emphasize a "
    "natural progression, contrast, or deeper context between the [[MAIN_NEWS]] BODY the [[RELATED_NEWS]] BODY. "
    "Highlight how the [[MAIN_NEWS]] BODY builds on, challenges, or expands reader's prior understanding. "
    "3. Avoid minimal rewording of the original [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE—introduce a fresh angle that "
    "makes the connection to [[RELATED_NEWS]] feel insightful and engaging. "
    "4. The rewritten article should have a HEADLINE and SUB_HEADLINE. "
    "5. The rewritten SUB_HEADLINE should NOT end in punctuation. "
    "6. The rewrriten HEADLINE should be approximately the same length as the [[MAIN_NEWS]] HEADLINE. "
    "7. The rewrriten SUB_HEADLINE should be approximately the same length as the [[MAIN_NEWS]] SUB_HEADLINE. "
    "8. Ensure the rewritten article is neutral and accurately describes the [[MAIN_NEWS]] BODY. "
    "9. Your response should only include JSON parseable by json.loads() in the form "
    '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'.'
)

topic_system_prompt = (
    "You are an Associated Press editor tasked to rewrite a news preview in a factual tone. "
    "You are provided with a list of a user's broad [[INTERESTED_TOPICS]] and [[MAIN_NEWS]] with a "
    "HEADLINE, SUB_HEADLINE, and BODY. "
    "The [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE should be rewritten using the following rules. "
    "Rules: "
    "1. ***Explicitly*** integrate one or more of the user's prior reading habbits from [[INTERESTED_TOPICS]] to "
    "rewrite the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE in a way that naturally reshapes the focus."
    "2. Reframe an element of the [[MAIN_NEWS]] BODY in the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE to emphasize an "
    "angle or passage that directly appeals to the user's [[INTERESTED_TOPICS]] to make this news particularly "
    "relevant. Highlight an unexpected connection or unique insight between [[MAIN_NEWS]] BODY and "
    "the user's broad [[INTERESTED_TOPICS]]. "
    "3. Avoid minimal rewording of the original [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE—introduce a fresh angle on the "
    "[[MAIN_NEWS]] BODY that makes the connection to [[INTERESTED_TOPICS]] feel insightful and engaging. "
    "4. The rewritten article should have a HEADLINE and SUB_HEADLINE. "
    "5. The rewritten SUB_HEADLINE should NOT end in punctuation. "
    "6. The rewrriten HEADLINE should be approximately the same length as the [[MAIN_NEWS]] HEADLINE. "
    "7. The rewrriten SUB_HEADLINE should be approximately the same length as the [[MAIN_NEWS]] SUB_HEADLINE. "
    "8. Ensure the rewritten article is neutral and accurately describes the [[MAIN_NEWS]] BODY. "
    "9. Your response should only include JSON parseable by json.loads() in the form "
    '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'.'
)

refine_system_prompt = (
    "You are an Associated Press editor tasked with refining rewritten news previews. "
    "Each preview consists of a HEADLINE and a SUB_HEADLINE. Your goal is to ensure that HEADLINEs and SUB_HEADLINEs "
    "in different previews do not rely on the same words or strategies to emphasize key topics or related news. "
    "If a HEADLINE and SUB_HEADLINE in a preview are too similar to others in how they highlight key points, rewrite "
    "them to introduce variation while preserving their meaning. However, if a preview already uses a distinct "
    "approach from others, do not modify it and return it unchanged. Changes should not alter the meaning of the "
    "HEADLINE or SUB_HEADLINE. Your response must include the same number of rewritten news previews as the input, "
    "keeping unmodified previews intact. The rewritten SUB_HEADLINE should NOT end in punctuation. "
    "Return only JSON parseable by json.loads(), in the format: "
    '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\''
)


class ContextGenerator(Component):
    def __init__(self, text_generation=False, time_decay=True, dev_mode="true"):
        self.text_generation = text_generation
        self.time_decay = time_decay
        self.dev_mode = dev_mode
        self.previous_context_articles = []
        if self.dev_mode:
            logger.info("Dev_mode is true, using live OpenAI client...")
            self.client = AsyncOpenAI(api_key="Insert your key here.")
            logger.info("Successfully instantiated OpenAI client...")
        self.model = SentenceTransformer(str(model_file_path("all-MiniLM-L6-v2")))

    def __call__(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ) -> RecommendationList:
        extras = []
        if self.dev_mode:
            # selected = self.generate_newsletter(clicked, selected, interest_profile)
            selected, extras = asyncio.run(self.generate_newsletter(clicked, selected, interest_profile))
        logger.error(f"Final extras: {extras}")
        return RecommendationList(articles=selected.articles, extras=extras)

    async def generate_newsletter(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ):
        topic_distribution = LocalityCalibrator.compute_topic_prefs(interest_profile)
        top_topics = []
        if topic_distribution:
            topic_distribution.pop("U.S. news", None)
            topic_distribution.pop("World news", None)
            sorted_topics = sorted(topic_distribution.items(), key=lambda item: item[1], reverse=True)
            top_topics = [(key, count) for key, count in sorted_topics[:NUM_TOPICS]]

        treated_articles = []
        tasks = []
        extras = [{} for _ in range(len(selected.articles))]

        for i in range(len(selected.articles)):
            article = selected.articles[i]
            if selected.treatment_flags[i]:
                task = self.generate_treatment_preview(article, clicked, self.time_decay, top_topics, extras[i])
                tasks.append((article, task))

        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)

        for (article, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error generating context for article: {result}")
            else:
                article.headline, article.subhead = result  # type: ignore
                treated_articles.append(article)

        await self.diversify_treatment_previews(treated_articles)

        return selected, extras

    async def generate_treatment_preview(
        self, article: Article, clicked_articles: CandidateSet, time_decay: bool, top_topics: list, extra_logging: dict
    ):
        related_article = self.related_context(article, clicked_articles, time_decay, extra_logging)

        if related_article is not None:
            # high similarity, use the top-1 article to rewrite the rec
            article_prompt = f"""
[[MAIN_NEWS]]
    HEADLINE: {article.headline}
    SUB_HEADLINE: {article.subhead}
    BODY_TEXT: {article.body}
[[RELATED_NEWS]]
    HEADLINE: {related_article.headline}
    SUB_HEADLINE: {related_article.subhead}
    BODY_TEXT: {related_article.body}
"""  # noqa: E501

            logger.info(
                f"Generating event-level narrative for '{article.headline[:30]}' from related article '{related_article.headline[:15]}'"  # noqa: E501
            )
            logger.info(f"Using prompt: {article_prompt}")
            extra_logging["prompt_level"] = "event"
            rec_headline, rec_subheadline = await self.async_gpt_generate(event_system_prompt, article_prompt)
        else:
            if top_topics:
                article_prompt = f"""
[[MAIN_NEWS]]
    HEADLINE: {article.headline}
    SUB_HEADLINE: {article.subhead}
    BODY_TEXT: {article.body}
[[INTERESTED_TOPICS]]: {[top_count_pair[0] for top_count_pair in top_topics]}
"""  # noqa: E501

                logger.info(f"Generating topic-level narrative for related article: {article.headline[:30]}")
                logger.info(f"Using prompt: {article_prompt}")
                extra_logging["prompt_level"] = "topic"
                for ind, top_count_pair in enumerate(top_topics):
                    extra_logging["top_{}_topic".format(ind)] = top_count_pair[0]
                    extra_logging["top_{}_topic_ratio".format(ind)] = float(top_count_pair[1])
                rec_headline, rec_subheadline = await self.async_gpt_generate(topic_system_prompt, article_prompt)
            else:
                logger.warning(
                    f"No topic_distribution for generating high-level narrative for {article.headline[:30]}. Falling back to original preview..."  # noqa: E501
                )
                extra_logging["prompt_level"] = "none"
                rec_headline, rec_subheadline = article.headline, article.subhead
        return rec_headline, rec_subheadline

    async def diversify_treatment_previews(
        self,
        articles: list[Article],
    ):
        input_prompt = []
        for i, article in enumerate(articles):
            article_prompt = f"""
                HEADLINE: {article.headline}
                SUB_HEADLINE: {article.subhead}
            """  # noqa: E501
            input_prompt.append(article_prompt)

        try:
            rewritten_previews = await self.async_gpt_diversify(refine_system_prompt, input_prompt, len(input_prompt))
        except Exception as e:
            logger.error(f"Error in call to OPENAI API: {e}. Falling back to all original preview...")
            return articles

        for i, rewritten_preview in enumerate(rewritten_previews):
            articles[i].headline = rewritten_preview["HEADLINE"]
            articles[i].subhead = rewritten_preview["SUB_HEADLINE"]

        return articles

    def related_context(self, article: Article, clicked: CandidateSet, time_decay: bool, extra_logging: dict):
        selected_subhead = article.subhead
        selected_date = article.published_at

        clicked_articles = clicked.articles
        time0 = selected_date - timedelta(days=DAYS)

        clicked_articles = [
            article
            for article in clicked_articles
            if article.published_at >= time0 and article not in self.previous_context_articles
        ]
        candidate_indices = self.related_indices(
            selected_subhead, selected_date, clicked_articles, time_decay, extra_logging
        )
        if len(candidate_indices) == 0:
            return None

        self.previous_context_articles.append(clicked_articles[candidate_indices[0]])
        return clicked_articles[candidate_indices[0]]

    def related_indices(
        self,
        selected_subhead: str,
        selected_date: datetime,
        clicked_articles: list,
        time_decay: bool,
        extra_logging: dict,
    ):
        all_subheads = [selected_subhead] + [article.subhead for article in clicked_articles]
        embeddings = self.model.encode(all_subheads)

        target_embedding = embeddings[0].reshape(1, -1)
        clicked_embeddings = embeddings[1:]
        if len(clicked_embeddings) != 0:
            similarities = cosine_similarity(target_embedding, clicked_embeddings)[0]
        else:
            return []

        # CHECK threshold [0.2, 0, 0.2]
        for i in range(len(similarities)):
            val = similarities[i]
            if val < SEMANTIC_THRESHOLD:
                similarities[i] = 0

        most_sim_article_ind = np.argmax(similarities)
        highest_sim = float(similarities[most_sim_article_ind])
        extra_logging["similarity"] = highest_sim
        extra_logging["context_article"] = str(clicked_articles[most_sim_article_ind].article_id)
        if highest_sim < SEMANTIC_THRESHOLD:
            return []

        elif time_decay:
            weights = [
                self.get_time_weight(selected_date, published_date)
                for published_date in [article.published_at for article in clicked_articles]
            ]
            weighted_similarities = similarities * weights

            selected_indices = np.argsort(weighted_similarities)[-1:]
            return selected_indices

        else:
            selected_indices = np.argsort(similarities)[-1:]
            return selected_indices

    def get_time_weight(self, published_target, published_clicked):
        time_distance = abs((published_clicked - published_target).days)
        weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
        return weight

    def rewritten_previews_feedback(self, rewritten_previews, expected_output_n):
        if not isinstance(rewritten_previews, dict) and "REWRITTEN_NEWS_PREVIEWS" not in rewritten_previews:
            logger.warning("GPT response invald and doesn't contain a list of previews. Retrying...")
            feedback = (
                "Your response isn't JSON parseable by json.loads() in the format "
                '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\'. '  # noqa: E501
                "It should include a rewritten HEADLINE and SUB_HEADLINE for each article in the list of "
                "REWRITTEN_NEWS_PREVIEWS. Ensure your response is valid JSON parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback
        elif len(rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]) != expected_output_n:
            logger.warning(
                f"GPT response invald and is missing previews {len(rewritten_previews['REWRITTEN_NEWS_PREVIEWS'])} != {expected_output_n}. Retrying..."  # noqa: E501
            )
            feedback = (
                f"Your response JSON of 'REWRITTEN_NEWS_PREVIEWS' doesn't include all {expected_output_n} "
                "rewritten articles. Ensure your response is valid JSON parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback
        for item in rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]:
            if not isinstance(item, dict) or set(item.keys()) != {"HEADLINE", "SUB_HEADLINE"}:
                logger.warning(f"GPT response invald for {item}. Retrying...")
                feedback = (
                    "Your response includes one or more articles not in the format "
                    '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '
                    f"Ensure all {expected_output_n} rewritten articles contain both a HEADLINE and SUB_HEADLINE "
                    "and are included in JSON parseable by json.loads() in the format "
                    '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\'. '  # noqa: E501
                )
                return feedback

        return False

    def rewritten_preview_feedback(self, rewritten_preview):
        if not isinstance(rewritten_preview, dict) and (
            "HEADLINE" not in rewritten_preview or "SUB_HEADLINE" not in rewritten_preview
        ):
            logger.warning(f"GPT response invald for {rewritten_preview}. Retrying...")
            feedback = (
                "Your response is not in the format "
                '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '
                "Ensure the rewritten article contains both a HEADLINE and SUB_HEADLINE "
                "and is valid JSON parseable by json.loads()."
            )
            return feedback
        return False

    async def async_gpt_generate(self, system_prompt, content_prompt):
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": content_prompt}]},
        ]

        temperature = 0.2
        max_tokens = 2000
        frequency_penalty = 0.0
        chat_completion = await self.client.beta.chat.completions.parse(
            messages=message,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model="gpt-4o-mini",
        )
        logger.info(f"GPT response: {chat_completion.choices[0].message.content}")

        rewritten_preview = json.loads(chat_completion.choices[0].message.content)
        feedback = self.rewritten_preview_feedback(rewritten_preview)
        if feedback:
            logger.warning(f"GPT response invalid. Retrying with feedback '{feedback}'")
            reprompt_message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": content_prompt}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": chat_completion.choices[0].message.content}],
                },
                {"role": "user", "content": [{"type": "text", "text": feedback}]},
            ]

            chat_completion = await self.client.beta.chat.completions.parse(
                messages=reprompt_message,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o-mini",
            )

            logger.info(f"GPT reprompt response: {chat_completion.choices[0].message.content}")
            rewritten_preview = json.loads(chat_completion.choices[0].message.content)

            feedback = self.rewritten_preview_feedback(rewritten_preview)
            if feedback:
                raise ValueError(f"GPT response still invalid. Failing from feedback '{feedback}'")

        return (
            rewritten_preview["HEADLINE"],
            rewritten_preview["SUB_HEADLINE"],
        )

    async def async_gpt_diversify(self, system_prompt, content_prompt, expected_output_n):
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": rewritten_news} for rewritten_news in content_prompt],
            },
        ]
        temperature = 0.2
        max_tokens = 2000
        frequency_penalty = 0.0
        chat_completion = await self.client.beta.chat.completions.parse(
            messages=message,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model="gpt-4o-mini",
        )
        logger.info(f"GPT response: {chat_completion.choices[0].message.content}")

        rewritten_previews = json.loads(chat_completion.choices[0].message.content)
        feedback = self.rewritten_previews_feedback(rewritten_previews, expected_output_n)
        if feedback:
            logger.warning(f"GPT response invalid. Retrying with feedback '{feedback}'")
            reprompt_message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": main_news} for main_news in content_prompt]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": chat_completion.choices[0].message.content}],
                },
                {"role": "user", "content": [{"type": "text", "text": feedback}]},
            ]

            chat_completion = await self.client.beta.chat.completions.parse(
                messages=reprompt_message,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o-mini",
            )

            logger.info(f"GPT reprompt response: {chat_completion.choices[0].message.content}")
            rewritten_previews = json.loads(chat_completion.choices[0].message.content)

            feedback = self.rewritten_previews_feedback(rewritten_previews, expected_output_n)
            if feedback:
                raise ValueError(f"GPT response still invalid. Failing from feedback '{feedback}'")

        return rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]
