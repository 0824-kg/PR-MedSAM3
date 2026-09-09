import random
import re
from pathlib import Path

import yaml


class PromptConfigError(ValueError):
    """Raised when prompt configuration is missing or inconsistent."""


def normalize_prompt_text(text, normalization_cfg=None):
    """Normalize prompt text using a small, configurable rule set."""
    if text is None:
        return ""

    normalized = str(text)
    normalization_cfg = normalization_cfg or {}

    if normalization_cfg.get("strip", True):
        normalized = normalized.strip()
    if normalization_cfg.get("replace_underscores_with_space", True):
        normalized = normalized.replace("_", " ")
    if normalization_cfg.get("replace_hyphens_with_space", False):
        normalized = normalized.replace("-", " ")
    if normalization_cfg.get("collapse_whitespace", True):
        normalized = re.sub(r"\s+", " ", normalized)
    if normalization_cfg.get("lowercase", True):
        normalized = normalized.lower()

    return normalized


class PromptManager:
    """Loads canonical->synonym mappings and samples normalized synonym prompts."""

    def __init__(self, prompt_config_path, dataset_key, warning_fn=None):
        self.prompt_config_path = Path(prompt_config_path)
        self.dataset_key = dataset_key
        self.warning_fn = warning_fn or print
        self.warned_missing_categories = set()

        with open(self.prompt_config_path, "r", encoding="utf-8") as f:
            prompt_cfg = yaml.safe_load(f) or {}

        self.normalization_cfg = prompt_cfg.get("normalization", {})
        dataset_cfg = (prompt_cfg.get("datasets") or {}).get(dataset_key)
        if dataset_cfg is None:
            raise PromptConfigError(
                f"Dataset key '{dataset_key}' not found in prompt config: {self.prompt_config_path}"
            )

        raw_synonyms = dataset_cfg.get("synonyms", {})
        self.synonyms_by_canonical = {}

        for canonical_prompt, synonym_list in raw_synonyms.items():
            canonical_normalized = self.normalize(canonical_prompt)
            if not canonical_normalized:
                continue

            cleaned_synonyms = []
            for synonym in synonym_list or []:
                synonym_normalized = self.normalize(synonym)
                if not synonym_normalized or synonym_normalized == canonical_normalized:
                    continue
                if synonym_normalized not in cleaned_synonyms:
                    cleaned_synonyms.append(synonym_normalized)

            self.synonyms_by_canonical[canonical_normalized] = cleaned_synonyms

    def normalize(self, text):
        return normalize_prompt_text(text, self.normalization_cfg)

    def get_canonical_prompt(self, category_name):
        canonical_prompt = self.normalize(category_name)
        if not canonical_prompt:
            raise PromptConfigError(f"Invalid canonical category name: {category_name!r}")
        return canonical_prompt

    def has_synonym(self, category_name):
        canonical_prompt = self.get_canonical_prompt(category_name)
        return canonical_prompt in self.synonyms_by_canonical and bool(
            self.synonyms_by_canonical[canonical_prompt]
        )

    def sample_synonym_prompt(self, category_name, allow_fallback=True):
        canonical_prompt = self.get_canonical_prompt(category_name)
        synonym_candidates = self.synonyms_by_canonical.get(canonical_prompt, [])
        if not synonym_candidates:
            if allow_fallback:
                self.warn_missing_category(category_name)
                return canonical_prompt
            raise PromptConfigError(
                f"No synonym prompt configured for canonical prompt '{canonical_prompt}' "
                f"in dataset '{self.dataset_key}'."
            )
        return random.choice(synonym_candidates)

    def warn_missing_category(self, category_name):
        canonical_prompt = self.get_canonical_prompt(category_name)
        if canonical_prompt in self.warned_missing_categories:
            return

        self.warned_missing_categories.add(canonical_prompt)
        self.warning_fn(
            f"[Prompt Warning] Missing synonym config for category '{canonical_prompt}' "
            f"in dataset '{self.dataset_key}'. Falling back to canonical prompt only."
        )

    def get_category_report(self, category_names):
        canonical_categories = []
        missing = []
        for category_name in category_names:
            canonical_prompt = self.get_canonical_prompt(category_name)
            canonical_categories.append(canonical_prompt)
            if canonical_prompt not in self.synonyms_by_canonical or not self.synonyms_by_canonical[canonical_prompt]:
                missing.append(canonical_prompt)

        canonical_categories = sorted(set(canonical_categories))
        missing = sorted(set(missing))

        return {
            "dataset_key": self.dataset_key,
            "canonical_categories": canonical_categories,
            "missing_categories": missing,
        }

    def warn_for_missing_categories(self, category_names):
        report = self.get_category_report(category_names)
        for category_name in report["missing_categories"]:
            self.warn_missing_category(category_name)
        return report

    def sample_prompt_examples(self, category_names, num_examples=3):
        canonical_categories = self.get_category_report(category_names)["canonical_categories"]
        if not canonical_categories:
            return []

        sample_size = min(num_examples, len(canonical_categories))
        selected = random.sample(canonical_categories, sample_size)

        examples = []
        for canonical_prompt in selected:
            synonym_prompt = self.sample_synonym_prompt(canonical_prompt, allow_fallback=True)
            examples.append((canonical_prompt, synonym_prompt))
        return examples
