"""Offline tests for the provider layer: planning, parsing, validation, secrecy."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import providers as P  # noqa: E402


class FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def uniform(criteria: dict[str, str]) -> dict[str, float]:
    return {option: 1.0 / len(criteria) for option in criteria}


class PlanTests(unittest.TestCase):
    def test_every_cell_is_asked_exactly_once(self) -> None:
        for representation, per_cell in (("palette", 1), ("hsl", 3), ("rgb", 12), ("silhouette", 1)):
            plan = P.build_jev_plan("a lighthouse at dusk", representation, 8)
            seen: dict[int, int] = {}
            for question in plan.questions:
                seen[question.cell] = seen.get(question.cell, 0) + 1
            self.assertEqual(len(seen), 64, representation)
            self.assertTrue(all(count == per_cell for count in seen.values()), representation)

    def test_batches_never_split_a_cells_channels(self) -> None:
        plan = P.build_jev_plan("red sails", "rgb", 24)
        home: dict[str, int] = {}
        for index, batch in enumerate(plan.batches):
            for question in batch:
                if question.qid in home:
                    self.fail("a question appears in two batches")
                home[question.qid] = index
        cell_batch: dict[int, int] = {}
        for index, batch in enumerate(plan.batches):
            for question in batch:
                first = cell_batch.setdefault(question.cell, index)
                self.assertEqual(first, index, "cell channels were split across batches")
        self.assertTrue(all(len(batch) <= P.MAX_QUESTIONS_PER_CALL for batch in plan.batches))

    def test_unknown_inputs_are_rejected(self) -> None:
        with self.assertRaises(P.ProviderError):
            P.build_jev_plan("x", "watercolour", 16)
        with self.assertRaises(P.ProviderError):
            P.build_jev_plan("x", "palette", 13)


class ProbabilityTests(unittest.TestCase):
    def test_accepts_maps_and_lists(self) -> None:
        self.assertEqual(P._probability_map({"a": 0.25, "b": 0.75}, "q0"), {"a": 0.25, "b": 0.75})
        listed = P._probability_map(
            [{"option": "a", "probability": 1}, {"name": "b", "p": 0}], "q1"
        )
        self.assertEqual(listed, {"a": 1.0, "b": 0.0})

    def test_rejects_broken_distributions(self) -> None:
        for bad in (None, {}, {"a": -1}, {"a": "warm"}, {"a": float("nan")}, []):
            with self.assertRaises(P.ProviderError):
                P._probability_map(bad, "q0")


class JevProviderTests(unittest.TestCase):
    def generate(self, representation: str, size: int = 8) -> dict:
        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            body = json.loads(request.data.decode("utf-8"))
            answers = {
                qid: {"probabilities": uniform(q["criteria"])}
                for qid, q in body["questions"].items()
            }
            self.assertEqual(body["model"], "jev-latest")
            self.assertIn("painting", body["state"])
            self.assertEqual(request.get_header("Authorization"), "Bearer sk-secret")
            return FakeStream(json.dumps({"answers": answers}).encode("utf-8"))

        provider = P.JevProvider("sk-secret")
        with patch.object(P, "_open", opener):
            return provider.generate("a painting of tulips", representation, size)

    def test_fields_are_well_formed(self) -> None:
        for representation in P.REPRESENTATIONS:
            field = self.generate(representation)
            self.assertEqual(field["width"], 8)
            self.assertEqual(len(field["cells"]), 64)
            for candidates in field["cells"]:
                self.assertLessEqual(len(candidates), 4)
                total = sum(pigment[3] for pigment in candidates)
                self.assertAlmostEqual(total, 1.0, places=6)

    def test_missing_answers_fail_loudly(self) -> None:
        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            body = json.loads(request.data.decode("utf-8"))
            first = next(iter(body["questions"]))
            return FakeStream(json.dumps({"answers": {first: {"probabilities": {"s0": 1}}}}).encode("utf-8"))

        with patch.object(P, "_open", opener):
            with self.assertRaises(P.ProviderError):
                P.JevProvider("sk-secret").generate("tulips", "silhouette", 8)

    def test_invalid_probability_fails_loudly(self) -> None:
        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            body = json.loads(request.data.decode("utf-8"))
            answers = {
                qid: {"probabilities": {"subject": 0.4, "ground": "plenty"}}
                for qid in body["questions"]
            }
            return FakeStream(json.dumps({"answers": answers}).encode("utf-8"))

        with patch.object(P, "_open", opener):
            with self.assertRaises(P.ProviderError):
                P.JevProvider("sk-secret").generate("tulips", "silhouette", 8)

    def test_upstream_http_error_becomes_provider_error(self) -> None:
        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            raise HTTPError("https://example.test", 429, "slow down", {}, None)

        with patch.object(P, "_open", opener):
            with self.assertRaises(P.ProviderError):
                P.JevProvider("sk-secret").generate("tulips", "palette", 8)

    def test_key_never_reaches_the_field(self) -> None:
        field = self.generate("palette")
        self.assertNotIn("sk-secret", json.dumps(field))


class GridParsingTests(unittest.TestCase):
    def test_reads_fenced_and_bare_json(self) -> None:
        grid = '{"palette": ["#112233", "#aabbcc"], "grid": ["01", "10"]}'
        for text in (grid, f"Here you go:\n```json\n{grid}\n```\nEnjoy!", f"sure {grid} done"):
            palette, rows = P.parse_grid_json(text, 2)
            self.assertEqual(palette[0], (0x11, 0x22, 0x33))
            self.assertEqual(rows, [[0, 1], [1, 0]])

    def test_resamples_a_small_answer(self) -> None:
        text = '{"palette": ["#000000", "#ffffff"], "grid": ["01", "10"]}'
        _, rows = P.parse_grid_json(text, 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == 4 for row in rows))

    def test_rejects_unusable_answers(self) -> None:
        for bad in (
            "no json here",
            '{"palette": ["#000000"], "grid": []}',
            '{"palette": ["not-a-color"], "grid": ["0"]}',
            '{"palette": ["#000000"], "grid": ["5"]}',
            '{"grid": ["0"]}',
        ):
            with self.assertRaises(P.ProviderError):
                P.parse_grid_json(bad, 4)


class ChatProviderTests(unittest.TestCase):
    def generate(self, representation: str) -> dict:
        reply = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"palette": ["#203040", "#c0b090"], '
                            '"grid": ["01010101", "10101010", "01010101", "10101010", '
                            '"01010101", "10101010", "01010101", "10101010"]}\n'
                            "```"
                        )
                    }
                }
            ]
        }

        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "gpt-4o-mini")
            self.assertNotIn("sk-secret", json.dumps(body))
            return FakeStream(json.dumps(reply).encode("utf-8"))

        provider = P.ChatProvider("sk-secret")
        with patch.object(P, "_open", opener):
            return provider.generate("a barn in snow", representation, 8)

    def test_fields_are_well_formed(self) -> None:
        for representation in P.REPRESENTATIONS:
            field = self.generate(representation)
            self.assertEqual(len(field["cells"]), 64)
            self.assertEqual(field["meta"]["provider"], "chat")

    def test_silhouette_is_two_tone(self) -> None:
        field = self.generate("silhouette")
        for candidates in field["cells"]:
            colours = {tuple(pigment[:3]) for pigment in candidates}
            self.assertLessEqual(len(colours), 2)

    def test_strict_json_falls_back_when_refused(self) -> None:
        attempts: list[dict] = []

        def opener(request: Request, timeout: int | None = None) -> FakeStream:
            body = json.loads(request.data.decode("utf-8"))
            attempts.append(body)
            if "response_format" in body:
                raise HTTPError("https://example.test", 400, "no schema", {}, None)
            return FakeStream(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"palette": ["#010203"], "grid": ["00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000"]}'
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        with patch.object(P, "_open", opener):
            field = P.ChatProvider("sk-secret").generate("a barn", "palette", 8)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(field["width"], 8)


class SelectionTests(unittest.TestCase):
    def test_env_selection(self) -> None:
        jev = {"JEV_API_KEY": "sk-secret"}
        chat = {"OPENAI_API_KEY": "sk-secret", "OPENAI_BASE_URL": "http://localhost:9/v1"}
        self.assertIsInstance(P.provider_from_env(jev), P.JevProvider)
        self.assertIsInstance(P.provider_from_env(chat), P.ChatProvider)
        self.assertIsInstance(P.provider_from_env({**jev, **chat}), P.JevProvider)
        self.assertIsInstance(P.provider_from_env({**jev, **chat, "PAINT_PROVIDER": "chat"}), P.ChatProvider)
        self.assertIsNone(P.provider_from_env({}))
        self.assertIsNone(P.provider_from_env({"PAINT_PROVIDER": "jev"}))
        self.assertIsInstance(P.provider_from_env({"AI_API_KEY": "sk-secret"}), P.ChatProvider)

    def test_provider_objects_do_not_leak_keys_in_repr(self) -> None:
        provider = P.provider_from_env({"JEV_API_KEY": "sk-secret"})
        self.assertNotIn("sk-secret", repr(provider))
        self.assertNotIn("sk-secret", repr(P.provider_from_env({"AI_API_KEY": "sk-secret"})))


if __name__ == "__main__":
    unittest.main()
