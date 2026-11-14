# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dialect-aware SQL identifier quoting."""

from __future__ import annotations

import pytest

from auto_ontology.utils.sql_identifiers import (
    needs_quoting,
    qualified_name,
    quoted_identifier,
    quoted_if_needed,
)


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("sqlite", '"Sales Orders"'),
        ("postgres", '"Sales Orders"'),
        ("mysql", "`Sales Orders`"),
        ("spark", "`Sales Orders`"),
        (None, '"Sales Orders"'),
        ("not-a-dialect", '"Sales Orders"'),
    ],
)
def test_quoted_identifier_per_dialect(dialect: str | None, expected: str) -> None:
    # Backticks on MySQL/Spark, double quotes elsewhere; an unknown dialect
    # still quotes, since a bare identifier is what loses the table.
    assert quoted_identifier("Sales Orders", dialect) == expected


def test_qualified_name_joins_parts() -> None:
    assert qualified_name("nvapp", "raw", "t", dialect="spark") == "`nvapp`.`raw`.`t`"


def test_qualified_name_skips_empty_parts() -> None:
    # A two-level engine passes no catalog; an unqualified table passes no
    # schema. Neither may leave a stray leading dot.
    assert qualified_name(None, "raw", "t", dialect="spark") == "`raw`.`t`"
    assert qualified_name(None, None, "t", dialect="spark") == "`t`"
    assert qualified_name("", "raw", "t", dialect="spark") == "`raw`.`t`"


@pytest.mark.parametrize(
    "name",
    [
        "sales-data",
        "01a1d278-8f97-7755-8907-45833c5c34a5",
        "2024_orders",
        "Sales Orders",
        "kubeType",
        "a.b",
    ],
)
def test_names_that_cannot_be_written_bare_need_quoting(name: str) -> None:
    for dialect in ("postgres", "spark", "mysql", None, "not-a-dialect"):
        assert needs_quoting(name, dialect)


@pytest.mark.parametrize("name", ["orders", "public", "order_items", "_tmp", "t1"])
def test_plain_names_stay_bare(name: str) -> None:
    for dialect in ("postgres", "spark", "mysql", None, "not-a-dialect"):
        assert not needs_quoting(name, dialect)
        assert quoted_if_needed(name, dialect) == name


def test_reserved_words_follow_the_dialect() -> None:
    # sqlglot knows MySQL reserves ``group``; the answer is the dialect's.
    assert quoted_if_needed("group", "mysql") == "`group`"


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("postgres", '"sales-data"'),
        ("spark", "`sales-data`"),
        ("mysql", "`sales-data`"),
        ("not-a-dialect", '"sales-data"'),
    ],
)
def test_quoted_if_needed_uses_the_dialect_quote(dialect: str, expected: str) -> None:
    assert quoted_if_needed("sales-data", dialect) == expected
