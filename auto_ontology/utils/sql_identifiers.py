# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dialect-aware quoting of SQL identifiers.

Shared by the connectors (which build fully-qualified references for metadata
queries), the semantic layer's profiling probes and the text-to-SQL prompt and
validation, so all of them quote names the same way.
"""

from __future__ import annotations

from sqlglot import exp


def quoted_identifier(name: str, dialect: str | None = None) -> str:
    """Quote a catalog, schema, table or column name for *dialect*.

    Names carrying a space or a reserved word have to be quoted or the probe
    silently loses the table: ``SELECT * FROM main.Sales Orders`` parses as
    table ``main.Sales``, raises "no such table", and the caller drops every
    column of that table from profiling. The quote character is dialect-specific
    (backticks on MySQL and Spark, double quotes elsewhere), so the naive
    hard-coded ``"`` would trade a SQLite bug for a MySQL one.
    """
    try:
        return exp.to_identifier(name, quoted=True).sql(dialect=dialect or None)
    except Exception:
        # Unknown dialect: fall back to the SQL-standard quote rather than
        # emitting a bare identifier, since bare is what breaks on spaces.
        return '"' + name.replace('"', '""') + '"'


def qualified_name(*parts: str | None, dialect: str | None = None) -> str:
    """Join *parts* into a dotted, quoted reference, skipping empty parts.

    ``qualified_name("nvapp", "raw", "t", dialect="spark")`` ->
    ```nvapp`.`raw`.`t```.
    """
    return ".".join(quoted_identifier(p, dialect) for p in parts if p)


def needs_quoting(name: str, dialect: str | None = None) -> bool:
    """Whether *name* stops naming the same object when written bare.

    One rule for every catalog, schema, table and column name. Either:

    * sqlglot would quote it itself when rendering it for *dialect* -- a hyphen,
      a space, a leading digit, a word the dialect reserves. Written bare,
      ``sales-data.orders`` is a subtraction and
      ``01a1d278-8f97-7755-8907-45833c5c34a5.public.orders`` does not parse at
      all; or
    * it carries an uppercase letter, which an engine that folds unquoted
      identifiers (Postgres) would silently lowercase.

    The answer is sqlglot's rather than a pattern kept here, so it follows the
    dialect and stays in step with the parser that later validates the SQL.
    """
    if any(ch.isupper() for ch in name):
        return True
    identifier = exp.to_identifier(name)
    try:
        return identifier.sql(dialect=dialect or None) != name
    except Exception:
        # Unknown dialect: keep sqlglot's dialect-independent answer.
        return bool(identifier.quoted)


def quoted_if_needed(name: str, dialect: str | None = None) -> str:
    """Return *name* as it must be written in SQL: quoted only when it has to be.

    This is the spelling shown to the text-to-SQL model, which copies names
    verbatim, and the one the validation step restores when the model drops the
    quotes anyway. Names that are safe bare stay bare, so ordinary schemas read
    exactly as before.
    """
    return quoted_identifier(name, dialect) if needs_quoting(name, dialect) else name
