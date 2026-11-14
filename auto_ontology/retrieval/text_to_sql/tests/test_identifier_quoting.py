# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import sqlglot

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    quote_known_identifiers,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_semantic_context,
    format_tables_for_prompt,
    qualify_table,
)

_TABLES = [
    {
        "name": "events",
        "columns": [{"name": "kubeType"}, {"name": "clusterId"}],
    }
]


def test_quotes_with_backticks_on_spark() -> None:
    """Spark reads a double-quoted token as a string literal, not an identifier.

    Rendering ``ci."kubeType"`` makes the statement fail to parse outright, so
    an unmapped dialect is worse here than doing nothing.
    """
    sql = "SELECT ci.kubeType FROM lakehouse.lakehouse.events AS ci"
    quoted = quote_known_identifiers(sql, _TABLES, "spark")
    assert "`kubeType`" in quoted
    assert '"kubeType"' not in quoted


def test_quotes_with_backticks_on_databricks() -> None:
    sql = "SELECT ci.kubeType FROM main.gold.events AS ci"
    quoted = quote_known_identifiers(sql, _TABLES, "databricks")
    assert "`kubeType`" in quoted
    assert '"kubeType"' not in quoted


def test_quotes_with_double_quotes_on_postgres() -> None:
    sql = "SELECT ci.kubeType FROM public.events AS ci"
    quoted = quote_known_identifiers(sql, _TABLES, "postgres")
    assert '"kubeType"' in quoted


def test_leaves_all_lowercase_sql_untouched() -> None:
    sql = "SELECT ci.id FROM lakehouse.lakehouse.events AS ci"
    assert quote_known_identifiers(sql, _TABLES, "spark") == sql


# A database named by a UUID and a schema carrying hyphens: neither is an
# identifier when written bare, so the statement does not even parse.
_UUID_DB = "01a1d278-8f97-7755-8907-45833c5c34a5"
_HYPHEN_SCHEMA = "proxy-api-key-1794532478869-7dumi-c3b64b8d-users-public"
_QUALIFIER_CASES = [(_UUID_DB, _HYPHEN_SCHEMA), ("shop", "sales-data")]


def _orders(database: str, schema: str) -> list[dict]:
    return [
        {
            "name": "orders",
            "database_name": database,
            "schema_name": schema,
            "columns": [{"name": "amount"}, {"name": "id"}],
        }
    ]


@pytest.mark.parametrize(("database", "schema"), _QUALIFIER_CASES)
def test_quotes_bare_catalog_and_schema_so_the_sql_parses(
    database: str, schema: str
) -> None:
    sql = f"SELECT SUM(amount) AS total_revenue FROM {database}.{schema}.orders"
    with pytest.raises(sqlglot.errors.ParseError):
        sqlglot.parse_one(sql, read="postgres")

    quoted = quote_known_identifiers(sql, _orders(database, schema), "postgres")

    expected_db = f'"{database}"' if database == _UUID_DB else database
    assert quoted == (
        f'SELECT SUM(amount) AS total_revenue FROM {expected_db}."{schema}".orders'
    )
    table = sqlglot.parse_one(quoted, read="postgres").find(sqlglot.exp.Table)
    assert (table.catalog, table.db, table.name) == (database, schema, "orders")


def test_quotes_schema_qualified_and_column_qualified_references() -> None:
    sql = (
        "SELECT sales-data.orders.amount FROM sales-data.orders "
        "JOIN shop.sales-data.orders AS o ON o.id = sales-data.orders.id"
    )
    quoted = quote_known_identifiers(sql, _orders("shop", "sales-data"), "postgres")
    assert quoted == (
        'SELECT "sales-data".orders.amount FROM "sales-data".orders '
        'JOIN shop."sales-data".orders AS o ON o.id = "sales-data".orders.id'
    )


def test_database_falls_back_to_target_db() -> None:
    tables = [{"name": "orders", "schema_name": "public", "columns": []}]
    sql = f"SELECT 1 FROM {_UUID_DB}.public.orders"
    assert quote_known_identifiers(sql, tables, "postgres", _UUID_DB) == (
        f'SELECT 1 FROM "{_UUID_DB}".public.orders'
    )


def test_completes_a_partially_quoted_path() -> None:
    sql = f'SELECT 1 FROM "{_UUID_DB}".{_HYPHEN_SCHEMA}.orders'
    quoted = quote_known_identifiers(sql, _orders(_UUID_DB, _HYPHEN_SCHEMA), "postgres")
    assert quoted == f'SELECT 1 FROM "{_UUID_DB}"."{_HYPHEN_SCHEMA}".orders'


def test_already_quoted_sql_is_untouched() -> None:
    sql = f'SELECT SUM(amount) FROM "{_UUID_DB}"."{_HYPHEN_SCHEMA}".orders'
    tables = _orders(_UUID_DB, _HYPHEN_SCHEMA)
    assert quote_known_identifiers(sql, tables, "postgres") == sql


def test_string_literals_and_comments_are_untouched() -> None:
    sql = (
        "SELECT 'sales-data.orders' AS src FROM sales-data.orders "
        "-- from sales-data.orders"
    )
    quoted = quote_known_identifiers(sql, _orders("shop", "sales-data"), "postgres")
    assert quoted == (
        "SELECT 'sales-data.orders' AS src FROM \"sales-data\".orders "
        "-- from sales-data.orders"
    )


@pytest.mark.parametrize(
    "sql",
    [
        # Not the known table, only a name that starts or ends like its path.
        "SELECT 1 FROM sales-data.orders_archive",
        "SELECT 1 FROM presales-data.orders",
        # A real subtraction between two columns, not a qualifier.
        "SELECT sales-data FROM public.totals",
        # Near-miss names are never case-corrected or renamed.
        "SELECT 1 FROM Sales-Data.orders",
    ],
)
def test_never_touches_anything_but_an_exact_known_path(sql: str) -> None:
    assert quote_known_identifiers(sql, _orders("shop", "sales-data"), "postgres") == (
        sql
    )


def test_quotes_mixed_case_schema_qualifier() -> None:
    tables = [{"name": "orders", "schema_name": "Sales", "columns": []}]
    sql = "SELECT o.id FROM Sales . orders AS o"
    quoted = quote_known_identifiers(sql, tables, "postgres")
    assert quoted == 'SELECT o.id FROM "Sales".orders AS o'


def test_quotes_unsafe_qualifiers_with_backticks_on_spark() -> None:
    tables = _orders("lake-house", "sales-data")
    sql = "SELECT amount FROM lake-house.sales-data.orders"
    assert quote_known_identifiers(sql, tables, "spark") == (
        "SELECT amount FROM `lake-house`.`sales-data`.orders"
    )


def test_plain_names_are_untouched_in_every_dialect() -> None:
    sql = "SELECT amount FROM shop.public.orders"
    tables = _orders("shop", "public")
    for dialect in ("postgres", "spark", "mysql", "trino", "sqlite", None):
        assert quote_known_identifiers(sql, tables, dialect) == sql


@pytest.mark.parametrize(("database", "schema"), _QUALIFIER_CASES)
def test_prompt_shows_the_spelling_validation_restores(
    database: str, schema: str
) -> None:
    """The name the model is told to copy is the one validation produces."""
    tables = _orders(database, schema)
    shown = qualify_table(database, schema, "orders", "postgres")

    assert shown in format_tables_for_prompt(tables, dialect="postgres")
    sqlglot.parse_one(f"SELECT 1 FROM {shown}", read="postgres")

    bare = f"SELECT 1 FROM {database}.{schema}.orders"
    assert quote_known_identifiers(bare, tables, "postgres") == (
        f"SELECT 1 FROM {shown}"
    )


def test_join_path_conditions_are_quoted_for_the_dialect() -> None:
    hop = {
        "source_schema": "sales-data",
        "source_table": "orders",
        "source_column": "customerId",
        "target_schema": "sales-data",
        "target_table": "customers",
        "target_column": "id",
    }
    context = format_semantic_context(
        {"schema_name": "sales-data", "table_name": "orders", "col_name": "amount"},
        [{"path": [hop]}],
        target_db="shop",
        dialect="postgres",
    )
    assert (
        'shop."sales-data".orders."customerId" = shop."sales-data".customers.id'
        in context
    )


def test_qualify_table_without_dialect_stays_a_plain_identity_key() -> None:
    assert qualify_table("shop", "sales-data", "Orders") == "shop.sales-data.Orders"
