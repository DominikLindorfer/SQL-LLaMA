import sqlglot
sqlglot.transpile("SELECT foo( FROM bar")

import sqlglot
try:
    sqlglot.transpile("SELECT foo( FROM bar")
except sqlglot.errors.ParseError as e:
    print(e.errors)

