// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Translate Comet's serialized Spark data filters into delta-kernel predicates for file-level
//! data skipping.
//!
//! Translation is deliberately conservative: only an exact, well-understood subset is converted,
//! and anything else yields `None` and is simply not pushed. Data skipping is best-effort and a
//! `Filter` above the scan still enforces correctness, so dropping a filter only costs skipping —
//! never correctness. A *wrong* predicate, by contrast, could skip files that hold matching rows,
//! so each mapping below is exact (operator and literal type both checked).

use std::sync::Arc;

use datafusion_comet_proto::spark_expression::{self, expr::ExprStruct, literal::Value};
use delta_kernel::expressions::{Expression, Predicate, PredicateRef, Scalar};

/// Spark `DataTypeId` ids (see `types.proto`) used to validate that a literal's declared type
/// matches the value we read, so e.g. a date-as-int is never treated as a plain integer.
mod type_id {
    pub const BOOL: i32 = 0;
    pub const INT8: i32 = 1;
    pub const INT16: i32 = 2;
    pub const INT32: i32 = 3;
    pub const INT64: i32 = 4;
    pub const FLOAT: i32 = 5;
    pub const DOUBLE: i32 = 6;
    pub const STRING: i32 = 7;
    pub const DATE: i32 = 12;
}

/// Translate a list of AND-ed Spark data filters into a single kernel predicate, dropping any
/// conjunct that is not in the supported subset. Returns `None` when nothing is translatable.
pub(crate) fn translate_filters(filters: &[spark_expression::Expr]) -> Option<PredicateRef> {
    let preds: Vec<Predicate> = filters.iter().filter_map(expr_to_predicate).collect();
    if preds.is_empty() {
        None
    } else {
        Some(Arc::new(Predicate::and_from(preds)))
    }
}

fn binary_operands(b: &spark_expression::BinaryExpr) -> Option<(Expression, Expression)> {
    let left = expr_to_expression(b.left.as_deref()?)?;
    let right = expr_to_expression(b.right.as_deref()?)?;
    Some((left, right))
}

fn expr_to_predicate(expr: &spark_expression::Expr) -> Option<Predicate> {
    match expr.expr_struct.as_ref()? {
        ExprStruct::Eq(b) => binary_operands(b).map(|(l, r)| Predicate::eq(l, r)),
        ExprStruct::Neq(b) => binary_operands(b).map(|(l, r)| Predicate::ne(l, r)),
        ExprStruct::Gt(b) => binary_operands(b).map(|(l, r)| Predicate::gt(l, r)),
        ExprStruct::GtEq(b) => binary_operands(b).map(|(l, r)| Predicate::ge(l, r)),
        ExprStruct::Lt(b) => binary_operands(b).map(|(l, r)| Predicate::lt(l, r)),
        ExprStruct::LtEq(b) => binary_operands(b).map(|(l, r)| Predicate::le(l, r)),
        ExprStruct::And(b) => {
            let left = expr_to_predicate(b.left.as_deref()?)?;
            let right = expr_to_predicate(b.right.as_deref()?)?;
            Some(Predicate::and(left, right))
        }
        ExprStruct::Or(b) => {
            let left = expr_to_predicate(b.left.as_deref()?)?;
            let right = expr_to_predicate(b.right.as_deref()?)?;
            Some(Predicate::or(left, right))
        }
        ExprStruct::Not(u) => Some(Predicate::not(expr_to_predicate(u.child.as_deref()?)?)),
        ExprStruct::IsNull(u) => Some(Predicate::is_null(expr_to_expression(u.child.as_deref()?)?)),
        ExprStruct::IsNotNull(u) => Some(Predicate::is_not_null(expr_to_expression(
            u.child.as_deref()?,
        )?)),
        _ => None,
    }
}

fn expr_to_expression(expr: &spark_expression::Expr) -> Option<Expression> {
    match expr.expr_struct.as_ref()? {
        // Comet serializes scan filters with `binding = false`, so columns arrive as unbound
        // (name) references.
        ExprStruct::Unbound(unbound) => Some(Expression::column([unbound.name.clone()])),
        ExprStruct::Literal(literal) => literal_to_scalar(literal).map(Expression::literal),
        _ => None,
    }
}

fn literal_to_scalar(literal: &spark_expression::Literal) -> Option<Scalar> {
    if literal.is_null {
        return None;
    }
    let declared = literal.datatype.as_ref().map(|d| d.type_id)?;
    match (literal.value.as_ref()?, declared) {
        (Value::BoolVal(v), type_id::BOOL) => Some(Scalar::Boolean(*v)),
        (Value::ByteVal(v), type_id::INT8) => Some(Scalar::Byte(*v as i8)),
        (Value::ShortVal(v), type_id::INT16) => Some(Scalar::Short(*v as i16)),
        (Value::IntVal(v), type_id::INT32) => Some(Scalar::Integer(*v)),
        (Value::IntVal(v), type_id::DATE) => Some(Scalar::Date(*v)),
        (Value::LongVal(v), type_id::INT64) => Some(Scalar::Long(*v)),
        (Value::FloatVal(v), type_id::FLOAT) => Some(Scalar::Float(*v)),
        (Value::DoubleVal(v), type_id::DOUBLE) => Some(Scalar::Double(*v)),
        (Value::StringVal(v), type_id::STRING) => Some(Scalar::String(v.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_comet_proto::spark_expression::{
        BinaryExpr, DataType, Expr, Literal, UnaryExpr, UnboundReference,
    };

    fn expr(s: ExprStruct) -> Expr {
        Expr {
            expr_struct: Some(s),
            ..Default::default()
        }
    }

    fn col(name: &str) -> Expr {
        expr(ExprStruct::Unbound(UnboundReference {
            name: name.to_string(),
            datatype: None,
        }))
    }

    fn lit_int(v: i32) -> Expr {
        lit_int_typed(v, type_id::INT32)
    }

    fn lit_int_typed(v: i32, type_id: i32) -> Expr {
        expr(ExprStruct::Literal(Literal {
            value: Some(Value::IntVal(v)),
            datatype: Some(DataType {
                type_id,
                type_info: None,
            }),
            is_null: false,
        }))
    }

    fn binary(left: Expr, right: Expr) -> Box<BinaryExpr> {
        Box::new(BinaryExpr {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        })
    }

    fn unary(child: Expr) -> Box<UnaryExpr> {
        Box::new(UnaryExpr {
            child: Some(Box::new(child)),
        })
    }

    #[test]
    fn translates_simple_comparison() {
        let gt = expr(ExprStruct::Gt(binary(col("id"), lit_int(10))));
        let pred = translate_filters(&[gt]).expect("should translate");
        // id > 10
        assert!(matches!(pred.as_ref(), Predicate::Binary(_)));
    }

    #[test]
    fn translates_and_of_two_comparisons() {
        let lt = expr(ExprStruct::Lt(binary(col("id"), lit_int(100))));
        let gt = expr(ExprStruct::Gt(binary(col("id"), lit_int(0))));
        let and = expr(ExprStruct::And(binary(lt, gt)));
        assert!(translate_filters(&[and]).is_some());
    }

    #[test]
    fn translates_is_null_and_is_not_null() {
        let isnull = expr(ExprStruct::IsNull(unary(col("id"))));
        let isnotnull = expr(ExprStruct::IsNotNull(unary(col("score"))));
        assert!(translate_filters(&[isnull, isnotnull]).is_some());
    }

    #[test]
    fn drops_unsupported_filter_but_keeps_supported_ones() {
        // A `like` filter (unsupported) plus a supported comparison: only the comparison is pushed.
        let like = expr(ExprStruct::Like(binary(col("name"), lit_int(1))));
        let gt = expr(ExprStruct::Gt(binary(col("id"), lit_int(5))));
        assert!(translate_filters(&[like.clone(), gt]).is_some());
        // The unsupported filter on its own yields nothing.
        assert!(translate_filters(&[like]).is_none());
    }

    #[test]
    fn rejects_literal_whose_declared_type_mismatches_the_value() {
        // An int value declared as TIMESTAMP (id 9) must not be treated as a plain integer.
        let bad = expr(ExprStruct::Gt(binary(col("ts"), lit_int_typed(10, 9))));
        assert!(translate_filters(&[bad]).is_none());
        // A date-typed int, however, maps to Scalar::Date.
        let date = expr(ExprStruct::Gt(binary(
            col("d"),
            lit_int_typed(18262, type_id::DATE),
        )));
        assert!(translate_filters(&[date]).is_some());
    }

    #[test]
    fn empty_filters_translate_to_none() {
        assert!(translate_filters(&[]).is_none());
    }
}
