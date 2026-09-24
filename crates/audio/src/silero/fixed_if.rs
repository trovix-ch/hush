//! tract 0.23.8 analyses both branches of an ONNX `If` even when its condition is a known
//! constant, and requires both to produce the same output facts. The Silero export (torch
//! LSTMCell) is full of shape-guard `If`s whose dead branch is ill-typed for the live
//! shapes, so stock tract refuses the graph. This parser analyses and translates only the
//! live branch once the condition is known; tract's own declutter then inlines it.

use tract_onnx::model::{OnnxOpRegister, ParseResult, ParsingContext};
use tract_onnx::pb::NodeProto;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::*;

pub fn register(reg: &mut OnnxOpRegister) {
    reg.insert("If", parse_if);
}

fn parse_if(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let ParseResult {
        model: then_body,
        unresolved_inputs: un_then,
        ..
    } = ctx.parse_graph(node.get_attr("then_branch")?)?;
    let ParseResult {
        model: else_body,
        unresolved_inputs: un_else,
        ..
    } = ctx.parse_graph(node.get_attr("else_branch")?)?;
    let mut unresolved: Vec<String> = un_then.iter().chain(un_else.iter()).cloned().collect();
    unresolved.sort();
    unresolved.dedup();
    // Input 0 is the condition; the branches' outer inputs follow in `unresolved` order.
    let map = |names: &[String]| -> TractResult<Vec<usize>> {
        names
            .iter()
            .map(|n| {
                unresolved
                    .binary_search(n)
                    .map(|i| i + 1)
                    .map_err(|_| format_err!("If input {n} not resolved"))
            })
            .collect()
    };
    let then_input_mapping = map(&un_then)?;
    let else_input_mapping = map(&un_else)?;
    Ok((
        Box::new(FixedIf {
            then_body,
            then_input_mapping,
            else_body,
            else_input_mapping,
        }),
        unresolved,
    ))
}

#[derive(Debug, Clone)]
struct FixedIf {
    then_body: InferenceModel,
    then_input_mapping: Vec<usize>,
    else_body: InferenceModel,
    else_input_mapping: Vec<usize>,
}

impl PartialEq for FixedIf {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}
impl Eq for FixedIf {}

impl Op for FixedIf {
    fn name(&self) -> StaticName {
        "FixedIf".into()
    }
    not_a_typed_op!();
}

impl FixedIf {
    fn branch(&self, cond: bool) -> (&Vec<usize>, &InferenceModel) {
        if cond {
            (&self.then_input_mapping, &self.then_body)
        } else {
            (&self.else_input_mapping, &self.else_body)
        }
    }
}

fn known_condition(fact: &InferenceFact) -> TractResult<Option<bool>> {
    match fact.value.concretize() {
        Some(c) => Ok(Some(c.cast_to_scalar()?)),
        None => Ok(None),
    }
}

impl EvalOp for FixedIf {
    op_out_of_plan!();
    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (m, body) = self.branch(inputs[0].cast_to_scalar::<bool>()?);
        let inputs: TVec<TValue> = m.iter().map(|&ix| inputs[ix].clone()).collect();
        body.clone().into_runnable()?.run(inputs)
    }
}

impl InferenceOp for FixedIf {
    fn infer_facts(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(
        TVec<InferenceFact>,
        TVec<InferenceFact>,
        TVec<InferenceFact>,
    )> {
        let mut inputs: TVec<InferenceFact> = inputs.into_iter().cloned().collect();
        let mut outputs: TVec<InferenceFact> = outputs.into_iter().cloned().collect();
        loop {
            let mut changed = inputs[0]
                .datum_type
                .unify_with(&bool::datum_type().into())?;
            let cond = known_condition(&inputs[0])?;
            let live: &[bool] = match cond {
                Some(true) => &[true],
                Some(false) => &[false],
                None => &[true, false],
            };
            for &which in live {
                let (m, body) = if which {
                    (&self.then_input_mapping, &mut self.then_body)
                } else {
                    (&self.else_input_mapping, &mut self.else_body)
                };
                for (bix, &oix) in m.iter().enumerate() {
                    changed |= body.input_fact_mut(bix)?.unify_with_mut(&mut inputs[oix])?;
                }
                for (oix, out) in outputs.iter_mut().enumerate() {
                    let f = body.output_fact_mut(oix)?;
                    if cond.is_some() {
                        changed |= f.unify_with_mut(out)?;
                    } else {
                        changed |= f.shape.unify_with_mut(&mut out.shape)?;
                        changed |= f.datum_type.unify_with_mut(&mut out.datum_type)?;
                    }
                }
                changed |= body.analyse(false)?;
            }
            if !changed {
                return Ok((inputs, outputs, observed.into_iter().cloned().collect()));
            }
        }
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.then_body.outputs.len())
    }

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs: TVec<_> = node.inputs.iter().map(|o| mapping[o]).collect();
        let op = match known_condition(source.outlet_fact(node.inputs[0])?)? {
            Some(cond) => {
                let (m, body) = self.branch(cond);
                let body = body.clone().into_typed()?;
                tract_onnx::tract_core::ops::logic::IfThenElse {
                    then_body: body.clone(),
                    else_body: body,
                    then_input_mapping: m.clone(),
                    else_input_mapping: m.clone(),
                }
            }
            None => tract_onnx::tract_core::ops::logic::IfThenElse {
                then_body: self.then_body.clone().into_typed()?,
                else_body: self.else_body.clone().into_typed()?,
                then_input_mapping: self.then_input_mapping.clone(),
                else_input_mapping: self.else_input_mapping.clone(),
            },
        };
        target.wire_node(&*node.name, op, &inputs)
    }

    as_op!();
}
