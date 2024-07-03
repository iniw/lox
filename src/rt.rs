use std::{borrow::Cow, collections::HashMap, fmt::Display, mem::replace};

use crate::{
    syntax::{AssignmentOp, BinaryOp, Block, Expr, Literal, Stmt, UnaryOp},
    LoxNumber,
};

#[derive(Debug, Clone)]
pub struct TreeWalker<'a> {
    env: EnvManager<'a>,
}

impl<'a> TreeWalker<'a> {
    pub fn new() -> Self {
        Self {
            env: EnvManager::new(),
        }
    }

    pub fn execute(&mut self, statements: Vec<Stmt<'a>>) -> Execution<'a> {
        let mut last_statement = Stated::Nothing;
        for statement in statements {
            last_statement = self.execute_statement(statement)?;
        }
        Ok(last_statement)
    }

    fn execute_statement(&mut self, statement: Stmt<'a>) -> Execution<'a> {
        match statement {
            Stmt::Block { statements } => self.execute_block(statements),
            Stmt::Break => self.execute_break(),
            Stmt::ClassDecl {
                identifier,
                methods,
                superclass,
            } => self.execute_class_decl(identifier, methods, superclass),
            Stmt::Continue => self.execute_continue(),
            Stmt::Empty => self.execute_empty(),
            Stmt::Expr { expr } => self.execute_expression(expr),
            Stmt::FunDecl {
                identifier,
                parameters,
                body,
            } => self.execute_fun_decl(identifier, parameters, body),
            Stmt::If {
                condition,
                branch,
                else_branch,
            } => self.execute_if(condition, *branch, else_branch.map(|s| *s)),
            Stmt::Print { expr } => self.execute_print(expr),
            Stmt::Return { expr } => self.execute_return(expr),
            Stmt::VarDecl { identifier, expr } => self.execute_var_decl(identifier, expr),
            Stmt::While { condition, body } => self.execute_while(condition, *body),
        }
    }

    fn execute_block(&mut self, statements: Vec<Stmt<'a>>) -> Execution<'a> {
        self.env.push_scope();
        let result = self.perform_block(statements);
        self.env.pop_scope();
        result
    }

    fn execute_break(&self) -> Execution<'a> {
        Ok(Stated::ControlFlow(ControlFlow::Break))
    }

    fn execute_class_decl(
        &mut self,
        identifier: &'a str,
        methods: Vec<(&'a str, Vec<&'a str>, Block<'a>)>,
        _superclass: Option<&'a str>,
    ) -> Execution<'a> {
        let scope = {
            let mut scope = Scope::with_capacity(methods.len());
            for (identifier, parameters, body) in methods {
                let function = Function {
                    identifier: Some(identifier),
                    parameters,
                    body,
                    parent_env: self.env.snapshot(),
                };
                scope.insert(identifier, Value::Callable(function));
            }
            scope
        };

        let class = Class {
            identifier,
            scope,
            superclass: None,
        };

        self.env.declare_class(identifier, class);

        Ok(Stated::Nothing)
    }

    fn execute_continue(&self) -> Execution<'a> {
        Ok(Stated::ControlFlow(ControlFlow::Continue))
    }

    fn execute_empty(&self) -> Execution<'a> {
        // No-op
        Ok(Stated::Nothing)
    }

    fn execute_expression(&mut self, expr: Expr<'a>) -> Execution<'a> {
        self.evaluate_expression(expr)
            .map(|expr| Stated::Value(expr))
    }

    fn execute_fun_decl(
        &mut self,
        identifier: &'a str,
        parameters: Vec<&'a str>,
        body: Block<'a>,
    ) -> Execution<'a> {
        let function = Function {
            identifier: Some(identifier),
            parameters,
            body,
            parent_env: self.env.snapshot(),
        };

        self.env.declare_function(identifier, function);

        Ok(Stated::Nothing)
    }

    fn execute_if(
        &mut self,
        condition: Expr<'a>,
        branch: Stmt<'a>,
        else_branch: Option<Stmt<'a>>,
    ) -> Execution<'a> {
        if self.evaluate_expression(condition)?.is_truthy() {
            self.execute_statement(branch)
        } else if let Some(else_branch) = else_branch {
            self.execute_statement(else_branch)
        } else {
            Ok(Stated::Nothing)
        }
    }

    fn perform_block(&mut self, statements: Vec<Stmt<'a>>) -> Execution<'a> {
        for statement in statements {
            if let flow @ Stated::ControlFlow(_) = self.execute_statement(statement)? {
                return Ok(flow);
            }
        }
        Ok(Stated::Nothing)
    }

    fn execute_print(&mut self, expr: Expr<'a>) -> Execution<'a> {
        println!("{}", self.evaluate_expression(expr)?);
        Ok(Stated::Nothing)
    }

    fn execute_return(&mut self, expr: Option<Expr<'a>>) -> Execution<'a> {
        let value = if let Some(expr) = expr {
            self.evaluate_expression(expr)?
        } else {
            Value::Undefined
        };

        Ok(Stated::ControlFlow(ControlFlow::Return(value)))
    }

    fn execute_var_decl(&mut self, identifier: &'a str, expr: Option<Expr<'a>>) -> Execution<'a> {
        let value = expr
            .map(|expr| self.evaluate_expression(expr))
            .transpose()?;

        let value = value.unwrap_or(Value::Undefined);

        self.env.declare_symbol(identifier, value);

        Ok(Stated::Nothing)
    }

    fn execute_while(&mut self, condition: Expr<'a>, body: Stmt<'a>) -> Execution<'a> {
        while self.evaluate_expression(condition.clone())?.is_truthy() {
            let result = self.execute_statement(body.clone())?;
            if let Stated::ControlFlow(control_flow) = result {
                match control_flow {
                    ControlFlow::Break => break,
                    ControlFlow::Continue => continue,
                    ret @ ControlFlow::Return(_) => return Ok(Stated::ControlFlow(ret)),
                }
            }
        }

        Ok(Stated::Nothing)
    }

    fn evaluate_expression(&mut self, expr: Expr<'a>) -> Evaluation<'a> {
        match expr {
            Expr::Assignment {
                op,
                identifier,
                expr,
            } => self.evaluate_assignment(op, identifier, *expr),
            Expr::Binary { left, op, right } => self.evaluate_binary(*left, op, *right),
            Expr::FunctionCall { expr, arguments } => self.evaluate_function_call(*expr, arguments),
            Expr::Grouping { expr } => self.evaluate_grouping(*expr),
            Expr::Lambda { parameters, body } => self.evaluate_lambda(parameters, body),
            Expr::Literal(literal) => self.evaluate_literal(literal),
            Expr::PropertyAccess { expr, property } => {
                self.evaluate_property_access(*expr, property)
            }
            Expr::PropertyAssignment {
                object,
                property,
                op,
                value,
            } => self.evaluate_property_assignment(*object, property, op, *value),
            Expr::Symbol { identifier } => self.evaluate_symbol(identifier),
            Expr::Unary { op, expr } => self.evaluate_unary(op, *expr),
        }
    }

    fn evaluate_assignment(
        &mut self,
        op: AssignmentOp,
        identifier: &'a str,
        expr: Expr<'a>,
    ) -> Evaluation<'a> {
        let value = self.evaluate_expression(expr)?;
        if let Some(var) = self.env.find_symbol(identifier) {
            match op {
                AssignmentOp::Set => *var = value,
                AssignmentOp::Add => var.add_assign(value)?,
                AssignmentOp::Sub => var.sub_assign(value)?,
                AssignmentOp::Mul => var.mul_assign(value)?,
                AssignmentOp::Div => var.div_assign(value)?,
            }
            Ok(var.clone())
        } else {
            Err(RuntimeError::UndefinedVariable(identifier))
        }
    }

    fn evaluate_binary(&mut self, left: Expr<'a>, op: BinaryOp, right: Expr<'a>) -> Evaluation<'a> {
        match op {
            // Logical (handled separately to implement short-circuiting)
            BinaryOp::And => {
                let left = self.evaluate_expression(left)?;
                if left.is_truthy() {
                    let right = self.evaluate_expression(right)?;
                    return Ok(Value::Bool(right.is_truthy()));
                } else {
                    return Ok(Value::Bool(false));
                }
            }
            BinaryOp::Or => {
                let left = self.evaluate_expression(left)?;
                if left.is_truthy() {
                    return Ok(Value::Bool(true));
                } else {
                    let right = self.evaluate_expression(right)?;
                    return Ok(Value::Bool(right.is_truthy()));
                }
            }
            op => {
                let left = self.evaluate_expression(left)?;
                let right = self.evaluate_expression(right)?;
                match op {
                    // Relational
                    BinaryOp::Equal => left.eq(right),
                    BinaryOp::NotEqual => left.neq(right),
                    BinaryOp::GreaterThan => left.gt(right),
                    BinaryOp::GreaterThanEqual => left.gte(right),
                    BinaryOp::LessThan => left.lt(right),
                    BinaryOp::LessThanEqual => left.lte(right),

                    // Arithmetic
                    BinaryOp::Add => left.add(right),
                    BinaryOp::Sub => left.sub(right),
                    BinaryOp::Mul => left.mul(right),
                    BinaryOp::Div => left.div(right),

                    // Logical operators were handled already
                    _ => unreachable!(),
                }
            }
        }
    }

    fn evaluate_function_call(
        &mut self,
        expr: Expr<'a>,
        arguments: Vec<Expr<'a>>,
    ) -> Evaluation<'a> {
        let value = self.evaluate_expression(expr)?;
        if let Value::Callable(callee) = value {
            if arguments.len() != callee.parameters.len() {
                return Err(RuntimeError::MismatchedArity {
                    expected: callee.parameters.len(),
                    got: arguments.len(),
                });
            }

            // Evaluate the arguments before replacing our environment
            let arguments = arguments
                .into_iter()
                .map(|expr| self.evaluate_expression(expr))
                .collect::<Result<Vec<Value<'a>>, RuntimeError>>()?;

            // Allocate a new environment for the call
            let new_env = self.env.push(callee.parent_env);

            // Replace ours with it and store the previous
            let old_env = self.env.switch(new_env);

            // Inject the arguments
            for (identifier, value) in callee.parameters.into_iter().zip(arguments.into_iter()) {
                self.env.declare_symbol(identifier, value)
            }

            // NOTE: `perform_block` instead of `execute_block` to avoid pushing a useless scope
            let result = self.perform_block(callee.body);

            // Restore the active environment
            self.env.active = old_env;

            // If no edge was connected to the newly created environment we can safely dispose of it
            // This avoids pilling up "weak" environments
            if !self.env.get(new_env).is_parent {
                assert_eq!(new_env, self.env.list.len() - 1);
                self.env.pop();
            }

            match result {
                Ok(Stated::ControlFlow(ControlFlow::Return(value))) => Ok(value),
                Ok(_) => Ok(Value::Nil),
                Err(e) => Err(e),
            }
        } else if let Value::Class(class) = value {
            Ok(Value::Instance(class))
        } else {
            Err(RuntimeError::InvalidCallee(value))
        }
    }

    fn evaluate_grouping(&mut self, expr: Expr<'a>) -> Evaluation<'a> {
        self.evaluate_expression(expr)
    }

    fn evaluate_lambda(&mut self, parameters: Vec<&'a str>, body: Block<'a>) -> Evaluation<'a> {
        Ok(Value::Callable(Function {
            identifier: None,
            parameters,
            body,
            parent_env: self.env.snapshot(),
        }))
    }

    fn evaluate_literal(&self, literal: Literal<'a>) -> Evaluation<'a> {
        match literal {
            Literal::Nil => Ok(Value::Nil),
            Literal::False => Ok(Value::Bool(false)),
            Literal::True => Ok(Value::Bool(true)),
            Literal::Number(n) => Ok(Value::Number(n)),
            Literal::String(s) => Ok(Value::String(Cow::Borrowed(s))),
        }
    }

    fn evaluate_property_access(&mut self, expr: Expr<'a>, property: &'a str) -> Evaluation<'a> {
        let value = self.evaluate_expression(expr)?;
        if let Value::Instance(object) = value {
            object
                .scope
                .get(property)
                .cloned()
                .ok_or(RuntimeError::UndefinedProperty(property))
        } else {
            Err(RuntimeError::InvalidObject(value))
        }
    }

    fn evaluate_property_assignment(
        &mut self,
        expr: Expr<'a>,
        property: &'a str,
        op: AssignmentOp,
        value: Expr<'a>,
    ) -> Evaluation<'a> {
        let value = self.evaluate_expression(value)?;

        fn implementation<'a>(
            object: &mut Class<'a>,
            property: &'a str,
            op: AssignmentOp,
            value: Value<'a>,
        ) -> Evaluation<'a> {
            match op {
                AssignmentOp::Set => {
                    object.scope.insert(property, value.clone());
                    Ok(value)
                }
                op => {
                    if let Some(var) = object.scope.get_mut(property) {
                        match op {
                            AssignmentOp::Add => var.add_assign(value)?,
                            AssignmentOp::Sub => var.sub_assign(value)?,
                            AssignmentOp::Mul => var.mul_assign(value)?,
                            AssignmentOp::Div => var.div_assign(value)?,
                            _ => unreachable!(),
                        };
                        Ok(var.clone())
                    } else {
                        Err(RuntimeError::UndefinedProperty(property))
                    }
                }
            }
        }

        match self.get_property(expr)? {
            MaybeOwned::Borrowed(object) => implementation(object, property, op, value),
            MaybeOwned::Owned(mut object) => implementation(&mut object, property, op, value),
        }
    }

    fn evaluate_symbol(&mut self, identifier: &'a str) -> Evaluation<'a> {
        if let Some(value) = self.env.find_symbol(identifier) {
            match value {
                Value::Undefined => Err(RuntimeError::UninitializedVariable(identifier)),
                value => Ok(value.clone()),
            }
        } else {
            Err(RuntimeError::UndefinedVariable(identifier))
        }
    }

    fn evaluate_unary(&mut self, op: UnaryOp, expr: Expr<'a>) -> Evaluation<'a> {
        let expr = self.evaluate_expression(expr)?;
        match op {
            UnaryOp::Minus => match expr {
                Value::Number(n) => Ok(Value::Number(-n)),
                expr => Err(RuntimeError::InvalidOperand(expr, op)),
            },
            UnaryOp::LogicalNot => Ok(Value::Bool(!expr.is_truthy())),
        }
    }

    fn get_property(&mut self, expr: Expr<'a>) -> Result<MaybeOwned<Class<'a>>, RuntimeError<'a>> {
        match expr {
            Expr::Symbol { identifier } => match self.env.find_symbol(identifier) {
                Some(Value::Instance(object)) => Ok(MaybeOwned::Borrowed(object)),
                Some(value) => Err(RuntimeError::InvalidObject(value.clone())),
                None => Err(RuntimeError::UndefinedVariable(identifier)),
            },
            Expr::PropertyAccess { expr, property } => {
                let class = self.get_property(*expr)?;
                match class {
                    MaybeOwned::Borrowed(class) => match class.scope.get_mut(property) {
                        Some(Value::Instance(object)) => Ok(MaybeOwned::Borrowed(object)),
                        Some(value) => Err(RuntimeError::InvalidObject(value.clone())),
                        None => Err(RuntimeError::UndefinedProperty(property)),
                    },
                    MaybeOwned::Owned(mut class) => match class.scope.remove(property) {
                        Some(Value::Instance(object)) => Ok(MaybeOwned::Owned(object)),
                        Some(value) => Err(RuntimeError::InvalidObject(value)),
                        None => Err(RuntimeError::UndefinedProperty(property)),
                    },
                }
            }
            expr => match self.evaluate_expression(expr)? {
                Value::Instance(object) => Ok(MaybeOwned::Owned(object)),
                value => Err(RuntimeError::InvalidObject(value)),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value<'a> {
    Number(LoxNumber),
    String(Cow<'a, str>),
    Bool(bool),
    Callable(Function<'a>),
    Class(Class<'a>),
    Instance(Class<'a>),
    Nil,
    Undefined,
}

impl<'a> Value<'a> {
    fn eq(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a == b)),

            (String(a), String(b)) => Ok(Bool(a == b)),

            (Bool(a), Bool(b)) => Ok(Bool(a == b)),

            (Nil, Nil) => Ok(Bool(true)),

            _ => Ok(Bool(false)),
        }
    }

    fn neq(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a != b)),

            (String(a), String(b)) => Ok(Bool(a != b)),

            (Bool(a), Bool(b)) => Ok(Bool(a != b)),

            (Nil, Nil) => Ok(Bool(false)),

            _ => Ok(Bool(true)),
        }
    }

    fn gt(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a > b)),
            (Number(a), Bool(b)) => Ok(Bool(a > LoxNumber::from(b))),

            (String(a), String(b)) => Ok(Bool(a > b)),

            (Bool(a), Bool(b)) => Ok(Bool(a > b)),
            (Bool(a), Number(b)) => Ok(Bool(b < LoxNumber::from(a))),

            (left, right) => Err(RuntimeError::InvalidOperands(
                left,
                right,
                BinaryOp::GreaterThan,
            )),
        }
    }

    fn gte(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a >= b)),
            (Number(a), Bool(b)) => Ok(Bool(a >= LoxNumber::from(b))),

            (String(a), String(b)) => Ok(Bool(a >= b)),

            (Bool(a), Bool(b)) => Ok(Bool(a >= b)),
            (Bool(a), Number(b)) => Ok(Bool(b <= LoxNumber::from(a))),

            (left, right) => Err(RuntimeError::InvalidOperands(
                left,
                right,
                BinaryOp::GreaterThanEqual,
            )),
        }
    }

    fn lt(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a < b)),
            (Number(a), Bool(b)) => Ok(Bool(a < LoxNumber::from(b))),

            (String(a), String(b)) => Ok(Bool(a < b)),

            (Bool(a), Bool(b)) => Ok(Bool(a < b)),
            (Bool(a), Number(b)) => Ok(Bool(b < LoxNumber::from(a))),

            (left, right) => Err(RuntimeError::InvalidOperands(
                left,
                right,
                BinaryOp::LessThan,
            )),
        }
    }

    fn lte(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Bool(a <= b)),
            (Number(a), Bool(b)) => Ok(Bool(a <= LoxNumber::from(b))),

            (String(a), String(b)) => Ok(Bool(a <= b)),

            (Bool(a), Bool(b)) => Ok(Bool(a <= b)),
            (Bool(a), Number(b)) => Ok(Bool(b <= LoxNumber::from(a))),

            (left, right) => Err(RuntimeError::InvalidOperands(
                left,
                right,
                BinaryOp::LessThanEqual,
            )),
        }
    }

    fn add(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Number(a + b)),
            (Number(a), Bool(b)) => Ok(Number(a + LoxNumber::from(b))),
            (Bool(a), Number(b)) => Ok(Number(LoxNumber::from(a) + b)),
            (Bool(a), Bool(b)) => Ok(Number(LoxNumber::from(a) + LoxNumber::from(b))),

            (String(a), String(b)) => Ok(String(a + b)),
            (String(a), Number(b)) => Ok(String(a + Cow::from(b.to_string()))),
            (String(a), Bool(b)) => Ok(String(a + Cow::from(b.to_string()))),
            (Number(a), String(b)) => Ok(String(Cow::from(a.to_string()) + b)),
            (Bool(a), String(b)) => Ok(String(Cow::from(a.to_string()) + b)),

            (a, b) => Err(RuntimeError::InvalidOperands(a, b, BinaryOp::Add)),
        }
    }

    fn add_assign(&mut self, right: Value<'a>) -> Result<(), RuntimeError<'a>> {
        *self = replace(self, Value::Undefined).add(right)?;
        Ok(())
    }

    fn sub(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Number(a - b)),
            (Number(a), Bool(b)) => Ok(Number(a - LoxNumber::from(b))),

            (Bool(a), Number(b)) => Ok(Number(LoxNumber::from(a) - b)),
            (Bool(a), Bool(b)) => Ok(Number(LoxNumber::from(a) - LoxNumber::from(b))),

            (a, b) => Err(RuntimeError::InvalidOperands(a, b, BinaryOp::Sub)),
        }
    }

    fn sub_assign(&mut self, right: Value<'a>) -> Result<(), RuntimeError<'a>> {
        *self = replace(self, Value::Undefined).sub(right)?;
        Ok(())
    }

    fn mul(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Number(a * b)),
            (Number(a), Bool(b)) => Ok(Number(a * LoxNumber::from(b))),
            (Bool(a), Number(b)) => Ok(Number(LoxNumber::from(a) * b)),
            (Bool(a), Bool(b)) => Ok(Number(LoxNumber::from(a) * LoxNumber::from(b))),

            (String(a), Number(b)) => Ok(String(Cow::from(a.repeat(b as usize)))),
            (Number(a), String(b)) => Ok(String(Cow::from(b.repeat(a as usize)))),

            (a, b) => Err(RuntimeError::InvalidOperands(a, b, BinaryOp::Mul)),
        }
    }

    fn mul_assign(&mut self, right: Value<'a>) -> Result<(), RuntimeError<'a>> {
        *self = replace(self, Value::Undefined).mul(right)?;
        Ok(())
    }

    fn div(self, right: Value<'a>) -> Evaluation<'a> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Ok(Number(a / b)),
            (Number(a), Bool(b)) => Ok(Number(a / LoxNumber::from(b))),

            (Bool(a), Number(b)) => Ok(Number(LoxNumber::from(a) / b)),
            (Bool(a), Bool(b)) => Ok(Number(LoxNumber::from(a) / LoxNumber::from(b))),

            (a, b) => Err(RuntimeError::InvalidOperands(a, b, BinaryOp::Div)),
        }
    }

    fn div_assign(&mut self, right: Value<'a>) -> Result<(), RuntimeError<'a>> {
        *self = replace(self, Value::Undefined).div(right)?;
        Ok(())
    }

    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Nil => false,
            _ => true,
        }
    }
}

impl<'a> Display for Value<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Callable(c) => write!(f, "function \"{}\"", c.identifier.unwrap_or("<anon>")),
            Value::Class(c) => write!(f, "class \"{}\"", c.identifier),
            Value::Instance(o) => write!(f, "object {}", o.identifier),
            Value::Nil => write!(f, "nil"),
            Value::Undefined => write!(f, "undefined"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function<'a> {
    identifier: Option<&'a str>,
    parameters: Vec<&'a str>,
    body: Block<'a>,
    parent_env: usize,
}

#[derive(Debug, Clone)]
pub struct Class<'a> {
    identifier: &'a str,
    scope: Scope<'a>,
    #[allow(dead_code)]
    superclass: Option<Box<Class<'a>>>,
}

/// Maps identifiers to `Value`s
type Scope<'a> = HashMap<&'a str, Value<'a>>;

/// A Lox "environment", containing a list of scopes and a handle to the parent it "inherits" (closes) from.
#[derive(Debug, Clone)]
struct Env<'a> {
    /// The FILO stack of scopes that gets pushed/popped by blocks and functions.
    /// There is always at least one scope and the active scope is always the last one.
    scopes: Vec<Scope<'a>>,

    /// The parent environment, if any.
    /// Optional because the global environment has no parent.
    parent: Option<EnvHandle>,

    /// Whether this environment is a parent to another environment.
    /// This is an optimization to avoid walking the entire tree when deciding weather an environment has SCCs.
    is_parent: bool,
}

impl<'a> Env<'a> {
    /// Creates a new environment with a single scope as a child of the given parent.
    fn with_parent(parent: usize) -> Self {
        Self {
            scopes: vec![Scope::new()],
            parent: Some(parent),
            is_parent: false,
        }
    }

    fn find_symbol(&mut self, identifier: &'a str) -> Option<&mut Value<'a>> {
        self.scopes
            .iter_mut()
            .rev()
            .find_map(|scope| scope.get_mut(identifier))
    }

    fn declare_symbol(&mut self, identifier: &'a str, value: Value<'a>) {
        self.current_scope().insert(identifier, value);
    }

    fn current_scope(&mut self) -> &mut Scope<'a> {
        // SAFETY: We always have at least one scope.
        unsafe { self.scopes.last_mut().unwrap_unchecked() }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

/// Manages the "tree" of environments and provides methods to interact with them.
/// This is a tree-like structure but is implemented as a flat list where parent nodes are an index into said list instead of a direct pointer/reference,
/// which makes the borrow checker happy.
#[derive(Debug, Clone)]
struct EnvManager<'a> {
    list: Vec<Env<'a>>,
    active: EnvHandle,
}

type EnvHandle = usize;

impl<'a> EnvManager<'a> {
    /// Creates a new environment manager with a single global environment.
    fn new() -> Self {
        // The global scope
        let global = Env {
            scopes: vec![Scope::new()],
            parent: None,
            is_parent: false,
        };

        Self {
            list: vec![global],
            active: 0,
        }
    }

    fn find_symbol(&mut self, identifier: &'a str) -> Option<&mut Value<'a>> {
        self.find_symbol_in(self.active, identifier)
    }

    fn declare_symbol(&mut self, identifier: &'a str, value: Value<'a>) {
        self.active_env().declare_symbol(identifier, value);
    }

    fn declare_function(&mut self, identifier: &'a str, function: Function<'a>) {
        self.declare_symbol(identifier, Value::Callable(function));
    }

    fn declare_class(&mut self, identifier: &'a str, class: Class<'a>) {
        self.declare_symbol(identifier, Value::Class(class));
    }

    fn snapshot(&mut self) -> EnvHandle {
        let new_env = self.push(self.active);
        self.switch(new_env)
    }

    fn push(&mut self, parent: EnvHandle) -> EnvHandle {
        self.list.push(Env::with_parent(parent));
        self.list[parent].is_parent = true;
        self.list.len() - 1
    }

    fn pop(&mut self) {
        self.list.pop();
    }

    fn switch(&mut self, env: EnvHandle) -> EnvHandle {
        replace(&mut self.active, env)
    }

    fn push_scope(&mut self) {
        self.active_env().push_scope();
    }

    fn pop_scope(&mut self) {
        self.active_env().pop_scope();
    }

    /// Walks backwards from the given environment to it's parents until it finds the first instance of the given identifier.
    fn find_symbol_in(&mut self, env: EnvHandle, identifier: &'a str) -> Option<&mut Value<'a>> {
        let env = self.get(env);
        if let Some(value) = env.find_symbol(identifier) {
            // SAFETY:
            // The output lifetime is bound to `self`, so this doesn't extend the lifetime.
            // FIXME: remove once polonius (or another new borrow checker) sees that it's fine.
            unsafe { Some(std::mem::transmute::<&mut Value<'a>, &mut Value<'a>>(value)) }
        } else if let Some(parent) = env.parent {
            self.find_symbol_in(parent, identifier)
        } else {
            None
        }
    }

    fn get(&mut self, env: EnvHandle) -> &mut Env<'a> {
        &mut self.list[env]
    }

    fn active_env(&mut self) -> &mut Env<'a> {
        self.get(self.active)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError<'a> {
    #[error("Invalid operand \"{0}\" for unary operator \"{1:?}\".")]
    InvalidOperand(Value<'a>, UnaryOp),

    #[error("Invalid operands \"{0}\" and \"{1}\" for binary operator \"{2:?}\".")]
    InvalidOperands(Value<'a>, Value<'a>, BinaryOp),

    #[error("Value \"{0}\" is not callable.")]
    InvalidCallee(Value<'a>),

    #[error("Value \"{0}\" is not a class object.")]
    InvalidObject(Value<'a>),

    #[error("Undefined property \"{0}\".")]
    UndefinedProperty(&'a str),

    #[error("Undefined variable \"{0}\".")]
    UndefinedVariable(&'a str),

    #[error("Variable \"{0}\" is unititialized and cannot be accessed.")]
    UninitializedVariable(&'a str),

    #[error("Mismatched argument count, expected ({expected}) but got ({got})")]
    MismatchedArity { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub enum ControlFlow<'a> {
    Break,
    Continue,
    Return(Value<'a>),
}

impl<'a> Display for ControlFlow<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Break => write!(f, "break"),
            Self::Continue => write!(f, "continue"),
            Self::Return(v) => write!(f, "{}", v),
        }
    }
}

/// The result of executing a statement.
#[derive(Debug, Clone)]
pub enum Stated<'a> {
    Nothing,
    #[allow(dead_code)]
    Value(Value<'a>),
    ControlFlow(ControlFlow<'a>),
}

impl<'a> Display for Stated<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nothing => Ok(()),
            Self::Value(v) => write!(f, "{}", v),
            Self::ControlFlow(c) => write!(f, "{}", c),
        }
    }
}

type Execution<'a> = Result<Stated<'a>, RuntimeError<'a>>;
type Evaluation<'a> = Result<Value<'a>, RuntimeError<'a>>;

enum MaybeOwned<'a, T> {
    Borrowed(&'a mut T),
    Owned(T),
}
