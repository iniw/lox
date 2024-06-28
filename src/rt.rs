use std::{borrow::Cow, collections::HashMap, fmt::Display, mem::replace};

use crate::{
    ast::{
        AssignmentOp, BinaryOp, ClassDecl, Expr, FunDecl, If, Literal, Stmt, UnaryOp, VarDecl,
        While,
    },
    LoxNumber,
};

/// A runtime representation of a Lox value.
#[derive(Debug, Clone)]
pub enum Value<'a> {
    /// A Lox number.
    Number(LoxNumber),

    /// A Lox string.
    String(Cow<'a, str>),

    /// A Lox boolean.
    Bool(bool),

    /// A Lox callable.
    /// Represents functions, lambdas and callable values that are, for example, returned by a function.
    Callable(Function<'a>),

    /// A Lox class object.
    Object(Class<'a>),

    /// The Lox `nil` value.
    Nil,

    /// The state of a variable that has not been initialized.
    /// It is also what functions that don't reach a return statement (or that don't return an expression) return.
    Undefined,
}

impl<'a> Value<'a> {
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
    fn equal(&self, right: &Value<'a>) -> Option<bool> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Some(a == b),
            (Number(a), Bool(b)) => Some(*a == LoxNumber::from(*b)),
            (Number(_), String(_)) => Some(false),

            (String(a), String(b)) => Some(a == b),
            (String(_), Number(_)) => Some(false),
            (String(_), Bool(_)) => Some(false),

            (Bool(a), Bool(b)) => Some(a == b),
            (Bool(a), Number(b)) => Some(LoxNumber::from(*a) == *b),

            (Nil, Nil) => Some(true),
            // Any non-nil value is not nil
            (_, Nil) => Some(false),
            (Nil, _) => Some(false),

            _ => None,
        }
    }

    fn not_equal(&self, right: &Value<'a>) -> Option<bool> {
        self.equal(right).map(|b| !b)
    }

    fn greater_than(&self, right: &Value<'a>) -> Option<bool> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Some(a > b),
            (Number(a), Bool(b)) => Some(*a > LoxNumber::from(*b)),

            (String(a), String(b)) => Some(a > b),

            (Bool(a), Bool(b)) => Some(a > b),
            (Bool(a), Number(b)) => Some(*b < LoxNumber::from(*a)),

            _ => None,
        }
    }

    fn greater_than_equal(&self, right: &Value<'a>) -> Option<bool> {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => Some(a >= b),
            (Number(a), Bool(b)) => Some(*a >= LoxNumber::from(*b)),

            (String(a), String(b)) => Some(a >= b),

            (Bool(a), Bool(b)) => Some(a >= b),
            (Bool(a), Number(b)) => Some(*b <= LoxNumber::from(*a)),

            _ => None,
        }
    }

    fn less_than(&self, right: &Value<'a>) -> Option<bool> {
        self.greater_than_equal(right).map(|b| !b)
    }

    fn less_than_equal(&self, right: &Value<'a>) -> Option<bool> {
        self.greater_than(right).map(|b| !b)
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
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Callable(c) => write!(f, "fun {}", c.identifier.unwrap_or("<anon>")),
            Value::Object(o) => write!(f, "object {}", o.identifier),
            Value::Nil => write!(f, "nil"),
            Value::Undefined => write!(f, "undefined"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function<'a> {
    identifier: Option<&'a str>,
    parameters: Vec<&'a str>,
    body: Stmt<'a>,
    environment: Environment<'a>,
}

#[derive(Debug, Clone)]
pub struct Method<'a> {
    parameters: Vec<&'a str>,
    body: Stmt<'a>,
}

#[derive(Debug, Clone)]
pub struct Class<'a> {
    identifier: &'a str,
    scope: Scope<'a>,
    methods: HashMap<&'a str, Method<'a>>,
    superclass: Option<Box<Class<'a>>>,
}

#[derive(Debug, Clone)]
struct Scope<'a> {
    symbols: HashMap<&'a str, Value<'a>>,
}

impl<'a> Scope<'a> {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }

    fn find_symbol(&mut self, identifier: &'a str) -> Option<&mut Value<'a>> {
        self.symbols.get_mut(identifier)
    }

    fn declare_symbol(&mut self, identifier: &'a str, value: Value<'a>) {
        self.symbols.insert(identifier, value);
    }

    fn declare_function(&mut self, identifier: &'a str, function: Function<'a>) {
        self.symbols.insert(identifier, Value::Callable(function));
    }
}

/// A stack of scopes.
type Environment<'a> = Vec<Scope<'a>>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError<'a> {
    #[error("Invalid operand \"{0}\" for unary operator \"{1:?}\".")]
    InvalidOperand(Value<'a>, UnaryOp),

    #[error("Invalid operands \"{0}\" and \"{1}\" for binary operator \"{2:?}\".")]
    InvalidOperands(Value<'a>, Value<'a>, BinaryOp),

    #[error("Value \"{0}\" is not callable.")]
    InvalidCallee(Value<'a>),

    #[error("Variable \"{0}\" is not in scope.")]
    UnknownVariable(&'a str),

    #[error("Variable \"{0}\" is unititialized and cannot be accessed.")]
    UninitializedVariableAccess(&'a str),

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
    /// Nothing was produced by the statement.
    Nothing,

    /// A value was produced by the statement.
    #[allow(dead_code)]
    Value(Value<'a>),

    /// Some control flow operation was produced, such as a `break`, `continue` or `return`.
    /// Will be propagated up the statement stack until it is handled.
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

/// Our tree-walk interpreter.
#[derive(Debug, Clone)]
pub struct TreeWalker<'a> {
    environment: Environment<'a>,
    classes: HashMap<&'a str, Class<'a>>,
}

impl<'a> TreeWalker<'a> {
    pub fn new() -> Self {
        Self {
            // INVARIANT: The first scope is the global one, it is always present
            environment: vec![Scope::new()],
            classes: HashMap::new(),
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
            Stmt::Empty => self.execute_empty(),
            Stmt::Expr(expr) => self.execute_expression(expr),
            Stmt::Block(statements) => self.execute_block(statements),
            Stmt::VarDecl(var_decl) => self.execute_var_decl(var_decl),
            Stmt::FunDecl(fun_decl) => self.execute_fun_decl(fun_decl),
            Stmt::ClassDecl(class_decl) => self.execute_class_decl(class_decl),
            Stmt::Print(expr) => self.execute_print(expr),
            Stmt::If(if_data) => self.execute_if(if_data),
            Stmt::While(while_data) => self.execute_while(while_data),
            Stmt::Break => self.execute_break(),
            Stmt::Continue => self.execute_continue(),
            Stmt::Return(expr) => self.execute_return(expr),
        }
    }

    fn execute_empty(&self) -> Execution<'a> {
        // No-op
        Ok(Stated::Nothing)
    }

    fn execute_expression(&mut self, expr: Expr<'a>) -> Execution<'a> {
        self.evaluate_expression(expr)
            .map(|expr| Stated::Value(expr))
    }

    fn execute_block(&mut self, statements: Vec<Stmt<'a>>) -> Execution<'a> {
        self.environment.push(Scope::new());
        let result = self.perform_block(statements);
        self.environment.pop();

        result
    }

    fn execute_var_decl(&mut self, var_decl: VarDecl<'a>) -> Execution<'a> {
        let value = var_decl
            .expr
            .map(|expr| self.evaluate_expression(expr))
            .transpose()?;

        let value = value.unwrap_or(Value::Undefined);
        self.current_scope()
            .declare_symbol(var_decl.identifier, value);

        Ok(Stated::Nothing)
    }

    fn execute_fun_decl(&mut self, fun_decl: FunDecl<'a>) -> Execution<'a> {
        let function = Function {
            identifier: Some(fun_decl.identifier),
            parameters: fun_decl.parameters,
            body: *fun_decl.body,
            // Capture the current environment
            environment: self.environment.clone(),
        };

        self.current_scope()
            .declare_function(fun_decl.identifier, function);

        Ok(Stated::Nothing)
    }

    fn execute_class_decl(&mut self, class_decl: ClassDecl<'a>) -> Execution<'a> {
        let methods = {
            let mut methods = HashMap::new();
            for FunDecl {
                identifier,
                parameters,
                body,
            } in class_decl.methods
            {
                methods.insert(
                    identifier,
                    Method {
                        parameters,
                        body: *body,
                    },
                );
            }
            methods
        };

        self.classes.insert(
            class_decl.identifier,
            Class {
                identifier: class_decl.identifier,
                scope: Scope::new(),
                methods,
                superclass: None,
            },
        );

        Ok(Stated::Nothing)
    }

    fn execute_print(&mut self, expr: Expr<'a>) -> Execution<'a> {
        println!("{}", self.evaluate_expression(expr)?);
        Ok(Stated::Nothing)
    }

    fn execute_if(&mut self, if_data: If<'a>) -> Execution<'a> {
        if self.evaluate_expression(if_data.condition)?.is_truthy() {
            self.execute_statement(*if_data.branch)
        } else if let Some(else_branch) = if_data.else_branch {
            self.execute_statement(*else_branch)
        } else {
            Ok(Stated::Nothing)
        }
    }

    fn execute_while(&mut self, while_data: While<'a>) -> Execution<'a> {
        while self
            .evaluate_expression(while_data.condition.clone())?
            .is_truthy()
        {
            let result = self.execute_statement(*while_data.body.clone())?;
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

    fn execute_break(&self) -> Execution<'a> {
        Ok(Stated::ControlFlow(ControlFlow::Break))
    }

    fn execute_continue(&self) -> Execution<'a> {
        Ok(Stated::ControlFlow(ControlFlow::Continue))
    }

    fn execute_return(&mut self, expr: Option<Expr<'a>>) -> Execution<'a> {
        let value = if let Some(expr) = expr {
            self.evaluate_expression(expr)?
        } else {
            Value::Undefined
        };

        Ok(Stated::ControlFlow(ControlFlow::Return(value)))
    }

    fn evaluate_expression(&mut self, expr: Expr<'a>) -> Evaluation<'a> {
        match expr {
            Expr::Literal(literal) => self.evaluate_literal(literal),
            Expr::Symbol { identifier } => self.evaluate_symbol(identifier),
            Expr::Assignment {
                op,
                identifier,
                expr,
            } => self.evaluate_assignment(op, identifier, *expr),
            Expr::Binary { left, op, right } => self.evaluate_binary(*left, op, *right),
            Expr::Unary { op, expr } => self.evaluate_unary(op, *expr),
            Expr::Grouping(expr) => self.evaluate_expression(*expr),
            Expr::Lambda { parameters, body } => self.evaluate_lambda(parameters, *body),
            Expr::FunctionCall { expr, arguments } => self.evaluate_function_call(*expr, arguments),
        }
    }

    fn evaluate_lambda(&mut self, parameters: Vec<&'a str>, body: Stmt<'a>) -> Evaluation<'a> {
        Ok(Value::Callable(Function {
            identifier: None,
            parameters,
            body,
            // Capture the current environment
            environment: self.environment.clone(),
        }))
    }

    fn evaluate_function_call(
        &mut self,
        expr: Expr<'a>,
        arguments: Vec<Expr<'a>>,
    ) -> Evaluation<'a> {
        // When the callable is bound to an identifier side-effects in it's captured scope should be observed when repeatedly calling through that identifier,
        // which is why the identifier case is handled separately here.
        // That behavior is not necessary when there is no identifier associated with the callable, since every invokation is a different `Value`, so we can just not worry about it.
        if let Expr::Symbol { identifier } = expr {
            if let Some(value) = self.find_symbol(identifier).cloned() {
                if let Value::Callable(callee) = value {
                    let (result, environment) = self.perform_function_call(callee, arguments)?;
                    if let Some(Value::Callable(callee)) = self.find_symbol(identifier) {
                        // Update the closed-in environment with the current one, this is how we "notify" the callable of any side-effects.
                        callee.environment = environment;
                    } else {
                        // The symbol was already searched, if it wasn't found or wasn't a Function we wouldn't have reached this point
                        unreachable!();
                    }

                    Ok(result)
                } else {
                    // TODO: try initiating a class instead
                    Err(RuntimeError::InvalidCallee(value))
                }
            } else if let Some(mut class) = self.classes.get(identifier).cloned() {
                if let Some(Value::Callable(init)) = class.scope.find_symbol("init").cloned() {
                    self.perform_function_call(init, arguments)?;
                    Ok(Value::Object(class))
                } else {
                    Err(RuntimeError::InvalidCallee(Value::Object(class)))
                }
            } else {
                Err(RuntimeError::UnknownVariable(identifier))
            }
        } else {
            let value = self.evaluate_expression(expr)?;
            if let Value::Callable(callee) = value {
                self.perform_function_call(callee, arguments)
                    .map(|(value, _)| value)
            } else {
                Err(RuntimeError::InvalidCallee(value))
            }
        }
    }

    fn perform_function_call(
        &mut self,
        callee: Function<'a>,
        arguments: Vec<Expr<'a>>,
    ) -> Result<(Value<'a>, Environment<'a>), RuntimeError<'a>> {
        if arguments.len() != callee.parameters.len() {
            return Err(RuntimeError::MismatchedArity {
                got: arguments.len(),
                expected: callee.parameters.len(),
            });
        }

        // Evaluate the arguments before replacing our environment
        let arguments = arguments
            .into_iter()
            .map(|expr| self.evaluate_expression(expr))
            .collect::<Result<Vec<Value<'a>>, RuntimeError>>()?;

        // Temporarily replace our environment with the callee's
        let backup = replace(&mut self.environment, callee.environment);

        // Put the parameters in scope
        for (identifier, value) in callee.parameters.into_iter().zip(arguments.into_iter()) {
            self.current_scope().declare_symbol(identifier, value)
        }

        // Execute the callee's body, capturing any errors it may throw
        let value = match self.execute_statement(callee.body) {
            Ok(Stated::ControlFlow(ControlFlow::Return(value))) => value,
            Ok(_) => Value::Undefined,
            Err(error) => return Err(error),
        };

        // Restore our environment and return the callee environment back
        Ok((value, replace(&mut self.environment, backup)))
    }

    fn perform_block(&mut self, statements: Vec<Stmt<'a>>) -> Execution<'a> {
        for statement in statements {
            if let flow @ Stated::ControlFlow(_) = self.execute_statement(statement)? {
                // Propagate the control flow up decision up the statement chain
                return Ok(flow);
            }
        }

        Ok(Stated::Nothing)
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

    fn evaluate_symbol(&mut self, identifier: &'a str) -> Evaluation<'a> {
        if let Some(value) = self.find_symbol(identifier) {
            match value {
                Value::Undefined => Err(RuntimeError::UninitializedVariableAccess(identifier)),
                value => Ok(value.clone()),
            }
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn find_symbol(&mut self, identifier: &'a str) -> Option<&mut Value<'a>> {
        for scope in self.environment.iter_mut().rev() {
            if let Some(value) = scope.find_symbol(identifier) {
                return Some(value);
            }
        }

        None
    }

    fn evaluate_assignment(
        &mut self,
        op: AssignmentOp,
        identifier: &'a str,
        expr: Expr<'a>,
    ) -> Evaluation<'a> {
        let value = self.evaluate_expression(expr)?;
        match op {
            AssignmentOp::Set => self.set_symbol(identifier, value),
            AssignmentOp::Add => self.add_set_symbol(identifier, value),
            AssignmentOp::Sub => self.sub_set_symbol(identifier, value),
            AssignmentOp::Mul => self.mul_set_symbol(identifier, value),
            AssignmentOp::Div => self.div_set_symbol(identifier, value),
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
                    BinaryOp::Equal => left
                        .equal(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

                    BinaryOp::NotEqual => left
                        .not_equal(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

                    BinaryOp::GreaterThan => left
                        .greater_than(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

                    BinaryOp::GreaterThanEqual => left
                        .greater_than_equal(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

                    BinaryOp::LessThan => left
                        .less_than(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

                    BinaryOp::LessThanEqual => left
                        .less_than_equal(&right)
                        .map(|b| Value::Bool(b))
                        .ok_or(RuntimeError::InvalidOperands(left, right, op)),

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

    fn evaluate_unary(&mut self, op: UnaryOp, expr: Expr<'a>) -> Evaluation<'a> {
        let expr = self.evaluate_expression(expr)?;
        match op {
            UnaryOp::Minus => match expr {
                Value::Number(n) => Ok(Value::Number(-n)),
                expr => Err(RuntimeError::InvalidOperand(expr, op)),
            },
            UnaryOp::LogicalNot => match expr {
                v => Ok(Value::Bool(!v.is_truthy())),
            },
        }
    }

    fn set_symbol(&mut self, identifier: &'a str, value: Value<'a>) -> Evaluation<'a> {
        if let Some(v) = self.find_symbol(identifier) {
            *v = value;
            Ok(v.clone())
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn add_set_symbol(&mut self, identifier: &'a str, value: Value<'a>) -> Evaluation<'a> {
        if let Some(v) = self.find_symbol(identifier) {
            // Temporarily replace it with a sentinel value so that we can
            // take it's ownership even though we are locked behind a (mutable) reference
            *v = replace(v, Value::Undefined).add(value)?;
            Ok(v.clone())
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn sub_set_symbol(&mut self, identifier: &'a str, value: Value<'a>) -> Evaluation<'a> {
        if let Some(v) = self.find_symbol(identifier) {
            // Temporarily replace it with a sentinel value so that we can
            // take it's ownership even though we are locked behind a (mutable) reference
            *v = replace(v, Value::Undefined).sub(value)?;
            Ok(v.clone())
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn mul_set_symbol(&mut self, identifier: &'a str, value: Value<'a>) -> Evaluation<'a> {
        if let Some(v) = self.find_symbol(identifier) {
            // Temporarily replace it with a sentinel value so that we can
            // take it's ownership even though we are locked behind a (mutable) reference
            *v = replace(v, Value::Undefined).mul(value)?;
            Ok(v.clone())
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn div_set_symbol(&mut self, identifier: &'a str, value: Value<'a>) -> Evaluation<'a> {
        if let Some(v) = self.find_symbol(identifier) {
            // Temporarily replace it with a sentinel value so that we can
            // take it's ownership even though we are locked behind a (mutable) reference
            *v = replace(v, Value::Undefined).div(value)?;
            Ok(v.clone())
        } else {
            Err(RuntimeError::UnknownVariable(identifier))
        }
    }

    fn current_scope(&mut self) -> &mut Scope<'a> {
        self.environment
            .last_mut()
            .expect("There should always be at least one scope (the global one)")
    }
}
