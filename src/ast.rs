use core::slice::Iter;
use std::{convert::From, iter::Peekable, mem::replace};

use crate::{
    lex::{Token, TokenInfo},
    LoxNumber,
};

/// A literal value in the Lox language.
#[derive(Debug, Copy, Clone)]
pub enum Literal<'a> {
    Number(LoxNumber),
    String(&'a str),
    True,
    False,
    Nil,
}

/// The unary operators in the Lox language.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Minus,
    LogicalNot,
}

impl<'a> From<&Token<'a>> for UnaryOp {
    /// Constructs a `UnaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid unary operator.
    fn from(token: &Token<'a>) -> Self {
        match token {
            Token::Minus => UnaryOp::Minus,
            Token::Bang => UnaryOp::LogicalNot,
            _ => unreachable!("Invalid token for unary operator"),
        }
    }
}

/// The binary operators in the Lox language.
#[derive(Debug, Copy, Clone)]
pub enum BinaryOp {
    // Logical
    And,
    Or,
    // Relational
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
}

impl<'a> From<&Token<'a>> for BinaryOp {
    /// Constructs a `BinaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid binary operator.
    fn from(token: &Token<'a>) -> Self {
        match token {
            Token::BangEqual => BinaryOp::NotEqual,
            Token::EqualEqual => BinaryOp::Equal,
            Token::Greater => BinaryOp::GreaterThan,
            Token::GreaterEqual => BinaryOp::GreaterThanEqual,
            Token::Less => BinaryOp::LessThan,
            Token::LessEqual => BinaryOp::LessThanEqual,
            Token::Plus => BinaryOp::Add,
            Token::Minus => BinaryOp::Sub,
            Token::Star => BinaryOp::Mul,
            Token::Slash => BinaryOp::Div,
            Token::And => BinaryOp::And,
            Token::Or => BinaryOp::Or,
            _ => unreachable!("Invalid token for binary operator"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AssignmentOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
}

impl<'a> From<&Token<'a>> for AssignmentOp {
    /// Constructs a `AssignmentOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid assignment or compound assignment operator.
    fn from(token: &Token<'a>) -> Self {
        match token {
            Token::Equal => AssignmentOp::Set,
            Token::PlusEqual => AssignmentOp::Add,
            Token::MinusEqual => AssignmentOp::Sub,
            Token::StarEqual => AssignmentOp::Mul,
            Token::SlashEqual => AssignmentOp::Div,
            _ => unreachable!("Invalid token for assignment or compound assignment operator"),
        }
    }
}

/// An expression in the Lox language.
#[derive(Debug, Clone)]
pub enum Expr<'a> {
    /// A literal value.
    Literal(Literal<'a>),

    /// An identifier referring to an in-scope symbol.
    Symbol { identifier: &'a str },

    /// A parenthesised group of expressions.
    Grouping(Box<Expr<'a>>),

    /// An assignment. Also includes compound assignments.
    Assignment {
        op: AssignmentOp,
        identifier: &'a str,
        expr: Box<Expr<'a>>,
    },

    /// A binary operation. Includes logical, relational, and arithmetic operations.
    Binary {
        left: Box<Expr<'a>>,
        op: BinaryOp,
        right: Box<Expr<'a>>,
    },

    /// A unary operation. Includes negation and logical NOT.
    Unary { op: UnaryOp, expr: Box<Expr<'a>> },

    /// A anonymous function (lambda) expression.
    Lambda {
        parameters: Vec<&'a str>,
        body: Box<Stmt<'a>>,
    },

    /// A function call. Can refer to an existing symbol or a nested expression.
    FunctionCall {
        expr: Box<Expr<'a>>,
        arguments: Vec<Expr<'a>>,
    },
}

pub type Block<'a> = Vec<Stmt<'a>>;

#[derive(Debug, Clone)]
pub struct VarDecl<'a> {
    pub identifier: &'a str,
    pub expr: Option<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct FunDecl<'a> {
    pub identifier: &'a str,
    pub parameters: Vec<&'a str>,
    pub body: Box<Stmt<'a>>,
}

#[derive(Debug, Clone)]
pub struct ClassDecl<'a> {
    pub identifier: &'a str,
    pub methods: Vec<FunDecl<'a>>,
    pub superclass: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct If<'a> {
    pub condition: Expr<'a>,
    pub branch: Box<Stmt<'a>>,
    pub else_branch: Option<Box<Stmt<'a>>>,
}

#[derive(Debug, Clone)]
pub struct While<'a> {
    pub condition: Expr<'a>,
    pub body: Box<Stmt<'a>>,
}

/// A statement in the Lox language.
#[derive(Debug, Clone)]
pub enum Stmt<'a> {
    /// An empty statement, represented by a single semicolon.
    Empty,

    /// An expression statement.
    Expr(Expr<'a>),

    /// A block of statements that get executed in a new scope.
    Block(Block<'a>),

    /// The `var`iable declaration statement.
    /// Inserts a new symbol in the current scope with an optional initial value that gets otherwise defaulted to "undefined" (not `nil`), similarly to javascript.
    VarDecl(VarDecl<'a>),

    /// The `fun`ction declaration statement.
    /// Inserts a new callable symbol in the current scope with the specified parameters and body.
    /// The function captures the current scope at the time of declaration, like a closure.
    FunDecl(FunDecl<'a>),

    /// The `class` declaration statement.
    /// Inserts a new class in the current scope with the specified data members and methods.
    ClassDecl(ClassDecl<'a>),

    /// The `print` statement.
    /// Prints the result of the expression to standard output.
    Print(Expr<'a>),

    /// The `if` statement.
    /// Conditionally executes a block of statements based on the value of the condition, with an optional `else` branch.
    If(If<'a>),

    /// The `while` loop.
    /// Also includes `for` loops, which are desugared into `while` loops.
    While(While<'a>),

    /// The `return` statement.
    /// Returns an optional value, otherwise defaulted to "undefined", from the current function.
    /// Only valid inside of functions.
    Return(Option<Expr<'a>>),

    /// The `break` statement.
    /// Stops the execution of the current loop.
    /// Only valid inside of loops.
    Break,

    /// A continue statement.
    /// Skips the remaining code in the current loop's body.
    /// Also executes the for loop's update statement if executed inside one.
    /// Only valid inside of loops.
    Continue,
}

impl<'a> Stmt<'a> {
    /// Wraps the statement in a block if it isn't already a block.
    fn wrapped_in_block(self) -> Self {
        match self {
            Self::Block(_) => self,
            other => Self::Block(vec![other]),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError<'a> {
    #[error("Expected closing parenthesis instead of {0}")]
    UnmatchedParens(&'a TokenInfo<'a>),

    #[error("Expected a semicolon instead of {0}.")]
    MissingSemicolon(&'a TokenInfo<'a>),

    #[error("Expected opening parenthesis after `if` instead of {0}.")]
    MissingLeftParen(&'a TokenInfo<'a>),

    #[error("Expected a symbol on the left-hand side of assignment instead of \"{0:?}\".")]
    InvalidAssignment(Expr<'a>),

    #[error("Break statements are only valid inside of loops.")]
    StrayBreakStatement,

    #[error("Continue statements are only valid inside of loops.")]
    StrayContinueStatement,

    #[error("Return statements are only valid inside of functions.")]
    StrayReturnStatement,

    #[error("Expected identifier instead of {0}.")]
    ExpectedIdentifier(&'a TokenInfo<'a>),

    #[error("Expected function body instead of {0}.")]
    ExpectedBody(&'a TokenInfo<'a>),

    #[error("Expected function parameter list instead of {0}.")]
    ExpectedFunctionParameterList(&'a TokenInfo<'a>),

    #[error("Expected identifier as parameter instead of {0}.")]
    ExpectedFunctionParameter(&'a TokenInfo<'a>),

    #[error("Expected method in function body instead of {0}.")]
    ExpectedMethod(&'a TokenInfo<'a>),

    #[error("Unexpected token {0}.")]
    UnexpectedToken(&'a TokenInfo<'a>),
}

#[derive(Debug, Clone)]
pub struct Parser<'a> {
    tokens: Peekable<Iter<'a, TokenInfo<'a>>>,
    // TODO: Move these into a later pass of the interpreter
    // is_parsing_loop: bool,
    // is_parsing_function: bool,
    // loop_update_statement: Option<Expr<'a>>,
}

/// Takes a token stream and a pattern, consuming and returning the next token if it matches the pattern - otherwise returns a peek of the next token.
macro_rules! chase {
    ($tokens:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {{
        if let Some(info) = $tokens.next_if(|next| match &next.token {
                $pattern $(if $guard)? => true,
                _ => false,
            }) {
            Found(&info.token)
        } else {
            NotFound($tokens.peek().expect("EOF can't be reached when chasing"))
        }
    }};
}

/// The type returned by the `chase!` macro.
enum Chased<'a> {
    Found(&'a Token<'a>),
    NotFound(&'a TokenInfo<'a>),
}
use Chased::*;

type ParseStmt<'a> = Result<Stmt<'a>, ParseError<'a>>;
type ParseExpr<'a> = Result<Expr<'a>, ParseError<'a>>;

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [TokenInfo<'a>]) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
        }
    }

    pub fn parse(self) -> (Vec<Stmt<'a>>, Vec<ParseError<'a>>) {
        self.parse_program()
    }

    fn parse_program(mut self) -> (Vec<Stmt<'a>>, Vec<ParseError<'a>>) {
        let mut statements = Vec::new();
        let mut errors = Vec::new();

        while let NotFound(_) = chase!(self.tokens, Token::EndOfFile) {
            match self.parse_statement() {
                Ok(statement) => {
                    statements.push(statement);
                }
                Err(error) => {
                    errors.push(error);
                    self.synchronize();
                }
            }
        }

        (statements, errors)
    }

    fn parse_statement(&mut self) -> ParseStmt<'a> {
        match chase!(
            self.tokens,
            Token::Semicolon
                | Token::LeftBrace
                | Token::Var
                | Token::Fun
                | Token::Class
                | Token::Print
                | Token::If
                | Token::While
                | Token::For
                | Token::Return
                | Token::Break
                | Token::Continue
        ) {
            Found(Token::Semicolon) => self.parse_empty(),
            Found(Token::LeftBrace) => self.parse_block(),
            Found(Token::Var) => self.parse_var_decl(),
            Found(Token::Fun) => self.parse_fun_decl(),
            Found(Token::Class) => self.parse_class_decl(),
            Found(Token::Print) => self.parse_print(),
            Found(Token::If) => self.parse_if(),
            Found(Token::While) => self.parse_while(),
            Found(Token::For) => self.parse_for(),
            Found(Token::Return) => self.parse_return(),
            Found(Token::Break) => self.parse_break(),
            Found(Token::Continue) => self.parse_continue(),
            NotFound(_) => self.parse_expr(),
            _ => unreachable!(),
        }
    }

    fn parse_empty(&mut self) -> ParseStmt<'a> {
        // No-op
        Ok(Stmt::Empty)
    }

    fn parse_block(&mut self) -> ParseStmt<'a> {
        let mut statements = Vec::new();
        while let NotFound(_) = chase!(self.tokens, Token::RightBrace) {
            statements.push(self.parse_statement()?);
        }
        Ok(Stmt::Block(statements))
    }

    fn parse_var_decl(&mut self) -> ParseStmt<'a> {
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            NotFound(info) => return Err(ParseError::ExpectedIdentifier(info)),
            _ => unreachable!(),
        };

        let expr = if let Found(_) = chase!(self.tokens, Token::Equal) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::VarDecl(VarDecl { identifier, expr })),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_fun_decl(&mut self) -> ParseStmt<'a> {
        // Identifier
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => *str,
            // We didn't find an identifier but we might have found a lambda
            NotFound(info) => match self.parse_lambda() {
                // If a lambda was indeed found, treat it as a statement-expression
                Ok(lambda) => match chase!(self.tokens, Token::Semicolon) {
                    Found(_) => return Ok(Stmt::Expr(lambda)),
                    NotFound(info) => return Err(ParseError::MissingSemicolon(info)),
                },
                // Otherwise just return an error
                Err(_) => return Err(ParseError::ExpectedIdentifier(info)),
            },
            _ => unreachable!(),
        };

        // Parameters
        let parameters = match chase!(self.tokens, Token::LeftParen) {
            Found(_) => self.parse_parameters()?,
            NotFound(info) => return Err(ParseError::ExpectedFunctionParameterList(info)),
        };

        // Body
        let body = match chase!(self.tokens, Token::LeftBrace) {
            Found(_) => self.parse_block()?,
            NotFound(info) => return Err(ParseError::ExpectedBody(info)),
        };

        Ok(Stmt::FunDecl(FunDecl {
            identifier,
            parameters,
            body: Box::new(body),
        }))
    }

    fn parse_class_decl(&mut self) -> ParseStmt<'a> {
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            NotFound(info) => return Err(ParseError::ExpectedIdentifier(info)),
            _ => unreachable!(),
        };

        let superclass = if let Found(_) = chase!(self.tokens, Token::Less) {
            match chase!(self.tokens, Token::Identifier(_)) {
                Found(Token::Identifier(str)) => Some(*str),
                NotFound(info) => return Err(ParseError::ExpectedIdentifier(info)),
                _ => unreachable!(),
            }
        } else {
            None
        };

        if let NotFound(info) = chase!(self.tokens, Token::LeftBrace) {
            return Err(ParseError::ExpectedBody(info));
        }

        let mut methods = Vec::new();
        while let NotFound(_) = chase!(self.tokens, Token::RightBrace) {
            if let Stmt::FunDecl(fun_decl) = self.parse_fun_decl()? {
                methods.push(fun_decl);
            }
        }

        Ok(Stmt::ClassDecl(ClassDecl {
            identifier,
            methods,
            superclass,
        }))
    }

    fn parse_print(&mut self) -> ParseStmt<'a> {
        let expr = self.parse_expression()?;

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Print(expr)),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_if(&mut self) -> ParseStmt<'a> {
        if let NotFound(info) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(info));
        };

        let condition = self.parse_expression()?;

        if let NotFound(info) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::UnmatchedParens(info));
        };

        let branch = self.parse_statement()?.wrapped_in_block();

        match chase!(self.tokens, Token::Else) {
            Found(_) => {
                let else_branch = self.parse_statement()?.wrapped_in_block();

                Ok(Stmt::If(If {
                    condition,
                    branch: Box::new(branch),
                    else_branch: Some(Box::new(else_branch)),
                }))
            }
            NotFound(_) => Ok(Stmt::If(If {
                condition,
                branch: Box::new(branch),
                else_branch: None,
            })),
        }
    }

    fn parse_while(&mut self) -> ParseStmt<'a> {
        if let NotFound(info) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(info));
        };

        let condition = self.parse_expression()?;

        if let NotFound(info) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::UnmatchedParens(info));
        };

        let body = self.parse_statement()?;

        Ok(Stmt::While(While {
            condition,
            body: Box::new(body),
        }))
    }

    fn parse_for(&mut self) -> ParseStmt<'a> {
        if let NotFound(info) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(info));
        };

        let initializer = match chase!(self.tokens, Token::Var | Token::Semicolon) {
            Found(Token::Var) => self.parse_var_decl()?,
            Found(Token::Semicolon) => self.parse_empty()?,
            NotFound(_) => self.parse_expr()?,
            _ => unreachable!(),
        };

        let test = match chase!(self.tokens, Token::Semicolon) {
            Found(_) => self.parse_empty()?,
            NotFound(_) => self.parse_expr()?,
        };

        let update = match chase!(self.tokens, Token::RightParen) {
            Found(_) => None,
            NotFound(_) => {
                let expr = self.parse_expression()?;
                if let NotFound(info) = chase!(self.tokens, Token::RightParen) {
                    return Err(ParseError::UnmatchedParens(info));
                };
                Some(expr)
            }
        };

        let mut body = self.parse_statement()?.wrapped_in_block();

        let desugared = {
            let condition = if let Stmt::Expr(expr) = test {
                expr
            } else {
                Expr::Literal(Literal::True)
            };

            if let Stmt::Block(statements) = &mut body {
                if let Some(update) = update {
                    statements.push(Stmt::Expr(update));
                }
            }

            Stmt::Block(vec![
                initializer,
                Stmt::While(While {
                    condition,
                    body: Box::new(body),
                }),
            ])
        };

        Ok(desugared)
    }

    fn parse_return(&mut self) -> ParseStmt<'a> {
        let expr = if let NotFound(_) = chase!(self.tokens, Token::Semicolon) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Return(expr)),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_break(&mut self) -> ParseStmt<'a> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Break),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_continue(&mut self) -> ParseStmt<'a> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Continue),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_expr(&mut self) -> ParseStmt<'a> {
        let expr = self.parse_expression()?;
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Expr(expr)),
            NotFound(info) => Err(ParseError::MissingSemicolon(info)),
        }
    }

    fn parse_expression(&mut self) -> ParseExpr<'a> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> ParseExpr<'a> {
        let expr = self.parse_equality()?;

        match chase!(
            self.tokens,
            Token::Equal
                | Token::PlusEqual
                | Token::MinusEqual
                | Token::StarEqual
                | Token::SlashEqual
        ) {
            Found(token) => {
                let value = self.parse_assignment()?;

                if let Expr::Symbol { identifier } = expr {
                    Ok(Expr::Assignment {
                        op: AssignmentOp::from(token),
                        identifier,
                        expr: Box::new(value),
                    })
                } else {
                    Err(ParseError::InvalidAssignment(expr))
                }
            }
            NotFound(_) => Ok(expr),
        }
    }

    fn parse_equality(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_comparison()?;

        while let Found(token) = chase!(self.tokens, Token::BangEqual | Token::EqualEqual) {
            let op = BinaryOp::from(token);
            let right = self.parse_comparison()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_logical()?;

        while let Found(token) = chase!(
            self.tokens,
            Token::Greater | Token::GreaterEqual | Token::Less | Token::LessEqual
        ) {
            let op = BinaryOp::from(token);
            let right = self.parse_logical()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_logical(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_term()?;

        while let Found(token) = chase!(self.tokens, Token::And | Token::Or) {
            let op = BinaryOp::from(token);
            let right = self.parse_logical()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_factor()?;

        while let Found(token) = chase!(self.tokens, Token::Minus | Token::Plus) {
            let op = BinaryOp::from(token);
            let right = self.parse_factor()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_unary()?;

        while let Found(token) = chase!(self.tokens, Token::Slash | Token::Star) {
            let op = BinaryOp::from(token);
            let right = self.parse_unary()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> ParseExpr<'a> {
        if let Found(token) = chase!(self.tokens, Token::Bang | Token::Minus) {
            let op = UnaryOp::from(token);
            let right = self.parse_unary()?;
            return Ok(Expr::Unary {
                op,
                expr: Box::new(right),
            });
        }

        self.parse_call()
    }

    fn parse_call(&mut self) -> ParseExpr<'a> {
        let mut expr = self.parse_primary()?;

        while let Found(_) = chase!(self.tokens, Token::LeftParen) {
            let arguments = self.parse_arguments()?;
            expr = Expr::FunctionCall {
                expr: Box::new(expr),
                arguments,
            };
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> ParseExpr<'a> {
        match chase!(
            self.tokens,
            Token::Number(_)
                | Token::String(_)
                | Token::Identifier(_)
                | Token::True
                | Token::False
                | Token::Nil
                | Token::LeftParen
                | Token::Fun
        ) {
            Found(Token::Number(num)) => Ok(Expr::Literal(Literal::Number(*num))),
            Found(Token::String(str)) => Ok(Expr::Literal(Literal::String(str))),
            Found(Token::Identifier(identifier)) => Ok(Expr::Symbol { identifier }),
            Found(Token::True) => Ok(Expr::Literal(Literal::True)),
            Found(Token::False) => Ok(Expr::Literal(Literal::False)),
            Found(Token::Nil) => Ok(Expr::Literal(Literal::Nil)),
            Found(Token::LeftParen) => {
                let expr = self.parse_expression()?;
                match chase!(self.tokens, Token::RightParen) {
                    Found(_) => Ok(Expr::Grouping(Box::new(expr))),
                    NotFound(info) => Err(ParseError::UnmatchedParens(info)),
                }
            }
            Found(Token::Fun) => self.parse_lambda(),
            NotFound(info) => return Err(ParseError::UnexpectedToken(info)),
            _ => unreachable!(),
        }
    }

    fn parse_lambda(&mut self) -> ParseExpr<'a> {
        // Parameters (optional)
        let parameters = match chase!(self.tokens, Token::LeftParen) {
            Found(_) => self.parse_parameters()?,
            NotFound(_) => vec![],
        };

        // Body
        let body = match chase!(self.tokens, Token::LeftBrace) {
            Found(_) => self.parse_block()?,
            NotFound(info) => return Err(ParseError::ExpectedBody(info)),
        };

        Ok(Expr::Lambda {
            parameters,
            body: Box::new(body),
        })
    }

    fn parse_arguments(&mut self) -> Result<Vec<Expr<'a>>, ParseError<'a>> {
        let mut arguments = vec![];

        if let NotFound(_) = chase!(self.tokens, Token::RightParen) {
            arguments.push(self.parse_expression()?);
            while let Found(_) = chase!(self.tokens, Token::Comma) {
                arguments.push(self.parse_expression()?);
            }

            if let NotFound(info) = chase!(self.tokens, Token::RightParen) {
                return Err(ParseError::UnmatchedParens(info));
            }
        }

        Ok(arguments)
    }

    fn parse_parameters(&mut self) -> Result<Vec<&'a str>, ParseError<'a>> {
        let mut parameters = vec![];

        if let NotFound(_) = chase!(self.tokens, Token::RightParen) {
            match chase!(self.tokens, Token::Identifier(_)) {
                Found(Token::Identifier(str)) => parameters.push(*str),
                NotFound(info) => return Err(ParseError::ExpectedFunctionParameter(info)),
                _ => unreachable!(),
            }

            while let Found(_) = chase!(self.tokens, Token::Comma) {
                match chase!(self.tokens, Token::Identifier(_)) {
                    Found(Token::Identifier(str)) => parameters.push(*str),
                    NotFound(info) => return Err(ParseError::ExpectedFunctionParameter(info)),
                    _ => unreachable!(),
                }
            }

            if let NotFound(info) = chase!(self.tokens, Token::RightParen) {
                return Err(ParseError::UnmatchedParens(info));
            }
        }

        Ok(parameters)
    }

    fn synchronize(&mut self) {
        // Consume tokens until we either a semicolon or the beginning of a statement is found
        while let Some(info) = self.tokens.peek() {
            match &info.token {
                Token::EndOfFile => return,
                token if Self::begins_statement(token) => return,
                _ => {
                    // Consume
                    self.tokens.next();
                }
            }
        }
    }

    fn begins_statement(token: &Token) -> bool {
        matches!(
            token,
            Token::Class
                | Token::Fun
                | Token::Var
                | Token::For
                | Token::If
                | Token::While
                | Token::Print
                | Token::Return
                | Token::LeftBrace
        )
    }
}
