use core::slice::Iter;
use std::{convert::From, iter::Peekable};

use crate::{
    lex::{ContextualizedToken, Token},
    LoxNumber,
};

/// Takes a token stream and a pattern, consuming and returning the next token if it matches the pattern - otherwise returns a peek of the next token.
macro_rules! chase {
    ($tokens:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {{
        if let Some(ctx) = $tokens.next_if(|next| match &next.token {
                $pattern $(if $guard)? => true,
                _ => false,
            }) {
            Found(ctx.token)
        } else {
            NotFound(**$tokens.peek().expect("EOF can't be reached when chasing"))
        }
    }};
}

#[derive(Debug, Clone)]
pub struct Parser<'a> {
    tokens: Peekable<Iter<'a, ContextualizedToken<'a>>>,
    // TODO: Move these into a later pass of the interpreter
    // is_parsing_loop: bool,
    // is_parsing_function: bool,
    // loop_update_statement: Option<Expr<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [ContextualizedToken<'a>]) -> Self {
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
            /*      Token            Statement */
            Token::LeftBrace        // Block
                | Token::Break      // Break
                | Token::Class      // ClassDecl
                | Token::Continue   // Continue
                | Token::Semicolon  // Empty
                | Token::Var        // VarDecl
                | Token::Fun        // FunDecl
                | Token::Print      // Print
                | Token::If         // If
                | Token::Return     // Return
                | Token::While      // While
                | Token::For // While (sugared)
        ) {
            Found(Token::LeftBrace) => self.parse_block(),
            Found(Token::Break) => self.parse_break(),
            Found(Token::Class) => self.parse_class_decl(),
            Found(Token::Continue) => self.parse_continue(),
            Found(Token::Semicolon) => self.parse_empty(),
            Found(Token::Fun) => self.parse_fun_decl(),
            Found(Token::If) => self.parse_if(),
            Found(Token::Print) => self.parse_print(),
            Found(Token::Return) => self.parse_return(),
            Found(Token::Var) => self.parse_var_decl(),
            Found(Token::While) => self.parse_while(),
            Found(Token::For) => self.parse_for(),
            NotFound(_) => self.parse_expr(),
            _ => unreachable!(),
        }
    }

    fn parse_block(&mut self) -> ParseStmt<'a> {
        let mut statements = Vec::new();
        while let NotFound(_) = chase!(self.tokens, Token::RightBrace) {
            statements.push(self.parse_statement()?);
        }
        Ok(Stmt::Block { statements })
    }

    fn parse_break(&mut self) -> ParseStmt<'a> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Break),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
        }
    }

    fn parse_class_decl(&mut self) -> ParseStmt<'a> {
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            NotFound(ctx) => return Err(ParseError::ExpectedIdentifier(ctx)),
            _ => unreachable!(),
        };

        let superclass = if let Found(_) = chase!(self.tokens, Token::Less) {
            match chase!(self.tokens, Token::Identifier(_)) {
                Found(Token::Identifier(str)) => Some(str),
                NotFound(ctx) => return Err(ParseError::ExpectedIdentifier(ctx)),
                _ => unreachable!(),
            }
        } else {
            None
        };

        if let NotFound(ctx) = chase!(self.tokens, Token::LeftBrace) {
            return Err(ParseError::ExpectedBody(ctx));
        }

        let mut methods = Vec::new();
        while let NotFound(_) = chase!(self.tokens, Token::RightBrace) {
            if let Stmt::FunDecl {
                identifier,
                parameters,
                body,
            } = self.parse_fun_decl()?
            {
                methods.push((identifier, parameters, body));
            }
        }

        Ok(Stmt::ClassDecl {
            identifier,
            methods,
            superclass,
        })
    }

    fn parse_continue(&mut self) -> ParseStmt<'a> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Continue),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
        }
    }

    fn parse_empty(&mut self) -> ParseStmt<'a> {
        // No-op
        Ok(Stmt::Empty)
    }

    fn parse_fun_decl(&mut self) -> ParseStmt<'a> {
        // Identifier
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            // We didn't find an identifier but we might have found a lambda
            NotFound(ctx) => match self.parse_lambda() {
                // If a lambda was indeed found, treat it as a statement-expression
                Ok(lambda) => match chase!(self.tokens, Token::Semicolon) {
                    Found(_) => return Ok(Stmt::Expr { expr: lambda }),
                    NotFound(ctx) => return Err(ParseError::MissingSemicolon(ctx)),
                },
                // Otherwise just return an error
                Err(_) => return Err(ParseError::ExpectedIdentifier(ctx)),
            },
            _ => unreachable!(),
        };

        // Parameters
        let parameters = match chase!(self.tokens, Token::LeftParen) {
            Found(_) => self.parse_parameters()?,
            NotFound(ctx) => return Err(ParseError::ExpectedFunctionParameterList(ctx)),
        };

        // Body
        if let Stmt::Block { statements } = match chase!(self.tokens, Token::LeftBrace) {
            Found(_) => self.parse_block()?,
            NotFound(ctx) => return Err(ParseError::ExpectedBody(ctx)),
        } {
            Ok(Stmt::FunDecl {
                identifier,
                parameters,
                body: statements,
            })
        } else {
            unreachable!()
        }
    }

    fn parse_if(&mut self) -> ParseStmt<'a> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(ctx));
        };

        let condition = self.parse_expression()?;

        if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::UnmatchedParens(ctx));
        };

        let branch = self.parse_statement()?.wrapped_in_block();

        match chase!(self.tokens, Token::Else) {
            Found(_) => {
                let else_branch = self.parse_statement()?.wrapped_in_block();

                Ok(Stmt::If {
                    condition,
                    branch: Box::new(branch),
                    else_branch: Some(Box::new(else_branch)),
                })
            }
            NotFound(_) => Ok(Stmt::If {
                condition,
                branch: Box::new(branch),
                else_branch: None,
            }),
        }
    }

    fn parse_print(&mut self) -> ParseStmt<'a> {
        let expr = self.parse_expression()?;

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Print { expr }),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
        }
    }

    fn parse_return(&mut self) -> ParseStmt<'a> {
        let expr = if let NotFound(_) = chase!(self.tokens, Token::Semicolon) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Return { expr }),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
        }
    }

    fn parse_var_decl(&mut self) -> ParseStmt<'a> {
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            NotFound(ctx) => return Err(ParseError::ExpectedIdentifier(ctx)),
            _ => unreachable!(),
        };

        let expr = if let Found(_) = chase!(self.tokens, Token::Equal) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::VarDecl { identifier, expr }),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
        }
    }

    fn parse_while(&mut self) -> ParseStmt<'a> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(ctx));
        };

        let condition = self.parse_expression()?;

        if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::UnmatchedParens(ctx));
        };

        let body = self.parse_statement()?;

        Ok(Stmt::While {
            condition,
            body: Box::new(body),
        })
    }

    fn parse_for(&mut self) -> ParseStmt<'a> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::MissingLeftParen(ctx));
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
                if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
                    return Err(ParseError::UnmatchedParens(ctx));
                };
                Some(expr)
            }
        };

        let mut body = self.parse_statement()?.wrapped_in_block();

        let desugared = {
            let condition = if let Stmt::Expr { expr } = test {
                expr
            } else {
                Expr::Literal(Literal::True)
            };

            if let Stmt::Block { statements } = &mut body {
                if let Some(update) = update {
                    statements.push(Stmt::Expr { expr: update });
                }
            }

            Stmt::Block {
                statements: vec![
                    initializer,
                    Stmt::While {
                        condition,
                        body: Box::new(body),
                    },
                ],
            }
        };

        Ok(desugared)
    }

    fn parse_expr(&mut self) -> ParseStmt<'a> {
        let expr = self.parse_expression()?;

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Expr { expr }),
            NotFound(ctx) => Err(ParseError::MissingSemicolon(ctx)),
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
                let op = AssignmentOp::from(token);

                match expr {
                    Expr::Symbol { identifier } => Ok(Expr::Assignment {
                        op,
                        identifier,
                        expr: Box::new(value),
                    }),
                    Expr::PropertyAccess { expr, property } => Ok(Expr::PropertyAssignment {
                        object: expr,
                        property,
                        op,
                        value: Box::new(value),
                    }),
                    _ => Err(ParseError::InvalidAssignment(expr)),
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

        while let Found(token) = chase!(self.tokens, Token::LeftParen | Token::Dot) {
            match token {
                Token::LeftParen => {
                    let arguments = self.parse_arguments()?;
                    expr = Expr::FunctionCall {
                        expr: Box::new(expr),
                        arguments,
                    };
                }
                Token::Dot => {
                    let identifier = match chase!(self.tokens, Token::Identifier(_)) {
                        Found(Token::Identifier(str)) => str,
                        NotFound(ctx) => return Err(ParseError::ExpectedIdentifier(ctx)),
                        _ => unreachable!(),
                    };
                    expr = Expr::PropertyAccess {
                        expr: Box::new(expr),
                        property: identifier,
                    };
                }
                _ => unreachable!(),
            }
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
            Found(Token::Number(num)) => Ok(Expr::Literal(Literal::Number(num))),
            Found(Token::String(str)) => Ok(Expr::Literal(Literal::String(str))),
            Found(Token::Identifier(identifier)) => Ok(Expr::Symbol { identifier }),
            Found(Token::True) => Ok(Expr::Literal(Literal::True)),
            Found(Token::False) => Ok(Expr::Literal(Literal::False)),
            Found(Token::Nil) => Ok(Expr::Literal(Literal::Nil)),
            Found(Token::LeftParen) => {
                let expr = self.parse_expression()?;
                match chase!(self.tokens, Token::RightParen) {
                    Found(_) => Ok(Expr::Grouping {
                        expr: Box::new(expr),
                    }),
                    NotFound(ctx) => Err(ParseError::UnmatchedParens(ctx)),
                }
            }
            Found(Token::Fun) => self.parse_lambda(),
            NotFound(ctx) => return Err(ParseError::UnexpectedToken(ctx)),
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
        if let Stmt::Block { statements } = match chase!(self.tokens, Token::LeftBrace) {
            Found(_) => self.parse_block()?,
            NotFound(ctx) => return Err(ParseError::ExpectedBody(ctx)),
        } {
            Ok(Expr::Lambda {
                parameters,
                body: statements,
            })
        } else {
            unreachable!();
        }
    }

    fn parse_arguments(&mut self) -> Result<Vec<Expr<'a>>, ParseError<'a>> {
        let mut arguments = vec![];

        if let NotFound(_) = chase!(self.tokens, Token::RightParen) {
            arguments.push(self.parse_expression()?);
            while let Found(_) = chase!(self.tokens, Token::Comma) {
                arguments.push(self.parse_expression()?);
            }

            if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
                return Err(ParseError::UnmatchedParens(ctx));
            }
        }

        Ok(arguments)
    }

    fn parse_parameters(&mut self) -> Result<Vec<&'a str>, ParseError<'a>> {
        let mut parameters = vec![];

        if let NotFound(_) = chase!(self.tokens, Token::RightParen) {
            match chase!(self.tokens, Token::Identifier(_)) {
                Found(Token::Identifier(str)) => parameters.push(str),
                NotFound(ctx) => return Err(ParseError::ExpectedFunctionParameter(ctx)),
                _ => unreachable!(),
            }

            while let Found(_) = chase!(self.tokens, Token::Comma) {
                match chase!(self.tokens, Token::Identifier(_)) {
                    Found(Token::Identifier(str)) => parameters.push(str),
                    NotFound(ctx) => return Err(ParseError::ExpectedFunctionParameter(ctx)),
                    _ => unreachable!(),
                }
            }

            if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
                return Err(ParseError::UnmatchedParens(ctx));
            }
        }

        Ok(parameters)
    }

    fn synchronize(&mut self) {
        fn begins_statement(token: Token) -> bool {
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

        // Consume tokens until we either a semicolon or the beginning of a statement is found
        while let Some(ctx) = self.tokens.peek() {
            match ctx.token {
                Token::EndOfFile => return,
                token if begins_statement(token) => return,
                _ => {
                    // Consume
                    self.tokens.next();
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt<'a> {
    Block {
        statements: Block<'a>,
    },
    Break,
    ClassDecl {
        identifier: &'a str,
        methods: Vec<(&'a str, Vec<&'a str>, Block<'a>)>,
        superclass: Option<&'a str>,
    },
    Continue,
    Empty,
    Expr {
        expr: Expr<'a>,
    },
    FunDecl {
        identifier: &'a str,
        parameters: Vec<&'a str>,
        body: Block<'a>,
    },
    If {
        condition: Expr<'a>,
        branch: Box<Stmt<'a>>,
        else_branch: Option<Box<Stmt<'a>>>,
    },
    Print {
        expr: Expr<'a>,
    },
    Return {
        expr: Option<Expr<'a>>,
    },
    VarDecl {
        identifier: &'a str,
        expr: Option<Expr<'a>>,
    },
    While {
        condition: Expr<'a>,
        body: Box<Stmt<'a>>,
    },
}

impl<'a> Stmt<'a> {
    /// Wraps the statement in a block if it isn't already a block.
    fn wrapped_in_block(self) -> Self {
        match self {
            Self::Block { .. } => self,
            other => Self::Block {
                statements: vec![other],
            },
        }
    }
}

pub type Block<'a> = Vec<Stmt<'a>>;

#[derive(Debug, Clone)]
pub enum Expr<'a> {
    Assignment {
        op: AssignmentOp,
        identifier: &'a str,
        expr: Box<Expr<'a>>,
    },
    Binary {
        left: Box<Expr<'a>>,
        op: BinaryOp,
        right: Box<Expr<'a>>,
    },
    FunctionCall {
        expr: Box<Expr<'a>>,
        arguments: Vec<Expr<'a>>,
    },
    Grouping {
        expr: Box<Expr<'a>>,
    },
    Lambda {
        parameters: Vec<&'a str>,
        body: Block<'a>,
    },
    Literal(Literal<'a>),
    PropertyAccess {
        expr: Box<Expr<'a>>,
        property: &'a str,
    },
    PropertyAssignment {
        object: Box<Expr<'a>>,
        property: &'a str,
        op: AssignmentOp,
        value: Box<Expr<'a>>,
    },
    Symbol {
        identifier: &'a str,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr<'a>>,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError<'a> {
    #[error("Expected closing parenthesis instead of {0}")]
    UnmatchedParens(ContextualizedToken<'a>),

    #[error("Expected a semicolon instead of {0}.")]
    MissingSemicolon(ContextualizedToken<'a>),

    #[error("Expected opening parenthesis after `if` instead of {0}.")]
    MissingLeftParen(ContextualizedToken<'a>),

    #[error("Expected a symbol on the left-hand side of assignment instead of \"{0:?}\".")]
    InvalidAssignment(Expr<'a>),

    #[error("Break statements are only valid inside of loops.")]
    StrayBreakStatement,

    #[error("Continue statements are only valid inside of loops.")]
    StrayContinueStatement,

    #[error("Return statements are only valid inside of functions.")]
    StrayReturnStatement,

    #[error("Expected identifier instead of {0}.")]
    ExpectedIdentifier(ContextualizedToken<'a>),

    #[error("Expected function body instead of {0}.")]
    ExpectedBody(ContextualizedToken<'a>),

    #[error("Expected function parameter list instead of {0}.")]
    ExpectedFunctionParameterList(ContextualizedToken<'a>),

    #[error("Expected identifier as parameter instead of {0}.")]
    ExpectedFunctionParameter(ContextualizedToken<'a>),

    #[error("Expected method in function body instead of {0}.")]
    ExpectedMethod(ContextualizedToken<'a>),

    #[error("Unexpected token {0}.")]
    UnexpectedToken(ContextualizedToken<'a>),
}

/// The type returned by the `chase!` macro.
enum Chased<'a> {
    Found(Token<'a>),
    NotFound(ContextualizedToken<'a>),
}

use Chased::*;

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

impl<'a> From<Token<'a>> for UnaryOp {
    /// Constructs a `UnaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid unary operator.
    fn from(token: Token<'a>) -> Self {
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
    Div,
    Mul,
    Sub,
}

impl<'a> From<Token<'a>> for BinaryOp {
    /// Constructs a `BinaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid binary operator.
    fn from(token: Token<'a>) -> Self {
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

impl<'a> From<Token<'a>> for AssignmentOp {
    /// Constructs a `AssignmentOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid assignment or compound assignment operator.
    fn from(token: Token<'a>) -> Self {
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

type ParseStmt<'a> = Result<Stmt<'a>, ParseError<'a>>;
type ParseExpr<'a> = Result<Expr<'a>, ParseError<'a>>;
