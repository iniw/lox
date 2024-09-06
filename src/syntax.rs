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
pub struct Parser<'lex> {
    tokens: Peekable<Iter<'lex, ContextualizedToken<'lex>>>,
    is_parsing_loop: bool,
    is_parsing_function: bool,
}

impl<'lex> Parser<'lex> {
    pub fn new(tokens: &'lex [ContextualizedToken<'lex>]) -> Self {
        Parser {
            tokens: tokens.iter().peekable(),
            is_parsing_loop: false,
            is_parsing_function: false,
        }
    }

    pub fn parse(self) -> (Vec<Stmt<'lex>>, Vec<ParseError<'lex>>) {
        self.parse_program()
    }

    fn parse_program(mut self) -> (Vec<Stmt<'lex>>, Vec<ParseError<'lex>>) {
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

    fn parse_statement(&mut self) -> ParseStmt<'lex> {
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
            Found(Token::Fun) => {
                self.is_parsing_function = true;
                let ret = self.parse_fun_decl();
                self.is_parsing_function = false;
                ret
            }
            Found(Token::If) => self.parse_if(),
            Found(Token::Print) => self.parse_print(),
            Found(Token::Return) => self.parse_return(),
            Found(Token::Var) => self.parse_var_decl(),
            Found(Token::While) => {
                self.is_parsing_loop = true;
                let ret = self.parse_while();
                self.is_parsing_loop = false;
                ret
            }
            Found(Token::For) => {
                self.is_parsing_loop = true;
                let ret = self.parse_for();
                self.is_parsing_loop = false;
                ret
            }
            NotFound(_) => self.parse_expr(),
            _ => unreachable!(),
        }
    }

    fn parse_block(&mut self) -> ParseStmt<'lex> {
        let mut statements = Vec::new();
        while let NotFound(_) = chase!(self.tokens, Token::RightBrace) {
            statements.push(self.parse_statement()?);
        }
        Ok(Stmt::Block { statements })
    }

    fn parse_break(&mut self) -> ParseStmt<'lex> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => {
                if self.is_parsing_loop {
                    Ok(Stmt::Break)
                } else {
                    Err(ParseError::StrayBreakStatement)
                }
            }
            NotFound(ctx) => Err(ParseError::ExpectedSemicolon(ctx)),
        }
    }

    fn parse_class_decl(&mut self) -> ParseStmt<'lex> {
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

    fn parse_continue(&mut self) -> ParseStmt<'lex> {
        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => {
                if self.is_parsing_loop {
                    Ok(Stmt::Continue)
                } else {
                    Err(ParseError::StrayContinueStatement)
                }
            }
            NotFound(ctx) => Err(ParseError::ExpectedSemicolon(ctx)),
        }
    }

    fn parse_empty(&mut self) -> ParseStmt<'lex> {
        // No-op
        Ok(Stmt::Empty)
    }

    fn parse_fun_decl(&mut self) -> ParseStmt<'lex> {
        // Identifier
        let identifier = match chase!(self.tokens, Token::Identifier(_)) {
            Found(Token::Identifier(str)) => str,
            // We didn't find an identifier but we might have found a lambda
            NotFound(ctx) => match self.parse_lambda() {
                // If a lambda was indeed found, treat it as a statement-expression
                Ok(lambda) => match chase!(self.tokens, Token::Semicolon) {
                    Found(_) => return Ok(Stmt::Expr { expr: lambda }),
                    NotFound(ctx) => return Err(ParseError::ExpectedSemicolon(ctx)),
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

    fn parse_if(&mut self) -> ParseStmt<'lex> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::ExpectedParenLeft(ctx));
        };

        let condition = self.parse_expression()?;

        if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::ExpectedParenRight(ctx));
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

    fn parse_print(&mut self) -> ParseStmt<'lex> {
        let expr = self.parse_expression()?;

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Print { expr }),
            NotFound(ctx) => Err(ParseError::ExpectedSemicolon(ctx)),
        }
    }

    fn parse_return(&mut self) -> ParseStmt<'lex> {
        let expr = if let NotFound(_) = chase!(self.tokens, Token::Semicolon) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        if let Some(expr) = expr {
            match chase!(self.tokens, Token::Semicolon) {
                Found(_) => {
                    if self.is_parsing_function {
                        return Ok(Stmt::Return { expr: Some(expr) });
                    } else {
                        return Err(ParseError::StrayReturnStatement);
                    }
                }
                NotFound(ctx) => return Err(ParseError::ExpectedSemicolon(ctx)),
            }
        }

        if self.is_parsing_function {
            Ok(Stmt::Return { expr: None })
        } else {
            Err(ParseError::StrayReturnStatement)
        }
    }

    fn parse_var_decl(&mut self) -> ParseStmt<'lex> {
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
            NotFound(ctx) => Err(ParseError::ExpectedSemicolon(ctx)),
        }
    }

    fn parse_while(&mut self) -> ParseStmt<'lex> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::ExpectedParenLeft(ctx));
        };

        let condition = self.parse_expression()?;

        if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
            return Err(ParseError::ExpectedParenRight(ctx));
        };

        let body = self.parse_statement()?;

        Ok(Stmt::While {
            condition,
            body: Box::new(body),
        })
    }

    fn parse_for(&mut self) -> ParseStmt<'lex> {
        if let NotFound(ctx) = chase!(self.tokens, Token::LeftParen) {
            return Err(ParseError::ExpectedParenLeft(ctx));
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
                    return Err(ParseError::ExpectedParenRight(ctx));
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

    fn parse_expr(&mut self) -> ParseStmt<'lex> {
        let expr = self.parse_expression()?;

        match chase!(self.tokens, Token::Semicolon) {
            Found(_) => Ok(Stmt::Expr { expr }),
            NotFound(ctx) => Err(ParseError::ExpectedSemicolon(ctx)),
        }
    }

    fn parse_expression(&mut self) -> ParseExpr<'lex> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> ParseExpr<'lex> {
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

    fn parse_equality(&mut self) -> ParseExpr<'lex> {
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

    fn parse_comparison(&mut self) -> ParseExpr<'lex> {
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

    fn parse_logical(&mut self) -> ParseExpr<'lex> {
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

    fn parse_term(&mut self) -> ParseExpr<'lex> {
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

    fn parse_factor(&mut self) -> ParseExpr<'lex> {
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

    fn parse_unary(&mut self) -> ParseExpr<'lex> {
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

    fn parse_call(&mut self) -> ParseExpr<'lex> {
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

    fn parse_primary(&mut self) -> ParseExpr<'lex> {
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
                    NotFound(ctx) => Err(ParseError::ExpectedParenRight(ctx)),
                }
            }
            Found(Token::Fun) => {
                self.is_parsing_function = true;
                let ret = self.parse_lambda();
                self.is_parsing_function = false;
                ret
            }
            NotFound(ctx) => return Err(ParseError::UnexpectedToken(ctx)),
            _ => unreachable!(),
        }
    }

    fn parse_lambda(&mut self) -> ParseExpr<'lex> {
        // Parameters (optional)
        let parameters = match chase!(self.tokens, Token::LeftParen) {
            Found(_) => self.parse_parameters()?,
            NotFound(_) => vec![],
        };

        // Body
        let Stmt::Block { statements } = (match chase!(self.tokens, Token::LeftBrace) {
            Found(_) => self.parse_block()?,
            NotFound(ctx) => {
                let expr = match self.parse_expression() {
                    Ok(expr) => expr,
                    Err(_) => return Err(ParseError::ExpectedBody(ctx)),
                };
                Stmt::Block {
                    statements: vec![Stmt::Return { expr: Some(expr) }],
                }
            }
        }) else {
            unreachable!();
        };

        Ok(Expr::Lambda {
            parameters,
            body: statements,
        })
    }

    fn parse_arguments(&mut self) -> Result<Vec<Expr<'lex>>, ParseError<'lex>> {
        let mut arguments = vec![];

        if let NotFound(_) = chase!(self.tokens, Token::RightParen) {
            arguments.push(self.parse_expression()?);
            while let Found(_) = chase!(self.tokens, Token::Comma) {
                arguments.push(self.parse_expression()?);
            }

            if let NotFound(ctx) = chase!(self.tokens, Token::RightParen) {
                return Err(ParseError::ExpectedParenRight(ctx));
            }
        }

        Ok(arguments)
    }

    fn parse_parameters(&mut self) -> Result<Vec<&'lex str>, ParseError<'lex>> {
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
                return Err(ParseError::ExpectedParenRight(ctx));
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
pub enum Stmt<'lex> {
    Block {
        statements: Block<'lex>,
    },
    Break,
    ClassDecl {
        identifier: &'lex str,
        methods: Vec<(&'lex str, Vec<&'lex str>, Block<'lex>)>,
        superclass: Option<&'lex str>,
    },
    Continue,
    Empty,
    Expr {
        expr: Expr<'lex>,
    },
    FunDecl {
        identifier: &'lex str,
        parameters: Vec<&'lex str>,
        body: Block<'lex>,
    },
    If {
        condition: Expr<'lex>,
        branch: Box<Stmt<'lex>>,
        else_branch: Option<Box<Stmt<'lex>>>,
    },
    Print {
        expr: Expr<'lex>,
    },
    Return {
        expr: Option<Expr<'lex>>,
    },
    VarDecl {
        identifier: &'lex str,
        expr: Option<Expr<'lex>>,
    },
    While {
        condition: Expr<'lex>,
        body: Box<Stmt<'lex>>,
    },
}

impl<'lex> Stmt<'lex> {
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

pub type Block<'lex> = Vec<Stmt<'lex>>;

#[derive(Debug, Clone)]
pub enum Expr<'lex> {
    Assignment {
        op: AssignmentOp,
        identifier: &'lex str,
        expr: Box<Expr<'lex>>,
    },
    Binary {
        left: Box<Expr<'lex>>,
        op: BinaryOp,
        right: Box<Expr<'lex>>,
    },
    FunctionCall {
        expr: Box<Expr<'lex>>,
        arguments: Vec<Expr<'lex>>,
    },
    Grouping {
        expr: Box<Expr<'lex>>,
    },
    Lambda {
        parameters: Vec<&'lex str>,
        body: Block<'lex>,
    },
    Literal(Literal<'lex>),
    PropertyAccess {
        expr: Box<Expr<'lex>>,
        property: &'lex str,
    },
    PropertyAssignment {
        object: Box<Expr<'lex>>,
        property: &'lex str,
        op: AssignmentOp,
        value: Box<Expr<'lex>>,
    },
    Symbol {
        identifier: &'lex str,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr<'lex>>,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError<'lex> {
    #[error("Expected function body instead of {0}.")]
    ExpectedBody(ContextualizedToken<'lex>),

    #[error("Expected function parameter list instead of {0}.")]
    ExpectedFunctionParameterList(ContextualizedToken<'lex>),

    #[error("Expected identifier as parameter instead of {0}.")]
    ExpectedFunctionParameter(ContextualizedToken<'lex>),

    #[error("Expected identifier instead of {0}.")]
    ExpectedIdentifier(ContextualizedToken<'lex>),

    #[error("Expected method in function body instead of {0}.")]
    ExpectedMethod(ContextualizedToken<'lex>),

    #[error("Expected opening parenthesis after `if` instead of {0}.")]
    ExpectedParenLeft(ContextualizedToken<'lex>),

    #[error("Expected closing parenthesis instead of {0}.")]
    ExpectedParenRight(ContextualizedToken<'lex>),

    #[error("Expected a semicolon instead of {0}.")]
    ExpectedSemicolon(ContextualizedToken<'lex>),

    #[error("Expected a symbol on the left-hand side of assignment instead of \"{0:?}\".")]
    InvalidAssignment(Expr<'lex>),

    #[error("Break statement is only valid inside of loops.")]
    StrayBreakStatement,

    #[error("Continue statement is only valid inside of loops.")]
    StrayContinueStatement,

    #[error("Return statement is only valid inside of functions.")]
    StrayReturnStatement,

    #[error("Unexpected token {0}.")]
    UnexpectedToken(ContextualizedToken<'lex>),
}

/// The type returned by the `chase!` macro.
enum Chased<'lex> {
    Found(Token<'lex>),
    NotFound(ContextualizedToken<'lex>),
}

use Chased::*;

/// A literal value in the Lox language.
#[derive(Debug, Copy, Clone)]
pub enum Literal<'lex> {
    Number(LoxNumber),
    String(&'lex str),
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

impl<'lex> From<Token<'lex>> for UnaryOp {
    /// Constructs a `UnaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid unary operator.
    fn from(token: Token<'lex>) -> Self {
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

impl<'lex> From<Token<'lex>> for BinaryOp {
    /// Constructs a `BinaryOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid binary operator.
    fn from(token: Token<'lex>) -> Self {
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

impl<'lex> From<Token<'lex>> for AssignmentOp {
    /// Constructs a `AssignmentOp` from it's equivalent `Token` counterpart.
    /// Panics if the token is not a valid assignment or compound assignment operator.
    fn from(token: Token<'lex>) -> Self {
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

type ParseStmt<'lex> = Result<Stmt<'lex>, ParseError<'lex>>;
type ParseExpr<'lex> = Result<Expr<'lex>, ParseError<'lex>>;
