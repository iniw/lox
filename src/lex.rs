use std::{fmt::Display, iter::Peekable, str::CharIndices};

use crate::LoxNumber;

#[derive(Debug, Clone)]
pub struct Lexer<'src> {
    source_data: &'src str,
    source: Peekable<CharIndices<'src>>,
    line: u32,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Lexer {
            source_data: source,
            source: source.char_indices().peekable(),
            line: 1,
        }
    }

    pub fn lex(mut self) -> (Vec<ContextualizedToken<'src>>, Vec<LexError<'src>>) {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        while let Some((start, c)) = self.source.next() {
            // The line counter can be modified inside `process_char` so save it here before calling it
            let line = self.line;

            match self.process_char(start, c) {
                Ok(Some(token)) => {
                    // Retrieve the lexeme from the source data, extracting either the token span or the remaining span if the pushed token is the last
                    let lexeme = if let Some((end, _)) = self.source.peek() {
                        &self.source_data[start..*end]
                    } else {
                        &self.source_data[start..]
                    };

                    tokens.push(ContextualizedToken {
                        token,
                        lexeme,
                        line,
                    });
                }
                Ok(None) => {}
                Err(error) => {
                    errors.push(error);
                }
            }
        }

        tokens.push(ContextualizedToken {
            token: Token::EndOfFile,
            lexeme: "",
            line: self.line,
        });

        (tokens, errors)
    }

    fn process_char(
        &mut self,
        start: usize,
        c: char,
    ) -> Result<Option<Token<'src>>, LexError<'src>> {
        match c {
            // Single characters
            '(' => Ok(Some(Token::LeftParen)),
            ')' => Ok(Some(Token::RightParen)),
            '{' => Ok(Some(Token::LeftBrace)),
            '}' => Ok(Some(Token::RightBrace)),
            ',' => Ok(Some(Token::Comma)),
            '.' => Ok(Some(Token::Dot)),
            ';' => Ok(Some(Token::Semicolon)),

            // One to two characters
            '+' if self.chase('=') => Ok(Some(Token::PlusEqual)),
            '+' => Ok(Some(Token::Plus)),

            '-' if self.chase('=') => Ok(Some(Token::MinusEqual)),
            '-' => Ok(Some(Token::Minus)),

            '*' if self.chase('=') => Ok(Some(Token::StarEqual)),
            '*' => Ok(Some(Token::Star)),

            '!' if self.chase('=') => Ok(Some(Token::BangEqual)),
            '!' => Ok(Some(Token::Bang)),

            '=' if self.chase('=') => Ok(Some(Token::EqualEqual)),
            '=' => Ok(Some(Token::Equal)),

            '<' if self.chase('=') => Ok(Some(Token::LessEqual)),
            '<' => Ok(Some(Token::Less)),

            '>' if self.chase('=') => Ok(Some(Token::GreaterEqual)),
            '>' => Ok(Some(Token::Greater)),

            // Single-line comment
            '/' if self.chase('/') => {
                self.consume_until_delimiter('\n');
                Ok(None)
            }
            // Multi-line/nested comment
            '/' if self.chase('*') => {
                let mut nesting = 1;
                while nesting > 0 {
                    match self.source.next() {
                        Some((_, '/')) if self.chase('*') => nesting += 1,
                        Some((_, '*')) if self.chase('/') => nesting -= 1,
                        Some(_) => continue,
                        None => break,
                    }
                }
                Ok(None)
            }

            '/' if self.chase('=') => Ok(Some(Token::SlashEqual)),
            '/' => Ok(Some(Token::Slash)),

            // String literals
            '"' => {
                // Consume until we find the closing quote
                while let Some((_, c)) = self.source.next_if(|(_, c)| *c != '"') {
                    // Allow multilined strings
                    if c == '\n' {
                        self.line += 1;
                    }
                }

                if let Some((end, _)) = self.source.next() {
                    let str = &self.source_data[start + 1..end];
                    Ok(Some(Token::String(str)))
                } else {
                    Err(LexError::UnterminatedStringLiteral(
                        &self.source_data[start..],
                    ))
                }
            }

            // Numeric literals
            c if c.is_ascii_digit() => match self.consume_while(char::is_ascii_digit) {
                // Floating point
                Some((_, '.')) => {
                    // Consume the dot
                    self.source.next();

                    if let Some((end, c)) = self.source.peek() {
                        if !c.is_ascii_alphanumeric() {
                            return Err(LexError::UnterminatedFloatingLiteral(
                                &self.source_data[start..*end],
                            ));
                        }
                    } else {
                        return Err(LexError::UnterminatedFloatingLiteral(
                            &self.source_data[start..],
                        ));
                    }

                    let number = if let Some((end, _)) = self.consume_while(char::is_ascii_digit) {
                        &self.source_data[start..end]
                    } else {
                        &self.source_data[start..]
                    };

                    if let Ok(number) = number.parse::<LoxNumber>() {
                        Ok(Some(Token::Number(number)))
                    } else {
                        Err(LexError::FailureParsingNumber(number))
                    }
                }
                ending => {
                    let number = if let Some((end, _)) = ending {
                        &self.source_data[start..end]
                    } else {
                        &self.source_data[start..]
                    };

                    if let Ok(number) = number.parse::<LoxNumber>() {
                        Ok(Some(Token::Number(number)))
                    } else {
                        Err(LexError::FailureParsingNumber(number))
                    }
                }
            },

            // Newlines.
            // The whitespace check in the arm below catches the newline symbol so it has to be handled before said check.
            '\n' => {
                self.line += 1;
                Ok(None)
            }

            // Whitespace
            c if c.is_whitespace() => Ok(None),

            // Identifiers | Keywords
            c if Self::is_valid_for_identifier(&c) => {
                let identifier =
                    if let Some((end, _)) = self.consume_while(Self::is_valid_for_identifier) {
                        &self.source_data[start..end]
                    } else {
                        &self.source_data[start..]
                    };

                Ok(self
                    .as_keyword(identifier)
                    .or(Some(Token::Identifier(identifier))))
            }

            // Unknown character
            _ => Err(LexError::UnexpectedCharacter(c)),
        }
    }

    fn chase(&mut self, expected: char) -> bool {
        self.source.next_if(|(_, c)| *c == expected).is_some()
    }

    fn consume_until_delimiter(&mut self, delimiter: char) {
        while self.source.next_if(|(_, c)| *c != delimiter).is_some() {}
    }

    fn consume_while(&mut self, f: impl Fn(&char) -> bool) -> Option<(usize, char)> {
        while self.source.next_if(|(_, c)| f(c)).is_some() {}
        self.source.peek().copied()
    }

    fn as_keyword(&mut self, text: &str) -> Option<Token<'src>> {
        match text {
            "and" => Some(Token::And),
            "class" => Some(Token::Class),
            "else" => Some(Token::Else),
            "false" => Some(Token::False),
            "for" => Some(Token::For),
            "fun" => Some(Token::Fun),
            "if" => Some(Token::If),
            "nil" => Some(Token::Nil),
            "or" => Some(Token::Or),
            "print" | "pedrao" => Some(Token::Print), // Easter egg
            "return" => Some(Token::Return),
            "super" => Some(Token::Super),
            "this" => Some(Token::This),
            "true" => Some(Token::True),
            "var" => Some(Token::Var),
            "while" => Some(Token::While),
            "break" => Some(Token::Break),
            "continue" => Some(Token::Continue),
            _ => None,
        }
    }

    fn is_valid_for_identifier(c: &char) -> bool {
        c.is_ascii_alphanumeric() || *c == '_'
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Token<'src> {
    // One character
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    Bang,
    Equal,
    Greater,
    Less,
    // Two characters
    BangEqual,
    EqualEqual,
    GreaterEqual,
    LessEqual,
    PlusEqual,
    MinusEqual,
    SlashEqual,
    StarEqual,
    // Literals
    Identifier(&'src str),
    String(&'src str),
    Number(LoxNumber),
    // Keywords
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    Break,
    Continue,
    EndOfFile,
}

#[derive(Debug, Copy, Clone)]
pub struct ContextualizedToken<'src> {
    pub token: Token<'src>,
    pub lexeme: &'src str,
    pub line: u32,
}

impl<'src> Display for ContextualizedToken<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lexeme = if let Token::EndOfFile = self.token {
            "EOF"
        } else {
            self.lexeme
        };

        write!(f, "\"{}\" @ line {}", lexeme, self.line)
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum LexError<'src> {
    #[error("Unterminated string literal \"{0}\".")]
    UnterminatedStringLiteral(&'src str),

    #[error("Unterminated floating point literal \"{0}\".")]
    UnterminatedFloatingLiteral(&'src str),

    #[error("Number literal \"{0}\" failed to parse.")]
    FailureParsingNumber(&'src str),

    #[error("Unexpected character \"{0}\".")]
    UnexpectedCharacter(char),
}
