use lox::*;
use std::{env, fs, io::Read, time};

fn run_file(file_name: &str) {
    let mut file = fs::File::open(file_name).expect("File not found");

    let mut buffer = String::new();
    file.read_to_string(&mut buffer)
        .expect("Failed to read file");

    let start = time::Instant::now();
    let lexer = lex::Lexer::new(&buffer);
    let (tokens, errors) = lexer.lex();
    println!("Lexing took: {:?}", start.elapsed());

    if !errors.is_empty() {
        dbg!(errors);
    }

    dbg!(&tokens);

    let start = time::Instant::now();
    let parser = ast::Parser::new(&tokens);
    let (statements, errors) = parser.parse();
    println!("Parsing took: {:?}", start.elapsed());

    if !errors.is_empty() {
        dbg!(errors);
    }
    dbg!(&statements);

    let start = time::Instant::now();
    let mut interpreter = rt::TreeWalker::new();
    let result = interpreter.execute(statements);
    println!("Interpreting took: {:?}", start.elapsed());
    match result {
        Ok(result) => {
            println!("Successfully interpreted! Result = {:?}", result);
        }
        Err(e) => {
            eprintln!("[RUNTIME ERROR] {}", e);
        }
    }
}

fn main() {
    // Skip the program name
    let mut args = env::args().skip(1);
    match args.next() {
        Some(arg) => run_file(&arg),
        None => {
            println!("Usage: lox [script]");
        }
    }
}
