use lox::*;
use std::{env, fs, io::Read, time};

fn main() {
    // Skip the program name
    let mut args = env::args().skip(1);
    match args.next() {
        Some(file_name) => {
            let mut file = fs::File::open(file_name).expect("File not found");

            let mut buffer = String::new();
            file.read_to_string(&mut buffer)
                .expect("Failed to read file");

            let start = time::Instant::now();
            let (tokens, errors) = lex::Lexer::new(&buffer).lex();
            println!("[Lexing took: {:?}]", start.elapsed());

            if !errors.is_empty() {
                eprintln!("Lexing errors:");
                for error in errors {
                    eprintln!("- {error}");
                }
                eprintln!();
            }

            let start = time::Instant::now();
            let (statements, errors) = syntax::Parser::new(&tokens).parse();
            println!("[Parsing took: {:?}]", start.elapsed());

            if !errors.is_empty() {
                eprintln!("Parsing errors:");
                for error in errors {
                    eprintln!("- {error}");
                }
                eprintln!();
            }

            let start = time::Instant::now();
            if let Err(e) = rt::TreeWalker::new().execute(statements) {
                eprintln!("[RUNTIME ERROR] {}", e);
            } else {
                println!("[Interpreting took: {:?}]", start.elapsed());
            }
        }
        None => {
            println!("Usage: lox [script]");
        }
    }
}
