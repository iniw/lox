use lox::*;

fn main() -> Result<(), rustyline::error::ReadlineError> {
    let mut rl = rustyline::DefaultEditor::new()?;

    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let (tokens, errors) = lex::Lexer::new(&line).lex();
                if !errors.is_empty() {
                    eprintln!("Lexing errors:");
                    for error in errors {
                        eprintln!("- {error}");
                    }
                    eprintln!();
                }

                let (statements, errors) = syntax::Parser::new(&tokens).parse();
                if !errors.is_empty() {
                    eprintln!("Parsing errors:");
                    for error in errors {
                        eprintln!("- {error}");
                    }
                    eprintln!();
                }

                match rt::TreeWalker::new().execute(statements) {
                    Ok(rt::Stated::Nothing) => {}
                    Ok(value) => println!("{value}"),
                    Err(error) => eprintln!("{error}"),
                }
            }
            Err(error) => {
                println!("Bye! ({error})");
                break;
            }
        }
    }

    Ok(())
}
