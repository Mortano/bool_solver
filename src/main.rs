use std::{
    collections::{HashSet, VecDeque},
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::{anyhow, bail, Result};

/// An expression in a prepositional logic formula. This is used to build an expression
/// tree that encodes the formula at runtime
enum Expr {
    Variable(usize),
    Negation(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Implication(Box<Expr>, Box<Expr>),
    Equality(Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Returns a set of unique variable IDs used in this expression. IDs are assigned to all variables during parsing,
    /// so that we don't have to use variable names during evaluation. As an example:
    /// a -> b gets converted to an Expr::Implication(Expr::Variable(0), Expr::Variable(1)), where a gets mapped to ID 0
    /// and b gets mapped to ID 1
    pub fn unique_variables(&self) -> HashSet<usize> {
        match self {
            Expr::Variable(variable_id) => [*variable_id].iter().cloned().collect(),
            Expr::Negation(expr) => expr.unique_variables(),
            Expr::And(left, right) => {
                let left_vars = left.unique_variables();
                let right_vars = right.unique_variables();
                left_vars.union(&right_vars).cloned().collect()
            }
            Expr::Or(left, right) => {
                let left_vars = left.unique_variables();
                let right_vars = right.unique_variables();
                left_vars.union(&right_vars).cloned().collect()
            }
            Expr::Implication(left, right) => {
                let left_vars = left.unique_variables();
                let right_vars = right.unique_variables();
                left_vars.union(&right_vars).cloned().collect()
            }
            Expr::Equality(left, right) => {
                let left_vars = left.unique_variables();
                let right_vars = right.unique_variables();
                left_vars.union(&right_vars).cloned().collect()
            }
        }
    }

    pub fn eval(&self, state: &[bool]) -> bool {
        match self {
            Expr::Variable(id) => state[*id],
            Expr::Negation(expr) => !expr.eval(state),
            Expr::And(left, right) => left.eval(state) && right.eval(state),
            Expr::Or(left, right) => left.eval(state) || right.eval(state),
            // Using equality of a -> b <-> !a OR b
            Expr::Implication(left, right) => !left.eval(state) || right.eval(state),
            Expr::Equality(left, right) => left.eval(state) == right.eval(state),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Variable(id) => write!(f, "{}", id),
            Expr::Negation(expr) => write!(f, "!{}", expr),
            Expr::And(left, right) => write!(f, "({} ^ {})", left, right),
            Expr::Or(left, right) => write!(f, "({} v {})", left, right),
            Expr::Implication(left, right) => write!(f, "({} -> {})", left, right),
            Expr::Equality(left, right) => write!(f, "({} <-> {})", left, right),
        }
    }
}

/// Load all formulas from a text-file, line by line
fn load_formula_lines() -> Result<Vec<String>> {
    let file = File::open("formulas.txt")?;
    let lines = BufReader::new(file)
        .lines()
        .collect::<Result<Vec<_>, _>>()?;
    Ok(lines)
}

/// Converts a variable name, such as 'a', into a variable index, starting from zero
fn variable_to_index(variable: char) -> usize {
    if variable < 'a' || variable > 'z' {
        panic!(
            "Encountered invalid variable '{}'. Variable names must be €[a;z]!",
            variable
        );
    }
    let var_ascii_code = variable as u32;
    (var_ascii_code - ('a'.to_ascii_lowercase() as u32)) as usize
}

/// Parse the given expression from the given token stream
fn parse_expression(token_stream: &mut VecDeque<&str>) -> Result<Expr> {
    // Try to parse all types of expression until something works
    if let Some(var_expr) = parse_variable_expr(token_stream)? {
        return Ok(var_expr);
    }
    if let Some(neg_expr) = parse_negation_expr(token_stream)? {
        return Ok(neg_expr);
    }
    if let Some(binary_expr) = parse_binary_expr(token_stream)? {
        return Ok(binary_expr);
    }

    bail!(
        "Could not parse expression from TokenStream {:?}",
        token_stream
    )
}

fn parse_variable_expr(token_stream: &mut VecDeque<&str>) -> Result<Option<Expr>> {
    if token_stream.is_empty() {
        bail!("TokenStream is empty!")
    }

    let token = token_stream[0];
    if token.len() != 1 {
        return Ok(None);
    }

    let token_as_char = token.chars().nth(0).unwrap();
    if !token_as_char.is_alphabetic() {
        return Ok(None);
    }

    // It is a valid variable! Remove the token from the stream and convert to expression
    token_stream.pop_front();
    let variable_index = variable_to_index(token_as_char);
    return Ok(Some(Expr::Variable(variable_index)));
}

fn parse_negation_expr(token_stream: &mut VecDeque<&str>) -> Result<Option<Expr>> {
    if token_stream.is_empty() {
        bail!("TokenStream is empty!");
    }

    let token = token_stream[0];
    if token != "!" {
        return Ok(None);
    }

    // Pop the first token and try to parse the remaining expression. If this fails, push the
    // first token back into the token_stream and exit
    token_stream.pop_front();
    match parse_expression(token_stream) {
        Ok(expr) => Ok(Some(Expr::Negation(Box::new(expr)))),
        _ => {
            token_stream.push_front(token);
            Ok(None)
        }
    }
}

fn parse_binary_expr(token_stream: &mut VecDeque<&str>) -> Result<Option<Expr>> {
    if token_stream.is_empty() {
        bail!("TokenStream is empty")
    }

    // Create copy of TokenStream because we might need to rollback after parsing the left expression, if some
    // error occurs after parsing the left sub-expression
    // TODO This uses backtracking. In theory, this parser should be able to work without backtracking by looking
    // ahead sufficiently far to figure out what is valid and what not. However then we would have to change the parse
    // functions so that we have a 'try_parse' and then a 'do_parse' so that 'try_parse' does not consume the token stream
    let token_stream_copy = token_stream.clone();

    // All binary expression MUST be enclosed in parens. If the start token is not an opening paren,
    // this can't be a binary expression! This will also mean that things like 'a -> b' will be greedily
    // parsed as just a Expr::Variable for 'a'
    if token_stream[0] != "(" {
        return Ok(None);
    }
    token_stream.pop_front();

    let left_expr = parse_expression(token_stream)?;

    let operator = token_stream[0];
    if operator != "^" && operator != "v" && operator != "->" && operator != "<->" {
        *token_stream = token_stream_copy;
        return Ok(None);
    }
    token_stream.pop_front();
    match parse_expression(token_stream) {
        Ok(right_expr) => {
            let combined_expr = match operator {
                "^" => Expr::And(Box::new(left_expr), Box::new(right_expr)),
                "v" => Expr::Or(Box::new(left_expr), Box::new(right_expr)),
                "->" => Expr::Implication(Box::new(left_expr), Box::new(right_expr)),
                "<->" => Expr::Equality(Box::new(left_expr), Box::new(right_expr)),
                _ => {
                    *token_stream = token_stream_copy;
                    bail!("Invalid operator {} for binary expression", operator)
                }
            };
            token_stream.pop_front();
            Ok(Some(combined_expr))
        }
        Err(why) => {
            *token_stream = token_stream_copy;
            Err(why)
        }
    }
}

fn get_next_char(line: &str) -> Result<char> {
    line.chars()
        .nth(0)
        .ok_or(anyhow!("Invalid first character in line {}", line))
}

/// Tokenize the given line, containing a prepositional logic formula
fn line_to_tokens(line: &str) -> Result<Vec<&str>> {
    let predefined_tokens = vec!["!", "(", ")", "v", "^", "->", "<->"];
    // Token can be one of the predefined tokens, or a letter
    let mut remaining_line = line;
    let mut tokens = vec![];
    while !remaining_line.is_empty() {
        let next_char = get_next_char(remaining_line)?;
        let chars_to_skip = if next_char.is_alphabetic() {
            tokens.push(&remaining_line[0..1]);
            1
        } else if next_char.is_whitespace() {
            1
        } else {
            // Check if we match one of the predefined tokens
            let mut chars_to_trim = None;
            for token in &predefined_tokens {
                if remaining_line.starts_with(token) {
                    let token_len = token.len();
                    tokens.push(&remaining_line[0..token_len]);
                    chars_to_trim = Some(token_len);
                    break;
                }
            }

            match chars_to_trim {
                Some(c) => c,
                None => bail!("Can't determine next token of line {}", remaining_line),
            }
        };

        // Cut of 'chars_to_skip' characters from line
        remaining_line = &remaining_line[chars_to_skip..];
    }

    Ok(tokens)
}

fn parse_statement(statement: &str) -> Result<Expr> {
    // A 'recursive descent' parser seems to be a good way to go

    if statement.is_empty() {
        bail!("Can't parse empty statement!")
    }

    // Split line into a stream of tokens
    let tokens = line_to_tokens(statement)?;
    println!("Tokens: {:?}", tokens);
    let mut token_stream = tokens.into_iter().collect::<VecDeque<_>>();
    let expression = parse_expression(&mut token_stream)?;

    if !token_stream.is_empty() {
        bail!(
            "Could not completely parse statement '{}'. There were leftover tokens after parsing!",
            statement
        )
    }

    Ok(expression)
}

fn main() -> Result<()> {
    let lines = load_formula_lines()?;
    println!("{:?}", lines);
    let expressions = lines
        .iter()
        .map(|line| parse_statement(&line))
        .collect::<Result<Vec<_>, _>>()?;
    let state = vec![false, true, false];
    for expr in expressions {
        println!("{}", expr);
        println!("Evaluated: {}", expr.eval(&state));
    }

    Ok(())
}
