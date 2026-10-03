//! Typed command tree derived from the Teamy Rust CLI template.
pub mod benchmark;
pub mod command;
pub mod decide;
pub mod global_args;
pub mod health;
pub mod interactive;
pub mod model;
pub mod model_selection;
pub mod output;
pub mod prompt;
pub mod rank;
pub mod serve;
pub mod top_completions;
pub mod top_tokens;

pub use command::Command;
use facet::Facet;
use figue as args;
pub use global_args::GlobalArgs;

/// Local Rust inference for text generation and finite choices.
///
/// TEAMY_LLM_SERVICE_HOME_DIR overrides configuration/registry storage;
/// TEAMY_LLM_SERVICE_CACHE_DIR overrides the model artifact cache. RUST_LOG
/// supplies logging when no explicit --log-filter/--log-level or --debug is given.
/// Diagnostic logging stays on standard error, independently of JSON stdout.
#[derive(Debug, Facet)]
pub struct Cli {
    #[facet(flatten)]
    pub global_args: GlobalArgs,
    #[facet(flatten)]
    pub builtins: figue::FigueBuiltins,
    #[facet(args::subcommand)]
    pub command: Command,
}

#[derive(Facet)]
struct InheritedHelp {
    #[facet(flatten)]
    global_args: GlobalArgs,
    #[facet(flatten)]
    builtins: figue::FigueBuiltins,
}

/// Parse without installing process-global logging, signals or runtime hooks.
pub fn parse(arguments: impl IntoIterator<Item = impl Into<String>>) -> figue::DriverOutcome<Cli> {
    let mut arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    normalize_help_aliases(&mut arguments);
    if arguments.is_empty() {
        arguments.push("--help".into());
    }
    let short_help_list = arguments.iter().any(|argument| argument == "--help=short");
    let config = figue::builder::<Cli>()
        .expect("Teamy LLM CLI schema should be valid")
        .cli(|cli| cli.args(arguments).strict())
        .help(|help| {
            help.program_name("teamy-llm")
                .version(crate::version())
                .include_implementation_source_file(true)
                .include_implementation_github_url(
                    "TeamDman/teamy-llm-service",
                    option_env!("GIT_REVISION").unwrap_or("main"),
                )
        })
        .build();
    match figue::Driver::new(config).run().into_result() {
        Ok(output) => figue::DriverOutcome::ok(output),
        Err(figue::DriverError::Help {
            suggestion: Some(report),
            ..
        }) => {
            // Figue offers corrective usage when required arguments are missing,
            // but its Help outcome exits successfully. Preserve the generated
            // missing-argument diagnostic as a failure; requested help remains
            // successful because it has no corrective suggestion.
            figue::DriverOutcome::err(figue::DriverError::Failed { report })
        }
        Err(figue::DriverError::Help {
            mut text,
            suggestion,
        }) => {
            // This frozen Figue revision parses inherited named options, but its
            // leaf help renders only that leaf's schema. Generate the missing
            // global block from the actual GlobalArgs schema, never a second
            // manually maintained list of flags or leaf usage strings.
            if !short_help_list && !text.contains("--log-filter") {
                let globals = figue::generate_help::<InheritedHelp>(&figue::HelpConfig {
                    program_name: Some("teamy-llm <command>".into()),
                    ..Default::default()
                });
                text.push_str("\nGlobal options accepted before or after subcommands:\n\n");
                text.push_str(&globals);
            }
            figue::DriverOutcome::err(figue::DriverError::Help { text, suggestion })
        }
        Err(error) => figue::DriverOutcome::err(error),
    }
}

fn normalize_help_aliases(arguments: &mut [String]) {
    // Derive value-taking flags from the same reflected types that Figue parses.
    // Help aliases never rewrite a prompt/model/filter value or anything after --.
    let mut value_flags = std::collections::HashSet::new();
    collect_value_flags(Cli::SHAPE, &mut value_flags);
    let mut cursor = 0;
    while cursor < arguments.len() {
        let argument = &arguments[cursor];
        if argument == "--" {
            break;
        }
        if value_flags.contains(argument) {
            cursor += 2;
            continue;
        }
        if matches!(argument.as_str(), "?" | "-?") {
            arguments[cursor] = "--help".into();
        }
        cursor += 1;
    }
    if arguments
        .first()
        .is_some_and(|argument| argument == "version")
    {
        arguments[0] = "--version".into();
    }
}

fn collect_value_flags(
    shape: &'static facet::Shape,
    flags: &mut std::collections::HashSet<String>,
) {
    use facet::{Def, ScalarType, Type, UserType};
    let shape = match shape.def {
        Def::Option(option) => option.t,
        _ => shape,
    };
    match shape.ty {
        Type::User(UserType::Struct(structure)) => {
            for field in structure.fields {
                if field.is_flattened() || field.has_attr(Some("args"), "subcommand") {
                    collect_value_flags(field.shape(), flags);
                } else if field.has_attr(Some("args"), "named") {
                    let value = match field.shape().def {
                        Def::Option(option) => option.t,
                        _ => field.shape(),
                    };
                    if value.scalar_type() == Some(ScalarType::Bool) {
                        continue;
                    }
                    flags.insert(format!("--{}", field.effective_name().replace('_', "-")));
                    for attribute in field.attributes {
                        if attribute.ns == Some("args")
                            && attribute.key == "alias"
                            && let Some(figue::Attr::Alias(alias)) =
                                attribute.get_as::<figue::Attr>()
                        {
                            flags.insert(format!("--{}", alias.replace('_', "-")));
                        }
                    }
                }
            }
        }
        Type::User(UserType::Enum(enumeration)) => {
            for variant in enumeration.variants {
                for field in variant.data.fields {
                    collect_value_flags(field.shape(), flags);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::model::ModelCommand;
    use crate::cli::model::default::ModelDefaultCommand;
    use crate::cli::output::OutputFormat;
    use eyre::Result;
    use teamy_llm_core::ThinkingMode;

    fn parsed(arguments: &[&str]) -> Result<Cli> {
        Ok(parse(arguments.iter().copied()).into_result()?.value)
    }
    #[test]
    fn generated_help_accepts_aliases_at_root_and_leaf_without_invocation() {
        for arguments in [
            vec!["--help"],
            vec!["help"],
            vec!["-h"],
            vec!["-?"],
            vec!["?"],
            vec!["decide", "--help"],
            vec!["decide", "help"],
            vec!["decide", "?"],
            vec!["model", "default", "show", "--help"],
            vec!["model", "default", "set", "-?"],
            vec!["interactive", "--help"],
        ] {
            let figue::DriverError::Help {
                text, suggestion, ..
            } = parse(arguments).unwrap_err()
            else {
                panic!("expected help")
            };
            assert!(suggestion.is_none());
            assert!(text.contains("teamy-llm"));
            assert!(text.contains("--log-filter"), "{text}");
            assert!(text.contains("--stop-after-duration"), "{text}");
        }
    }
    #[test]
    fn missing_required_arguments_fail_while_requested_help_succeeds() {
        for arguments in [vec!["decide", "--prompt"], vec!["model", "default", "set"]] {
            assert!(
                matches!(
                    parse(arguments).unwrap_err(),
                    figue::DriverError::Failed { .. }
                ),
                "missing required arguments must not become a successful help outcome"
            );
        }
        for arguments in [
            vec!["decide", "--provider", "native", "--help"],
            vec!["model", "default", "set", "--help"],
            vec!["model"],
        ] {
            assert!(parse(arguments).unwrap_err().is_help());
        }
    }
    #[test]
    fn literal_separator_preserves_flag_like_prompt_and_alias_words() {
        let cli = parsed(&[
            "prompt",
            "--",
            "--debug",
            "--model",
            "x",
            "?",
            "help",
            "--version",
        ])
        .unwrap();
        assert!(!cli.global_args.debug);
        assert!(!cli.builtins.version);
        let Command::Prompt(prompt) = cli.command else {
            panic!("prompt")
        };
        assert!(prompt.selection.model.is_none());
        assert_eq!(
            prompt.prompt,
            ["--debug", "--model", "x", "?", "help", "--version"]
        );
    }
    #[test]
    fn help_alias_normalization_preserves_flag_values() {
        for value in ["?", "-?"] {
            let cli = parsed(&[
                "prompt",
                "--system-prompt",
                value,
                "--model",
                value,
                "--",
                "ordinary question",
            ])
            .unwrap();
            let Command::Prompt(prompt) = cli.command else {
                panic!("prompt")
            };
            assert_eq!(prompt.system_prompt.as_deref(), Some(value));
            assert_eq!(prompt.selection.model.as_deref(), Some(value));
            assert_eq!(prompt.prompt, ["ordinary question"]);
            assert!(!cli.builtins.help);
        }
        let cli = parsed(&[
            "prompt",
            "--system-prompt=?",
            "--model=?",
            "ordinary question",
        ])
        .unwrap();
        let Command::Prompt(prompt) = cli.command else {
            panic!("prompt")
        };
        assert_eq!(prompt.system_prompt.as_deref(), Some("?"));
        assert_eq!(prompt.selection.model.as_deref(), Some("?"));
    }
    #[test]
    fn inherited_globals_and_filter_alias_are_typed_once() {
        let cli = parsed(&[
            "model",
            "default",
            "show",
            "--log-level",
            "teamy_llm=trace",
            "--log-file",
            "capture.ndjson",
            "--stop-after-duration",
            "2s",
            "--output-format",
            "json",
        ])
        .unwrap();
        assert_eq!(
            cli.global_args.log_filter.as_deref(),
            Some("teamy_llm=trace")
        );
        assert_eq!(
            cli.global_args.log_file.unwrap().to_string_lossy(),
            "capture.ndjson"
        );
        assert_eq!(
            cli.global_args.stop_after.stop_after_duration.as_deref(),
            Some("2s")
        );
        assert_eq!(cli.global_args.output_format, Some(OutputFormat::Json));
    }
    #[test]
    fn default_selection_has_unambiguous_set_and_show_commands() {
        let cli = parsed(&["model", "default", "set", "qwen3.8-27b-uncensored-q4-k-m"]).unwrap();
        let Command::Model(model) = cli.command else {
            panic!("model")
        };
        let ModelCommand::Default(default) = model.command else {
            panic!("default")
        };
        let ModelDefaultCommand::Set(set) = default.command else {
            panic!("set")
        };
        assert_eq!(set.model, "qwen3.8-27b-uncensored-q4-k-m");
        assert!(parsed(&["model", "default", "set"]).is_err());
        assert!(parsed(&["model", "default", "show", "unexpected"]).is_err());
    }
    #[test]
    fn benchmark_omission_differs_from_explicit_no_thinking() {
        for (arguments, expected) in [
            (
                vec!["benchmark"],
                vec![ThinkingMode::Direct, ThinkingMode::Thinking],
            ),
            (
                vec!["benchmark", "--thinking"],
                vec![ThinkingMode::Thinking],
            ),
            (
                vec!["benchmark", "--no-thinking"],
                vec![ThinkingMode::Direct],
            ),
        ] {
            let Command::Benchmark(options) = parsed(&arguments).unwrap().command else {
                panic!("benchmark")
            };
            assert_eq!(options.thinking_modes().unwrap(), expected);
        }
        let Command::Benchmark(conflict) =
            parsed(&["benchmark", "--both-thinking-modes", "--no-thinking"])
                .unwrap()
                .command
        else {
            panic!("benchmark")
        };
        assert!(conflict.thinking_modes().is_err());
        assert!(parsed(&["benchmark", "--thinking", "--no-thinking"]).is_err());
    }
    #[test]
    fn duplicate_and_unknown_arguments_fail_instead_of_becoming_prompt_text() {
        for arguments in [
            vec!["prompt", "--image", "image.png", "Question"],
            vec!["prompt", "--model", "one", "--model", "two", "Question"],
            vec!["prompt", "--max-new-tokens"],
            vec!["prompt", "--timeout-ms", "invalid", "Question"],
            vec!["model", "list", "--unknown"],
            vec!["model", "show", "--model"],
            vec!["source.txt", "destination.txt"],
        ] {
            assert!(parsed(&arguments).is_err(), "{arguments:?}");
        }
        let Command::Prompt(prompt) =
            parsed(&["prompt", "--model", "one", "--model-dir", "two", "Question"])
                .unwrap()
                .command
        else {
            panic!("prompt")
        };
        assert!(prompt.selection.to_selector().is_err());
    }
    #[test]
    fn decide_request_sources_remain_literal_data() {
        for request in [
            "@review.json",
            "-",
            r#"{"version":1,"input":{"text":"example"}}"#,
        ] {
            let Command::Decide(options) = parsed(&[
                "decide",
                "--provider",
                "native",
                "--model-dir",
                "models",
                "--request",
                request,
            ])
            .unwrap()
            .command
            else {
                panic!("decide")
            };
            assert_eq!(options.request.as_deref(), Some(request));
            options.validate().unwrap();
        }
    }

    #[test]
    fn decide_repeated_choices_preserve_literals_duplicates_and_input_order() {
        let values = [
            "?",
            "-?",
            "help",
            "--help",
            "--debug",
            "--version",
            "--",
            "café 日本語",
            "line one\nline two",
            "key=value",
            "?",
        ];
        let mut arguments = vec!["decide", "--prompt", "ordinary question"];
        for value in values {
            arguments.extend(["--choice", value]);
        }
        let cli = parsed(&arguments).unwrap();
        assert!(!cli.global_args.debug);
        assert!(!cli.builtins.help);
        assert!(!cli.builtins.version);
        let Command::Decide(options) = cli.command else {
            panic!("decide")
        };
        assert_eq!(options.prompt.as_deref(), Some("ordinary question"));
        assert_eq!(options.choice, values.map(String::from));
        assert!(options.model_dir.is_none());
        assert_eq!(
            options.provider,
            crate::cli::decide::DecisionProvider::Native
        );
        options.validate().unwrap();
    }

    #[test]
    fn decide_named_prompt_values_and_inline_choices_remain_literal_data() {
        for value in [
            "?",
            "-?",
            "help",
            "--help",
            "--debug",
            "--version",
            "--",
            "café 日本語\nsecond line",
        ] {
            let cli =
                parsed(&["decide", "--prompt", value, "--choice=?", "--choice=--help"]).unwrap();
            assert!(!cli.global_args.debug);
            assert!(!cli.builtins.help);
            assert!(!cli.builtins.version);
            let Command::Decide(options) = cli.command else {
                panic!("decide")
            };
            assert_eq!(options.prompt.as_deref(), Some(value));
            assert_eq!(options.choice, ["?", "--help"]);
            options.validate().unwrap();
        }
    }

    #[test]
    fn decide_rejects_duplicate_prompt_and_unknown_flags() {
        for arguments in [
            vec![
                "decide", "--prompt", "first", "--prompt", "second", "--choice", "a", "--choice",
                "b",
            ],
            vec![
                "decide",
                "--prompt",
                "question",
                "--choice",
                "a",
                "--choice",
                "b",
                "--unknown",
            ],
        ] {
            assert!(parsed(&arguments).is_err(), "{arguments:?}");
        }
    }
    #[test]
    fn model_inspect_alias_and_interactive_configuration_are_typed() {
        assert!(matches!(
            parsed(&["model", "inspect", "--model", "example"])
                .unwrap()
                .command,
            Command::Model(_)
        ));
        let Command::Interactive(options) = parsed(&[
            "interactive",
            "--model",
            "qwen3.8-27b-uncensored-q4-k-m",
            "--decision-provider",
            "onnx",
            "--decision-model-dir",
            "julia",
            "--runtime-library",
            "runtime.dll",
            "--allow-cpu-fallback",
            "--script",
            "session.jsonl",
        ])
        .unwrap()
        .command
        else {
            panic!("interactive")
        };
        assert_eq!(
            options.script_file.unwrap().to_string_lossy(),
            "session.jsonl"
        );
        assert_eq!(
            options.decision_provider,
            Some(crate::cli::decide::DecisionProvider::Onnx)
        );
        assert!(options.allow_cpu_fallback);
    }
    #[test]
    fn version_and_empty_invocation_are_success_outcomes() {
        for args in [
            vec!["--version"],
            vec!["version"],
            vec!["model", "list", "--version"],
        ] {
            let figue::DriverError::Version { text } = parse(args).unwrap_err() else {
                panic!("version")
            };
            assert!(text.contains("repo"));
            assert!(text.contains("rev"));
            assert!(text.contains("built"));
        }
        assert!(parse(Vec::<String>::new()).unwrap_err().is_help());
    }
}
