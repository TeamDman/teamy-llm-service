//! Bounded text-only single-turn rendering for the pinned OrcaRouter Qwen3.8 template.
//!
//! Source: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored/blob/8cb32d72080f6a47bf34dc5adf8067daa6c63a31/tokenizer_config.json>.
//! This helper has no message-history, tool or visual-input surface. It does not implement the
//! publisher's complete multimodal/tool template or grant authority to execute generated text.

use teamy_llm_core::ThinkingMode;

const DEFAULT_THINKING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

/// Render system + user + assistant generation prefix with the publisher's default reasoning
/// effort. Direct mode corresponds to `enable_thinking=false`; Thinking corresponds to
/// `enable_thinking=true` with its default `reasoning_effort=xhigh`.
///
/// No default reasoning instruction is added in Direct mode. Caller-provided text is trimmed in
/// the same way as the bounded publisher-template fixtures; structural token text is not escaped.
pub(crate) fn render_orca_qwen_single_turn_prompt(
    system_prompt: Option<&str>,
    user_prompt: &str,
    thinking_mode: ThinkingMode,
) -> String {
    let system = system_prompt.map(str::trim).filter(|text| !text.is_empty());
    let thinking = matches!(thinking_mode, ThinkingMode::Thinking);
    let mut prompt = String::new();
    if thinking || system.is_some() {
        prompt.push_str("<|im_start|>system\n");
        if thinking {
            prompt.push_str(DEFAULT_THINKING_INSTRUCTION);
            if system.is_some() {
                prompt.push_str("\n\n");
            }
        }
        if let Some(system) = system {
            prompt.push_str(system);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user_prompt.trim());
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    match thinking_mode {
        ThinkingMode::Direct => prompt.push_str("<think>\n\n</think>\n\n"),
        ThinkingMode::Thinking => prompt.push_str("<think>\n"),
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::render_orca_qwen_single_turn_prompt;
    use teamy_llm_core::ThinkingMode;

    // Expected bytes were independently produced by the pinned publisher Jinja template through
    // Transformers5.0.0, not by this Rust renderer. qwen_chat_fixtures.json retains token arrays,
    // source hashes and reference provenance for the same four requests.
    const USER: &str = "  Hello, café and 日本語.\n";
    const SYSTEM: &str = "  You are concise.\n";

    #[test]
    fn direct_without_system_matches_publisher_fixture() {
        assert_eq!(
            render_orca_qwen_single_turn_prompt(None, USER, ThinkingMode::Direct),
            "<|im_start|>user\nHello, café and 日本語.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn direct_with_system_matches_publisher_fixture() {
        assert_eq!(
            render_orca_qwen_single_turn_prompt(Some(SYSTEM), USER, ThinkingMode::Direct),
            "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHello, café and 日本語.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn thinking_without_system_matches_publisher_default_xhigh_fixture() {
        assert_eq!(
            render_orca_qwen_single_turn_prompt(None, USER, ThinkingMode::Thinking),
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHello, café and 日本語.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn thinking_with_system_matches_publisher_default_xhigh_fixture() {
        assert_eq!(
            render_orca_qwen_single_turn_prompt(Some(SYSTEM), USER, ThinkingMode::Thinking),
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are concise.<|im_end|>\n<|im_start|>user\nHello, café and 日本語.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn blank_system_is_absent_and_direct_mode_does_not_add_thinking_instruction() {
        for mode in [ThinkingMode::Direct, ThinkingMode::Thinking] {
            assert_eq!(
                render_orca_qwen_single_turn_prompt(Some(" \n\t"), USER, mode),
                render_orca_qwen_single_turn_prompt(None, USER, mode)
            );
        }
        assert!(
            !render_orca_qwen_single_turn_prompt(None, USER, ThinkingMode::Direct)
                .contains("Reasoning effort")
        );
    }
}
