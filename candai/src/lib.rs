#![allow(unused)]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use candle_core::quantized::gguf_file;
use candle_core::utils;
use candle_core::{ Device, Tensor };
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as quantized_model;

use anyhow::Result;

mod output;
use output::TokenOutputStream;

fn main() -> anyhow::Result<()> {
    let mut file = File::open(&PathBuf::from("./hf_hub/openchat_3.5.Q8_0.gguf"))?;
    let model = gguf_file::Content::read(&mut file)?;
    let mut model = quantized_model::ModelWeights::from_gguf(model, &mut file)?;

    let tokenizer = Tokenizer::from_file(PathBuf::from("./hf_hub/openchat_3.5_tokenizer.json")).map_err(anyhow::Error::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt = format!("User: 人工智能有哪些应用场景 <|end_of_turn|> Assistant: ");
    let tokens = tos.tokenizer().encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut all_tokens = vec![];
    let mut processor = LogitsProcessor::new(299792458, Some(0.8), None);
    let prompt_tokens = tokens.get_ids();
    let mut next_token = {
        let input = Tensor::new(prompt_tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        processor.sample(&logits)?
    };
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("t:{t}");
        std::io::stdout().flush()?;
    }

    let eos_token = "<|end_of_turn|>";
    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    let to_sample = 1000_usize.saturating_sub(1);
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let start_at = all_tokens.len().saturating_sub(64);
        let logits = candle_transformers::utils::apply_repeat_penalty(
            &logits,
            1.1,
            &all_tokens[start_at..],
        )?;
        next_token = processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("t:{t}");
            std::io::stdout().flush()?;
        }
        if next_token == eos_token {
            break;
        };
    }
    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("r:{rest}");
        std::io::stdout().flush()?;
    }

    Ok(())
}