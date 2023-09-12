use rust_bert::{
    gpt2::GPT2Generator,
    pipelines::{
        conversation::{ConversationManager, ConversationModel},
        generation_utils::{GenerateOptions, LanguageGenerator},
        question_answering::{QaInput, QuestionAnsweringModel},
    },
};
use std::{collections::HashMap, io::stdin};

fn main() {
    let mut knowledge: HashMap<String, String> = [
        ("human name", "Aksel"),
        ("robot identity", "lonely robot"),
        ("human age", "20"),
        ("earths diameter", "69 meters"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();

    let qa = QuestionAnsweringModel::new(Default::default()).unwrap();
    let txt = GPT2Generator::new(Default::default()).unwrap();
    let chat = ConversationModel::new(Default::default()).unwrap();
    let mut chat_man = ConversationManager::new();
    let mut memo_man = ConversationManager::new();

    for question in stdin().lines() {
        println!();
        println!();
        let question = question.unwrap();
        //println!("Stage 1:");
        let context = relevant_context(&qa, &question, &knowledge);
        println!("---  context ---");
        println!("{context}");
        println!("--- \\context ---\n");

        //println!("Stage 2:");
        let context = format!("{context}\n\n{question}");
        let answer = answer(&txt, &context);
        println!("ROBOT:\n{answer}\n");

        let title = conversation_name(&qa, &answer);
        println!(" --- title: {title}");
        knowledge.insert("previous conversation '".to_owned() + &title + "'", answer);
        println!("HUMAN:");
    }
}

fn relevant_context(
    model: &QuestionAnsweringModel,
    question: &str,
    knowledge: &HashMap<String, String>,
) -> String {
    let stage1_context = format!(
        "keys:\n{}",
        knowledge
            .keys()
            .map(|s| format!("'{s}'\n"))
            .collect::<String>()
    );

    let mut answer = model.predict(
        &[QaInput {
            question: format!("Pick the most relevant keys for the prompt '{question}'"),
            context: stage1_context,
        }],
        1,
        32,
    )[0][0]
        .answer
        .clone();

    for (key, val) in knowledge.iter() {
        answer = answer.replace(key, &format!("{key} is {val}."));
    }

    answer
}

fn relevant_context_chat(
    model: &ConversationModel,
    man: &mut ConversationManager,
    question: &str,
    knowledge: &HashMap<String, String>,
) -> String {
    let context = format!(
        "prompt: {question}\nkeys:\n{}\nPick most relevant keys for the prompt",
        knowledge
            .keys()
            .map(|s| format!(" - '{s}'\n"))
            .collect::<String>()
    );

    let id = man.create(&context);

    let mut answer = model.generate_responses(man)[&id].to_string();

    for (key, val) in knowledge.iter() {
        answer = answer.replace(key, &format!("{key} is {val}."));
    }

    answer
}

fn context_gpt(
    model: &GPT2Generator,
    question: &str,
    knowledge: &HashMap<String, String>,
) -> String {
    let context = format!(
        "keys:\n{}\nprompt: {question}\nkeys relevant to prompt:\n",
        knowledge
            .keys()
            .map(|s| format!(" - '{s}'\n"))
            .collect::<String>()
    );

    let settings = GenerateOptions {
        max_length: Some(150),
        ..Default::default()
    };

    model.generate(Some(&[context]), Some(settings))[0]
        .text
        .clone()
}

fn answer(model: &GPT2Generator, context: &str) -> String {
    let settings = GenerateOptions {
        max_length: Some(150),
        ..Default::default()
    };

    model.generate(Some(&[context]), Some(settings))[0]
        .text
        .clone()
}

fn answer_chat(model: &ConversationModel, man: &mut ConversationManager, context: &str) -> String {
    let id = man.create(context);
    model.generate_responses(man)[&id].to_string()
}

fn conversation_name(model: &QuestionAnsweringModel, conversation: &str) -> String {
    model.predict(
        &[QaInput {
            question: "database key for the conversation".to_string(),
            context: conversation.to_string(),
        }],
        1,
        32,
    )[0][0]
        .answer
        .clone()
}
