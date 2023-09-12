#![feature(test)]
mod spatial_hash_grid;

use std::io::Write;

use anyhow::*;
use markdown;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use tch::{nn, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NodeData {
    text: String,
    siblings: Vec<isize>,
    position: String,
}

impl NodeData {
    fn new(text: String) -> Self {
        Self {
            text,
            siblings: vec![-1],
            position: String::new(),
        }
    }
}

#[derive(Default)]
struct VectorDB {
    embeddings: Vec<Tensor>,
    data: Vec<NodeData>,
}

#[derive(Serialize, Deserialize, Default)]
struct SerializedVectorDB {
    data: Vec<NodeData>,
}

// mean squared error
fn distance(a: &Tensor, b: &Tensor) -> f32 {
    let d = ((a-b).pow_tensor_scalar(2.).sum(None));
    //panic!("{:?}", a.transpose(0, 1).size());
    //d.double_value(&[]) as f32
}

// Split by line
// If line is longer than 200 characters, split by '.'
// If sentence is longer than 200 characters, split by ','
fn paragraphs(text: &str) -> Vec<&str> {
    let mut paragraphs = vec![text];
    //for line in text.split("\n\n") {
    //    if line.len() > 400 {
    //        for sentence in line.split('.') {
    //            if sentence.len() > 400 {
    //                for clause in sentence.split(',') {
    //                    paragraphs.push(clause);
    //                }
    //            } else {
    //                paragraphs.push(sentence);
    //            }
    //        }
    //    } else {
    //        paragraphs.push(line);
    //    }
    //}

    // filter out empty paragraphs
    paragraphs.into_iter().filter(|p| !p.is_empty() && p.matches(" ").count() >= 2 && p.len() > 5).collect()
}

fn collect_mdast(ast: markdown::mdast::Node) -> Vec<NodeData> {
    use markdown::mdast::Node::*;
    match ast {
        Root(root) => {
            let mut text = vec![];
            for child in root.children {
                text.extend(collect_mdast(child));
            }
            text
        }
        Paragraph(paragraph) => {
            let mut text = vec![];
            for child in paragraph.children {
                //let mut children = collect_mdast(child);
                //children.iter_mut().for_each(|child| child.position = format!("{:?}", paragraph.position));
                text.extend(collect_mdast(child));
            }
            text
        }
        Heading(heading) => {
            let mut text = vec![];
            for child in heading.children {
                let mut children = collect_mdast(child);
                for (n, child) in children.iter_mut().enumerate() {
                    child.siblings.extend([- (n as isize) - text.len() as isize]);
                }
                text.extend(children);
            }
            text
        }
        Emphasis(emphasis) => {
            let mut text = vec![];
            for child in emphasis.children {
                text.extend(collect_mdast(child));
            }
            text
        }
        List(list) => {
            let mut text = vec![];
            for child in list.children {
                text.extend(collect_mdast(child));
            }
            text
        }

        Text(text) => paragraphs(&text.value).into_iter().map(|text| NodeData::new(text.to_string()) ).collect(),
        _ => vec![],
    }.into_iter().enumerate().map(|(n, mut node)| {
        node.siblings.iter_mut().for_each(|sibling| *sibling += n as isize);
        node
    }).collect()
}

impl VectorDB {
    fn save(&self, path: &str) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let serialized = SerializedVectorDB {
            data: self.data.clone(),
        };
        for (n, embedding) in self.embeddings.iter().enumerate() {
            embedding.save(format!("{}.pt", self.data[n].position))?;
        }
        serde_yaml::to_writer(&mut file, &serialized)?;
        Ok(())
    }

    fn load(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let serialized: SerializedVectorDB = serde_yaml::from_reader(file)?;
        let mut embeddings = vec![];

        for (n, data) in serialized.data.iter().enumerate() {
            let embedding = Tensor::load(format!("{}.pt", data.position))?;
            embeddings.push(embedding);
        }

        Ok(Self {
            embeddings,
            data: serialized.data,
        })
    }

    fn semantic_search(&self, query: &Tensor) -> Vec<&NodeData> {
        //let mut min = 0;
        //let mut d = std::f32::INFINITY;
        //let mut embed = vec![];
        //for entry in &self.embeddings {
        //    embed.push(entry);
        //}
        //for (n, _) in self.embeddings.iter().enumerate() {
        //    let mut cum = embed[n].clone();
        //    for sibling in self.data[n].siblings.iter() {
        //        if sibling < &0 || sibling >= &(self.embeddings.len() as isize){
        //            continue;
        //        }
        //        cum.iter_mut().zip(self.embeddings[*sibling as usize].iter()).for_each(|(a, b)| *a += b / 2.0);
        //    }

        //    //cum.iter_mut().for_each(|a| *a /= self.data[n].siblings.len() as f32);

        //    let d2 = distance(&query, &cum);
        //    if d > d2 {
        //        min = n;
        //        d = d2;
        //    }
        //}
        let mut ret = vec![];

        let mut dist = self
            .embeddings
            .iter()
            .map(|embed| distance(query, embed))
            .enumerate()
            .collect::<Vec<_>>();

        dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (n, _) in dist.iter().take(5) {
            ret.push(&self.data[*n]);
        }

        return ret;
    }

    //fn extractive_qa(&self, query: &str, model: QuestionAnsweringModel) -> Result<Vec<&NodeData>> {
    //    let mut score = 0.;
    //    let mut node = 0;
    //    let mut nodes = vec![];

    //    for (n, data) in self.data.iter().enumerate() {
    //        let input = QaInput {
    //            question: query.to_string(),
    //            context: data.text.clone(),
    //        };
    //        let output = model.predict(&[input], 1, 32);
    //        for output in output.iter().flatten() {
    //            if output.score > score {
    //                score = output.score;
    //                node = n;
    //                nodes.push(data);
    //            }
    //        }
    //    }
    //    let top5 = (nodes.len()-5..nodes.len());
    //    Ok(nodes[top5].to_vec())
    //}

    fn from_archive(archive_dir: &str, model: &SentenceEmbeddingsModel) -> Result<Self> {
        let mut db = VectorDB::default();
        let files = std::fs::read_dir(archive_dir)?
            .map(|res| res.map(|e| e.path()))

            .collect::<Result<Vec<_>, std::io::Error>>()?;
        for (n, file) in files.iter().enumerate() {
            print!("Reading {archive_dir:?}... {}%\r", n*100 / files.len());
            std::io::stdout().flush().expect("Failed to flush stdout");
            let text = std::fs::read_to_string(&file)?;
            let mut text = collect_mdast(
                match markdown::to_mdast(&text, &markdown::ParseOptions::default()) {
                    core::result::Result::Ok(ast) => ast,
                    Err(e) => bail!("Failed to parse markdown: {}", e),
                },
            );

            if text.is_empty() {
                continue;
            }

            let embeddings = model.encode_as_tensor(
                &text
                    .iter()
                    .map(|node| node.text.as_str())
                    .collect::<Vec<_>>(),
            )?;
            let offset = db.embeddings.len();
            for (n, node) in text.iter_mut().enumerate() {
                db.embeddings.push(embeddings.embeddings.slice(0, n as i64, (n + 1) as i64, 1));
                node.siblings.iter_mut().for_each(|sibling| *sibling += offset as isize);
                node.position = format!("{}", file.display());
                db.data.push(node.clone());
            }
        }
        Ok(db)
    }
}

fn main() -> Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    if std::env::args().len() <= 1 {
        println!("BRUH");
        VectorDB::from_archive("archive", &model)?.save("db.yaml")?;
        println!("Wrote db.yaml          ");
        return Ok(());
    };

    println!("{}", Tensor::load("archive/xmas.org.md.pt")?);

    let db = VectorDB::load("db.yaml")?;

    let q = model.encode_as_tensor(&std::env::args().skip(1).collect::<Vec<_>>())?.embeddings;
    let node = db.semantic_search(&q);

    println!();
    for node in node.iter() {
        println!("{}\t:: {}", node.position, node.text);
    }
    println!("Done");
    Ok(())
}

#[test]
fn embeddings_size(){
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model().unwrap();
    let emb = model.encode_as_tensor(&["Hello world"]).unwrap().embeddings;
    println!("{:?}", emb.size());
    println!("{:?}", emb.double_value(&[0, 2]));
}
