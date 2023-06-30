import { CSVLoader } from "langchain/document_loaders/fs/csv";

const prompt = require('prompt-sync')({ sigint: true });

// https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/csv

// export const run = async () => {

// const loader = new CSVLoader(
//   "/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv",
//   "text"
// );

// const docs = await loader.load();

// console.log(docs);
//   // const loader = new CSVLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv");

//   // const docs = await loader.load();
//   // console.log(docs);

// };

import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";

import { FaissStore } from "langchain/vectorstores/faiss";

import * as fs from "fs";
import { VectorStoreRetriever } from "langchain/dist/vectorstores/base";

import { RetrievalQAChain } from "langchain/chains";

export const run = async () => {
  // Initialize the LLM to use to answer the question.
  const model = new OpenAI({
    temperature: 0,
  });
  // const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
  const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);

  const loader = new CSVLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv");
  const csvDocs = await loader.load();
  console.log(csvDocs.length);
  // console.log(csvDocs[2]);

  csvDocs.forEach((csvDoc: Document<Record<string, any>>) => {
    docs.push(csvDoc);
  })

  // Create a vector store from the documents.
  const vectorStore = await FaissStore.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = RetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      returnSourceDocuments: true,
    }
  );
  // const query: string = "List the mountain bike trails in Bend Oregon that both Ted likes to ride?";
  // const query: string = "How can I debug a node js application from the command line?";
  // const query: string = "How can I debug a node js application?";
  // const query: string = "How can I debug a node js application using vscode?";
  // const query: string = "How can I debug server code?";
  // query: "What did the president say about Justice Breyer?",
  // query: "What is Ted Shaffer's favorite food?",
  // query: "Search the metadata to find the ingredients in the ooni pizza recipe",
  // query: "What is the list the ingredients in the ooni pizza recipe",
  // query: "Tell me about the Tiddlywinks mountain bike trail in Bend."
  // query: "Does Ted like the Tiddlywinks mountain bike trail in Bend."
  // query: "Does Lori ride Tiddlywinks?"
  // query: "What mountain bike trails does Ted like to ride?"
  // query: "Who likes to ride Storm King?"
  // query: "What mountain bike trails does Morgan like?"
  // query: "List the mountain bike trails in Bend Oregon that both Joel and Ted like to ride?"
  // query: "List the mountain bike trails in Bend Oregon that both Morgan likes to ride?"

  while (true) {
    const query = prompt('Enter your query: ');
    const res = await chain.call({
      query
    });
    console.log("Answer: " + res.text);
  }
  
  /*
  {
    res: {
      text: 'The president said that Justice Breyer was an Army veteran, Constitutional scholar,
      and retiring Justice of the United States Supreme Court and thanked him for his service.'
    }
  }
  */
};
