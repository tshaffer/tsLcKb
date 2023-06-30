import * as fs from "fs";

import { OpenAI } from "langchain/llms/openai";
import { Document } from "langchain/document";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { ConversationalRetrievalQAChain, RetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";

// const prompt = require('prompt-sync')({ sigint: true });

// export const run = async () => {
//   const model = new OpenAI({
//     temperature: 0,
//   });
//   const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
//   // const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt", "utf8");
//   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
//   const docs = await textSplitter.createDocuments([text]);

//   const loader = new CSVLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv");
//   const csvDocs = await loader.load();
//   console.log(csvDocs.length);

//   csvDocs.forEach((csvDoc: Document<Record<string, any>>) => {
//     docs.push(csvDoc);
//   })

//   const vectorStore = await FaissStore.fromDocuments(docs, new OpenAIEmbeddings());

//   const chain = ConversationalRetrievalQAChain.fromLLM(
//     model,
//     vectorStore.asRetriever(),
//     {
//       memory: new BufferMemory({
//         memoryKey: "chat_history", // Must be set to "chat_history"
//       }),
//     }
//   );
//   // const chain = RetrievalQAChain.fromLLM(
//   //   model,
//   //   vectorStore.asRetriever(),
//   //   {
//   //     returnSourceDocuments: true,
//   //   }
//   // );
//   // const query: string = "List the mountain bike trails in Bend Oregon that both Ted likes to ride?";
//   // const query: string = "How can I debug a node js application from the command line?";
//   // const query: string = "How can I debug a node js application?";
//   // const query: string = "How can I debug a node js application using vscode?";
//   // const query: string = "How can I debug server code?";
//   // query: "What did the president say about Justice Breyer?",
//   // query: "What is Ted Shaffer's favorite food?",
//   // query: "Search the metadata to find the ingredients in the ooni pizza recipe",
//   // query: "What is the list the ingredients in the ooni pizza recipe",
//   // query: "Tell me about the Tiddlywinks mountain bike trail in Bend."
//   // query: "Does Ted like the Tiddlywinks mountain bike trail in Bend."
//   // query: "Does Lori ride Tiddlywinks?"
//   // query: "What mountain bike trails does Ted like to ride?"
//   // query: "Who likes to ride Storm King?"
//   // query: "What mountain bike trails does Morgan like?"
//   // query: "List the mountain bike trails in Bend Oregon that both Joel and Ted like to ride?"
//   // query: "List the mountain bike trails in Bend Oregon that both Morgan likes to ride?"

//   while (true) {
//     const query = prompt('Enter your query: ');
//     const res = await chain.call({
//       query
//     });
//     console.log("Answer: " + res.text);
//   }
// };

import { HNSWLib } from "langchain/vectorstores/hnswlib";

export const run = async () => {
  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({});
  /* Load in the file we want to do question answering over */
  // const text = fs.readFileSync("state_of_the_union.txt", "utf8");
  // const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
  const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt", "utf8");    /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  /* Create the chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      memory: new BufferMemory({
        memoryKey: "chat_history", // Must be set to "chat_history"
      }),
    }
  );
  /* Ask it a question */
  const question = "What mountain bike trails does Morgan like?";
  const res = await chain.call({ question });
  console.log(res);
  /* Ask it a follow up question */
  const followUpRes = await chain.call({
    question: "Does Lori ride Tiddlywinks?",
  });
  console.log(followUpRes);
};