// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";

// export const run = async () => {
//   const vectorStore = await FaissStore.fromTexts(
//     ["Hello world", "Bye bye", "hello nice world"],
//     [{ id: 2 }, { id: 1 }, { id: 3 }],
//     new OpenAIEmbeddings()
//   );

//   const resultOne = await vectorStore.similaritySearch("hello world", 1);
//   console.log(resultOne);
// };

import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";

export const run = async () => {
  // Create docs with a loader
  const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
  const docs = await loader.load();

  // Load the docs into the vector store
  const vectorStore = await FaissStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );

  // Search for the most similar document
  const resultOne = await vectorStore.similaritySearch("hello world", 1);
  console.log(resultOne);
};

