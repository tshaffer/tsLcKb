import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

export const run = async () => {
  try {

    // const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
    // const docs = await loader.load();

    //split text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 150,
      chunkOverlap: 20,
    });

    const text = fs.readFileSync(
      require.resolve("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt"),
      "utf8"
    );

    const texts = await textSplitter.createDocuments([text])
    console.log('Number of documents created from splitter: ', texts.length);

    console.log('Preview');
    console.log(texts[0].pageContent + '\n');
    console.log(texts[1].pageContent + '\n');


  } catch (error) {
    console.log("error", error);
  }
};





// Create a new index from texts
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

// Create a new index from a loader
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import { TextLoader } from "langchain/document_loaders/fs/text";

// export const run = async () => {
//   // Create docs with a loader
//   const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
//   const docs = await loader.load();

//   // Load the docs into the vector store
//   const vectorStore = await FaissStore.fromDocuments(
//     docs,
//     new OpenAIEmbeddings()
//   );

//   // Search for the most similar document
//   const resultOne = await vectorStore.similaritySearch("hello world", 1);
//   console.log(resultOne);
// };

// Save an index to file and load it again
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";

// export const run = async () => {

//   // Create a vector store through any method, here from texts as an example
//   const vectorStore = await FaissStore.fromTexts(
//     ["Hello world", "Bye bye", "hello nice world"],
//     [{ id: 2 }, { id: 1 }, { id: 3 }],
//     new OpenAIEmbeddings()
//   );

//   // Save the vector store to a directory
//   const directory = "/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src";

//   await vectorStore.save(directory);

//   // Load the vector store from the same directory
//   const loadedVectorStore = await FaissStore.load(
//     directory,
//     new OpenAIEmbeddings()
//   );

//   // vectorStore and loadedVectorStore are identical
//   const result = await loadedVectorStore.similaritySearch("hello world", 1);
//   console.log(result);
// }